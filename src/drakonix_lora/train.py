"""Train a pixel-art style LoRA on top of a base Stable Diffusion model.

Usage:
    python -m drakonix_lora.train --captions-dir data/captions --steps 1500
    python -m drakonix_lora.train --steps 10 --force  # quick smoke test

STATUS: first working version — UNet-only LoRA (attention projections),
fp32, step-based training loop. Text encoder is frozen; only the UNet
learns the style. See README.md "dataset requirements" for where the
default hyperparameters below come from.
"""

import argparse
import hashlib
import itertools
import re
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from .dataset import PixelArtCaptionDataset
from .device import get_device

LORA_DIR = Path("lora_weights")
DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
UNET_LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]

# Use the text encoder's second-to-last hidden layer instead of its last —
# a common setting for stylized/anime-adjacent LoRA training (see README.md
# "dataset requirements"). Not exposed as a parameter: nothing in this
# project varies it, so it isn't config, it's a decision.
CLIP_SKIP = 2


def dataset_fingerprint(captions_dir: str) -> str:
    """8 hex chars summarizing dataset contents, so a rerun on the same
    (filename, size) pairs is recognized as the same dataset even if the
    directory path differs, and a changed dataset gets a new checkpoint
    name instead of silently colliding with an old one."""
    root = Path(captions_dir)
    entries = sorted((p.name, p.stat().st_size) for p in root.glob("*.png"))
    digest = hashlib.sha256(repr(entries).encode()).hexdigest()
    return digest[:8]


def checkpoint_label(base_model: str, captions_dir: str, rank: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_model).strip("-").lower()
    return f"{slug}_r{rank}_{dataset_fingerprint(captions_dir)}"


def default_checkpoint_path(label: str, steps: int) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return LORA_DIR / f"{label}_{steps}steps_{stamp}.safetensors"


def find_matching_checkpoints(label: str, steps: int) -> list[Path]:
    """Existing checkpoints trained with these exact params — a rerun would
    reproduce (approximately) the same result."""
    if not LORA_DIR.exists():
        return []
    prefix = f"{label}_{steps}steps_"
    return sorted(
        LORA_DIR.glob(f"{prefix}*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _encode_prompts(
    tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, captions: list[str], device
) -> torch.Tensor:
    tokens = tokenizer(
        list(captions),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    output = text_encoder(tokens, output_hidden_states=True)
    hidden_states = output.hidden_states[-CLIP_SKIP]
    # this transformers version exposes final_layer_norm directly on
    # CLIPTextModel rather than nested under a .text_model submodule
    return text_encoder.final_layer_norm(hidden_states)


def run_training(
    base_model: str,
    captions_dir: str,
    rank: int,
    steps: int,
    out_path: Path,
    lr: float = 1e-4,
    batch_size: int = 1,
    on_step: Callable[[int, int, float], None] | None = None,
) -> Path:
    device = get_device()

    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.eval()
    vae.eval()

    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=UNET_LORA_TARGET_MODULES,
        )
    )
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    dataset = PixelArtCaptionDataset(captions_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batches = itertools.cycle(loader)

    unet.train()
    for step in range(1, steps + 1):
        pixel_values, captions = next(batches)
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            encoder_hidden_states = _encode_prompts(tokenizer, text_encoder, captions, device)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        loss = F.mse_loss(model_pred.float(), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if on_step:
            on_step(step, steps, loss.item())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=out_path.parent,
        unet_lora_layers=get_peft_model_state_dict(unet),
        weight_name=out_path.name,
        safe_serialization=True,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-dir", type=str, default="data/captions")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    label = checkpoint_label(args.base_model, args.captions_dir, args.rank)
    existing = find_matching_checkpoints(label, args.steps)
    if existing and not args.force:
        parser.error(
            f"a checkpoint with these exact params already exists: {existing[0]} "
            "— pass --force to retrain anyway"
        )

    out_path = default_checkpoint_path(label, args.steps)
    print(f"using device: {get_device()}")

    def on_step(step: int, total: int, loss: float) -> None:
        if step % 10 == 0 or step == total:
            print(f"step {step}/{total}  loss={loss:.4f}")

    run_training(
        base_model=args.base_model,
        captions_dir=args.captions_dir,
        rank=args.rank,
        steps=args.steps,
        out_path=out_path,
        lr=args.lr,
        batch_size=args.batch_size,
        on_step=on_step,
    )
    print(f"saved LoRA weights to {out_path}")


if __name__ == "__main__":
    main()
