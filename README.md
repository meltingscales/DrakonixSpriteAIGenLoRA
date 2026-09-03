# DrakonixSpriteAIGenLoRA

Trained LoRA weights: [huggingface.co/henryfbp/DrakonixSpriteAIGenLoRA](https://huggingface.co/henryfbp/DrakonixSpriteAIGenLoRA)

Sibling project to [DrakonixSpriteAIGen](../DrakonixSpriteAIGen) — that repo
is a from-scratch class-conditional VAE (pick a label, get a sprite). This
repo is the answer to "what if we want to generate sprites *from text
descriptions*?": fine-tune an existing pretrained text-to-image model with
[LoRA](https://arxiv.org/abs/2106.09685) instead of training text
understanding from scratch. From-scratch text-to-image needs a captioned
dataset and a text encoder built from nothing — LoRA reuses a model that
already knows what "dragon" or "knight" looks like, and only teaches it a
new *style* (pixel art) cheaply.

## the plan

- **Base model**: Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`
  via [diffusers](https://github.com/huggingface/diffusers)). Smallest/
  oldest capable SD version — lowest VRAM/disk, and most existing pixel-art
  LoRA tutorials and reference LoRAs target it.
- **Training stack**: `diffusers` + [`peft`](https://github.com/huggingface/peft)
  for the LoRA layers, `accelerate` for the training loop, `transformers`
  for the CLIP tokenizer/text encoder that ships with SD1.5.
- **What we don't have yet**: a captioned pixel-art dataset. This is the
  actual blocker, not the training code. LoRA needs (image, caption) pairs;
  DrakonixSpriteAIGen's Kaggle data has only 5 numeric classes, no text.

## what we learned from DrakonixSpriteAIGen that applies here

- **GPU detection can lie.** `torch.cuda.is_available()` returned `True` on
  our ROCm 6.4 / gfx1151 (Strix Halo) setup but segfaulted on the first
  real op. Ported `device.py`'s subprocess-probe pattern as-is — it's
  fully generic, not VAE-specific.
- **LoRA training is realistically GPU-only.** Unlike the tiny 16x16 VAE
  (which trained fine on CPU in minutes), SD1.5 LoRA training is a full
  diffusion U-Net forward/backward at 512x512 — CPU training would take
  hours to days. If `get_device()` reports `cpu` here, that's a hard
  blocker, not just "slower" — check `nvidia-smi`/`rocminfo` before
  starting a training run.
- **Working ROCm PyTorch build found (2026-09-03).** `torch==2.14.0+rocm7.2`
  from `https://download.pytorch.org/whl/rocm7.2` (Python 3.11) reports
  GPU correctly, and — unlike the ROCm 6.4 build — doesn't segfault:
  verified with both `device.py`'s trivial-op probe and a heavier
  conv2d+backward / attention+backward workload closer to actual U-Net
  shape. `pyproject.toml` is pinned to this build via `[tool.uv.sources]`.
  See AI-NOTES.md for the version-pinning details and why plain `uv add
  torch` doesn't get you here.

## dataset requirements (from current LoRA training practice, 2026)

- **Image count**: 15–30 images is a commonly cited working range for a
  style LoRA (not a per-character concept LoRA — we want "pixel art
  style", not one specific character).
- **Resolution**: match SD1.5's native 512x512 (512x768/768x512 also
  used). Our sprites are natively 8x8/16x16 — upscale with **nearest-
  neighbor** (not bilinear/bicubic) to preserve the blocky pixel-art look;
  smooth upscaling defeats the point.
- **Captions**: every training image needs a caption/tag file (e.g.
  `sprite_001.txt` next to `sprite_001.png`) describing its content, plus
  a consistent trigger word (e.g. `pixelart style`) so the LoRA associates
  that phrase with the learned style at generation time. Caption quality
  matters more than volume here.
- **Training params** (SD1.5 LoRA, typical starting point): learning rate
  1e-4–2e-4, 1500–2500 steps depending on dataset size, batch size 1 (8GB
  VRAM) to 2–4 (12GB+), clip skip 2. Watch sample previews during training
  — stop once generated previews start looking identical to training
  images rather than chasing a fixed step count.

Sources: [Stable Diffusion Art — How to train LoRA models](https://stable-diffusion-art.com/train-lora/),
[SeaArt — dataset creation guide](https://docs.seaart.ai/guide-1/3-advanced-guide/3-2-lora-training-advance/how-to-create-dataset-for-training)

## status

All five original build steps are done: dataset (`just data`, 17 curated
images — see AI-NOTES.md for sourcing/licensing details), training
(`just train` / the GUI's Train tab, `src/drakonix_lora/train.py`), and
inference (GUI's Generate tab, loads a cached base pipeline + swaps in
the selected LoRA). Verified end-to-end with a real 1500-step checkpoint
— generated output visibly carries the trained pixel-art style (flat
shading, black outlines, blocky palette).

Open threads, not blockers:
- Only the UNet is trained (text encoder frozen) — reasonable first cut,
  but text-encoder LoRA is a lever if style transfer feels too shallow.
- No inference-side upscale/palette-snap post-processing yet (see the
  "dataset requirements" section above on why that matters for pixel
  art).
- Hyperparameters (rank 16, lr 1e-4, clip skip 2) are untuned defaults,
  not the result of a sweep.

## what's already ported from DrakonixSpriteAIGen

- `justfile` — `just setup` (uv sync), `just data` (fetch dataset),
  `just train`, `just gui`, `just test`.
- `pyproject.toml` — uv-managed, dependencies swapped for the LoRA stack
  (`diffusers`, `transformers`, `accelerate`, `peft` instead of `kaggle`).
- `src/drakonix_lora/device.py` — GPU-probe-with-subprocess pattern,
  reused verbatim (see "what we learned" above).
- `src/drakonix_lora/gui.py` — Gradio Blocks structure (Train tab +
  Generate tab, checkpoint/weights dropdown with Refresh + on-page-load
  refresh) ported from the same pattern, fields adapted to LoRA
  training/inference params.
- `tests/test_gui.py` — smoke test that the GUI builds, same pattern as
  the original repo's approach of actually exercising code rather than
  just reading it.

## setup

```
just setup
just data    # fetch + prepare the curated dataset
```

## training

```
just train                      # default hyperparameters, 1500 steps
just train data/captions runwayml/stable-diffusion-v1-5 10   # quick smoke test
```

Saves a `.safetensors` LoRA checkpoint to `lora_weights/`. Re-running with
identical params refuses (pass `--force` via the CLI, or check "Force
retrain" in the GUI) rather than silently redoing the same work.

## GUI

```
just gui
```

Train and Generate tabs are both live.
