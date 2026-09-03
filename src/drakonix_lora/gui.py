"""GUI for training a pixel-art LoRA and generating sprites from a prompt.

Usage:
    python -m drakonix_lora.gui

STATUS: both tabs are wired up — training via `train.run_training`,
generation via a cached `StableDiffusionPipeline` + the selected LoRA
weights.
"""

import queue
import re
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

from .device import get_device
from .postprocess import pixelate, upscale_for_viewing
from .train import (
    DEFAULT_BASE_MODEL,
    LORA_DIR,
    checkpoint_label,
    default_checkpoint_path,
    find_matching_checkpoints,
    run_training,
)

# Cache across calls: reloading the ~4GB base pipeline on every click would
# make the Generate tab unusably slow. Keyed on what's currently loaded so
# switching LoRA files swaps just the adapter, not the whole pipeline, and
# switching base models only reloads when the base model actually changes.
_state: dict = {"pipeline": None, "base_model": None, "lora_path": None}

OUTPUTS_DIR = Path("outputs")

# Sentinel for "generate straight from the base model, no LoRA" — lets
# trying a different base model (e.g. a community pixel-art checkpoint)
# stand on its own instead of forcing one of our own trained LoRAs onto it.
NO_LORA = "(none — base model only)"

# Single global cancel flag — fine for this app's single-operator local-tool
# scope (same assumption _state already makes). Set by the Cancel button,
# checked from inside the denoising callback so a cancel can interrupt
# mid-generation, not just between queued prompts.
_cancel_event = threading.Event()


class _Cancelled(Exception):
    """Raised inside the denoising callback to unwind out of a running
    pipeline() call when the user hits Cancel — not a real error."""


EXAMPLE_PROMPTS = [
    ["pixelart style, a fire-breathing dragon"],
    ["pixelart style, a knight in armor holding a sword"],
    ["pixelart style, a frog wizard casting a spell"],
    ["pixelart style, a small cottage in a forest"],
    ["pixelart style, a shark wearing an eyepatch"],
    ["pixelart style, a strawberry with a cute face"],
]


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


class _StepTimer:
    """Rolling ETA from recent inter-tick deltas, not total elapsed time —
    total-elapsed would fold in one-time setup (model loading) and badly
    skew the estimate, especially for short runs."""

    def __init__(self, window: int = 20):
        self._window = window
        self._deltas: list[float] = []
        self._last: float | None = None

    def tick(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            self._deltas.append(now - self._last)
            del self._deltas[: -self._window]
        self._last = now

    def eta(self, remaining: int) -> str:
        if not self._deltas:
            return "estimating…"
        avg = sum(self._deltas) / len(self._deltas)
        return _format_duration(avg * remaining)


def _render_queue_status(prompt_queue: list[str], current_idx: int, current_desc: str) -> str:
    lines = []
    for i, prompt in enumerate(prompt_queue):
        short = prompt if len(prompt) <= 50 else prompt[:47] + "..."
        if i < current_idx:
            lines.append(f"done      {short}")
        elif i == current_idx:
            lines.append(f"active >  {short}  ({current_desc})")
        else:
            lines.append(f"queued    {short}")
    return "\n".join(lines)


def list_lora_weights() -> list[str]:
    if not LORA_DIR.exists():
        return []
    paths = sorted(LORA_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in paths]


def lora_choices() -> list[str]:
    """Newest-trained-first, with the no-LoRA sentinel always available
    last so it never displaces a real checkpoint as the default choice."""
    return list_lora_weights() + [NO_LORA]


def list_outputs() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return []
    paths = sorted(OUTPUTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in paths]


def _save_output(image, prompt: str, index: int) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", prompt).strip("-").lower()[:40] or "sprite"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{slug}_{stamp}_{index}.png"
    image.save(path)
    return path


def refresh_lora_weights(current: str):
    choices = lora_choices()
    value = current if current in choices else choices[0]
    return gr.Dropdown(choices=choices, value=value)


def train(
    base_model: str,
    captions_dir: str,
    rank: int,
    steps: int,
    force: bool,
    progress=gr.Progress(),
):
    label = checkpoint_label(base_model, captions_dir, rank)

    existing = find_matching_checkpoints(label, steps)
    if existing and not force:
        names = ", ".join(p.name for p in existing[:3])
        raise gr.Error(
            f"a checkpoint with these exact params already exists ({names}) — "
            "training again would waste time reaching the same result. "
            "Check 'Force retrain' to do it anyway."
        )

    out_path = default_checkpoint_path(label, steps)
    timer = _StepTimer()

    def on_step(step: int, total: int, loss: float) -> None:
        timer.tick()
        eta = timer.eta(total - step)
        progress(step / total, desc=f"step {step}/{total}  loss={loss:.4f}  ETA {eta}")

    run_training(
        base_model=base_model,
        captions_dir=captions_dir,
        rank=rank,
        steps=steps,
        out_path=out_path,
        on_step=on_step,
    )

    return f"saved {out_path.name}", gr.Dropdown(choices=lora_choices(), value=str(out_path))


def _ensure_pipeline(base_model: str) -> StableDiffusionPipeline:
    if _state["pipeline"] is None or _state["base_model"] != base_model:
        # SD1.5's bundled safety checker false-positives heavily on
        # non-photorealistic content — flat-shaded pixel art with
        # skin-tone-ish color blocks routinely trips it, returning a black
        # image instead of the render. This is local, offline generation of
        # sprite art, not a hosted service serving untrusted users, so
        # disabling it here is the standard fix rather than a workaround.
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model, safety_checker=None, requires_safety_checker=False
        )
        pipeline = pipeline.to(get_device())
        _state["pipeline"] = pipeline
        _state["base_model"] = base_model
        # a LoRA trained against a different base's weights is meaningless
        # here — force it to be reloaded (or left off) against the new one
        _state["lora_path"] = None
    return _state["pipeline"]


def _ensure_lora(pipeline: StableDiffusionPipeline, lora_path: str) -> None:
    target = lora_path if lora_path and lora_path != NO_LORA else None
    if _state["lora_path"] == target:
        return
    if _state["lora_path"] is not None:
        pipeline.unload_lora_weights()
    if target is not None:
        lora_file = Path(target)
        pipeline.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)
    _state["lora_path"] = target


def _decode_preview(
    pipeline: StableDiffusionPipeline,
    latents: torch.Tensor,
    pixelate_enabled: bool,
    sprite_size: int,
    palette_colors: int,
):
    """Quick VAE decode of the first image in the current batch's
    in-progress latents, for a live "watch it draw" preview. Only the
    first image (not the whole batch) to keep the extra decode cheap.
    When pixelation is on, applies it here too so the preview shows what
    the sprite is actually converging toward, not just the raw render."""
    with torch.no_grad():
        scaled = latents[:1] / pipeline.vae.config.scaling_factor
        decoded = pipeline.vae.decode(scaled).sample
    image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
    if pixelate_enabled:
        sprite = pixelate(image, sprite_size, palette_colors)
        image = upscale_for_viewing(sprite, display_size=image.size[0])
    return image


def cancel_generation() -> str:
    _cancel_event.set()
    return "cancelling…"


def generate(
    base_model: str,
    lora_path: str,
    prompts_text: str,
    guidance: float,
    num_steps: int,
    batch_size: int,
    pixelate_enabled: bool,
    sprite_size: int,
    palette_colors: int,
):
    """Generator so the live-preview image and progress bar can stream
    mid-generation. diffusers' step callback is a plain synchronous nested
    function — it can't `yield` through this function's frame — so the
    pipeline runs in a background thread and hands preview frames back
    through a queue for this generator to yield.

    Progress is rendered through our own Slider + Textbox rather than
    gr.Progress() — the latter's built-in overlay re-triggers on every
    yield (one per denoising step here) and visually fights with the
    live-preview image update, causing a flash. Plain output components
    don't have that overlay."""
    prompt_queue = [line.strip() for line in prompts_text.splitlines() if line.strip()]
    if not prompt_queue:
        raise gr.Error("no prompts — enter at least one line")

    pipeline = _ensure_pipeline(base_model)
    _ensure_lora(pipeline, lora_path)

    num_steps = int(num_steps)
    batch_size = int(batch_size)
    sprite_size = int(sprite_size)
    palette_colors = int(palette_colors)
    # diffusers invokes the step callback num_inference_steps + 1 times per
    # generation (verified empirically, not documented) — account for that
    # in the total so the ETA/progress fraction stay accurate and the
    # display doesn't show e.g. "step 16/15".
    calls_per_prompt = num_steps + 1
    total_calls = len(prompt_queue) * calls_per_prompt
    timer = _StepTimer()
    completed_calls = 0
    current_prompt_idx = 0
    events: queue.Queue = queue.Queue()

    _cancel_event.clear()

    def on_denoise_step(pipe, step, timestep, callback_kwargs):
        nonlocal completed_calls
        if _cancel_event.is_set():
            raise _Cancelled()
        completed_calls += 1
        timer.tick()
        eta = timer.eta(total_calls - completed_calls)
        desc = f"step {min(step + 1, num_steps)}/{num_steps}  ETA {eta}"
        preview = _decode_preview(
            pipeline, callback_kwargs["latents"], pixelate_enabled, sprite_size, palette_colors
        )
        queue_text = _render_queue_status(prompt_queue, current_prompt_idx, desc)
        events.put(("step", completed_calls / total_calls, desc, preview, queue_text))
        return callback_kwargs

    def run() -> None:
        nonlocal current_prompt_idx
        images = []
        try:
            for current_prompt_idx, prompt in enumerate(prompt_queue):
                if _cancel_event.is_set():
                    raise _Cancelled()
                result = pipeline(
                    prompt,
                    guidance_scale=guidance,
                    num_inference_steps=num_steps,
                    num_images_per_prompt=batch_size,
                    callback_on_step_end=on_denoise_step,
                )
                for i, image in enumerate(result.images):
                    if pixelate_enabled:
                        image = pixelate(image, sprite_size, palette_colors)
                    _save_output(image, prompt, i)
                    images.append(
                        upscale_for_viewing(image, 512) if pixelate_enabled else image
                    )
            events.put(("done", images, None, None))
        except _Cancelled:
            events.put(("cancelled", images, None, None))
        except Exception as exc:  # re-raised on the main thread below
            events.put(("error", exc, None, None))

    threading.Thread(target=run, daemon=True).start()

    # disable Generate / enable Cancel immediately, don't wait for the
    # first denoising step (model loading alone can take a few seconds)
    yield (
        gr.update(),
        gr.update(),
        "starting…",
        gr.update(),
        0,
        _render_queue_status(prompt_queue, 0, "starting…"),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )

    while True:
        kind, a, b, c, *rest = events.get()
        if kind == "step":
            frac, desc, preview, queue_text = a, b, c, rest[0]
            yield (
                preview,
                gr.update(),
                desc,
                gr.update(),
                int(frac * 100),
                queue_text,
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
        elif kind == "error":
            yield (
                gr.update(),
                gr.update(),
                f"error: {a}",
                gr.update(),
                0,
                gr.update(),
                gr.update(interactive=True),
                gr.update(interactive=False),
            )
            raise gr.Error(f"generation failed: {a}")
        elif kind == "cancelled":
            images = a
            status = f"cancelled — generated {len(images)} image(s) before stopping"
            yield (
                gr.update(),
                images,
                status,
                list_outputs(),
                0,
                "cancelled",
                gr.update(interactive=True),
                gr.update(interactive=False),
            )
            return
        else:  # done
            images = a
            status = f"generated {len(images)} image(s) from {len(prompt_queue)} prompt(s)"
            yield (
                gr.update(),
                images,
                status,
                list_outputs(),
                100,
                "done",
                gr.update(interactive=True),
                gr.update(interactive=False),
            )
            return


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Drakonix Sprite LoRA") as app:
        gr.Markdown("# Drakonix Sprite LoRA")
        with gr.Tab("Train LoRA"):
            base_model = gr.Textbox(label="Base model", value=DEFAULT_BASE_MODEL)
            captions_dir = gr.Textbox(label="Captioned dataset dir", value="data/captions")
            rank = gr.Slider(1, 128, value=16, step=1, label="LoRA rank")
            steps = gr.Slider(100, 5000, value=1500, step=100, label="Training steps")
            force = gr.Checkbox(
                label="Force retrain (ignore existing checkpoint with same params)",
                value=False,
            )
            train_btn = gr.Button("Start training", variant="primary")
            train_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Generate"):
            gen_base_model = gr.Textbox(
                label="Base model",
                value=DEFAULT_BASE_MODEL,
                info=(
                    "Any SD1.5-compatible HF repo — try a community pixel-art "
                    "model like PublicPrompts/All-In-One-Pixel-Model here "
                    "(pair it with 'no LoRA' below, its own trigger words are "
                    "'pixelsprite' / '16bitscene')"
                ),
            )
            with gr.Row():
                # explicit default (newest-trained-first, so this is our
                # own most recent checkpoint) rather than relying on
                # Gradio's implicit "select the first choice" behavior
                _lora_choices = lora_choices()
                lora_path = gr.Dropdown(
                    label="LoRA weights", choices=_lora_choices, value=_lora_choices[0]
                )
                refresh_btn = gr.Button("Refresh", scale=0)
            prompts = gr.Textbox(
                label="Prompts (one per line — each line is queued and generated in turn)",
                lines=4,
                placeholder="pixelart style, a fire-breathing dragon\npixelart style, a knight in armor",
            )
            gr.Examples(examples=EXAMPLE_PROMPTS, inputs=[prompts], label="Example prompts")
            guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance scale")
            num_steps = gr.Slider(10, 100, value=30, step=5, label="Inference steps")
            batch_size = gr.Slider(1, 8, value=1, step=1, label="Images per prompt")
            with gr.Row():
                pixelate_enabled = gr.Checkbox(
                    label="Pixelate to a true sprite grid", value=True
                )
                sprite_size = gr.Dropdown(
                    choices=[8, 16, 32, 64], value=16, label="Sprite size", type="value"
                )
                palette_colors = gr.Slider(2, 64, value=16, step=1, label="Palette colors")
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop", interactive=False)

            with gr.Group():
                gr.Markdown("### Queue")
                progress_bar = gr.Slider(
                    0, 100, value=0, label="Progress", interactive=False
                )
                gen_status = gr.Textbox(label="Status", interactive=False)
                queue_status = gr.Textbox(
                    label="Queue (one prompt per line)",
                    interactive=False,
                    lines=4,
                )

            live_preview = gr.Image(
                label="Live preview (denoising in progress)", interactive=False
            )
            output = gr.Gallery(label="Generated sprites", columns=4)

        with gr.Tab("Gallery"):
            gallery_refresh_btn = gr.Button("Refresh")
            gallery_view = gr.Gallery(
                label="Previously generated sprites", columns=6, value=list_outputs()
            )

        train_btn.click(
            train,
            inputs=[base_model, captions_dir, rank, steps, force],
            outputs=[train_status, lora_path],
        )
        refresh_btn.click(refresh_lora_weights, inputs=lora_path, outputs=lora_path)
        generate_btn.click(
            generate,
            inputs=[
                gen_base_model,
                lora_path,
                prompts,
                guidance,
                num_steps,
                batch_size,
                pixelate_enabled,
                sprite_size,
                palette_colors,
            ],
            outputs=[
                live_preview,
                output,
                gen_status,
                gallery_view,
                progress_bar,
                queue_status,
                generate_btn,
                cancel_btn,
            ],
            # Progress here is our own Slider + Textbox, not gr.Progress() —
            # its built-in overlay re-triggers on every yield (one per
            # denoising step) and visually fights with the live-preview
            # image update, causing a flash. "hidden" drops that overlay
            # entirely since we don't need Gradio's own indicator at all.
            show_progress="hidden",
        )
        # Deliberately NOT using Gradio's built-in `cancels=` here: that
        # would only stop this event's own consuming loop, not the actual
        # GPU work — pipeline() runs in a separate background thread that
        # Gradio's cancellation has no reach into. Setting _cancel_event
        # and letting the denoising callback notice it is the only thing
        # that actually interrupts generation in progress.
        cancel_btn.click(cancel_generation, outputs=gen_status)
        gallery_refresh_btn.click(lambda: list_outputs(), outputs=gallery_view)
        app.load(refresh_lora_weights, inputs=lora_path, outputs=lora_path)

    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
