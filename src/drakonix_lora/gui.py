"""GUI for training a pixel-art LoRA and generating sprites from a prompt.

Usage:
    python -m drakonix_lora.gui

STATUS: both tabs are wired up — training via `train.run_training`,
generation via a cached `StableDiffusionPipeline` + the selected LoRA
weights.
"""

import time
from pathlib import Path

import gradio as gr
from diffusers import StableDiffusionPipeline

from .device import get_device
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
# switching LoRA files swaps just the adapter, not the whole pipeline.
_state: dict = {"pipeline": None, "lora_path": None}


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


def list_lora_weights() -> list[str]:
    if not LORA_DIR.exists():
        return []
    paths = sorted(LORA_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in paths]


def refresh_lora_weights(current: str):
    choices = list_lora_weights()
    value = current if current in choices else (choices[0] if choices else None)
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

    choices = list_lora_weights()
    return f"saved {out_path.name}", gr.Dropdown(choices=choices, value=str(out_path))


def _ensure_pipeline() -> StableDiffusionPipeline:
    if _state["pipeline"] is None:
        pipeline = StableDiffusionPipeline.from_pretrained(DEFAULT_BASE_MODEL)
        pipeline = pipeline.to(get_device())
        _state["pipeline"] = pipeline
    return _state["pipeline"]


def _ensure_lora(pipeline: StableDiffusionPipeline, lora_path: str) -> None:
    if _state["lora_path"] == lora_path:
        return
    if _state["lora_path"] is not None:
        pipeline.unload_lora_weights()
    lora_file = Path(lora_path)
    pipeline.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)
    _state["lora_path"] = lora_path


def generate(
    lora_path: str,
    prompts_text: str,
    guidance: float,
    num_steps: int,
    batch_size: int,
    progress=gr.Progress(),
):
    if not lora_path:
        raise gr.Error("no LoRA weights selected — train one first, or click Refresh")

    queue = [line.strip() for line in prompts_text.splitlines() if line.strip()]
    if not queue:
        raise gr.Error("no prompts — enter at least one line")

    pipeline = _ensure_pipeline()
    _ensure_lora(pipeline, lora_path)

    num_steps = int(num_steps)
    batch_size = int(batch_size)
    # diffusers invokes the step callback num_inference_steps + 1 times per
    # generation (verified empirically, not documented) — account for that
    # in the total so the ETA/progress fraction stay accurate and the
    # display doesn't show e.g. "step 16/15".
    calls_per_prompt = num_steps + 1
    total_calls = len(queue) * calls_per_prompt
    timer = _StepTimer()
    completed_calls = 0
    current_prompt_idx = 0

    def on_denoise_step(pipe, step, timestep, callback_kwargs):
        nonlocal completed_calls
        completed_calls += 1
        timer.tick()
        eta = timer.eta(total_calls - completed_calls)
        progress(
            completed_calls / total_calls,
            desc=(
                f"prompt {current_prompt_idx + 1}/{len(queue)}  "
                f"step {min(step + 1, num_steps)}/{num_steps}  ETA {eta}"
            ),
        )
        return callback_kwargs

    images = []
    for current_prompt_idx, prompt in enumerate(queue):
        result = pipeline(
            prompt,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
            num_images_per_prompt=batch_size,
            callback_on_step_end=on_denoise_step,
        )
        images.extend(result.images)

    status = f"generated {len(images)} image(s) from {len(queue)} prompt(s)"
    return images, status


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
            with gr.Row():
                lora_path = gr.Dropdown(label="LoRA weights", choices=list_lora_weights())
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
            generate_btn = gr.Button("Generate", variant="primary")
            gen_status = gr.Textbox(label="Status", interactive=False)
            output = gr.Gallery(label="Generated sprites", columns=4)

        train_btn.click(
            train,
            inputs=[base_model, captions_dir, rank, steps, force],
            outputs=[train_status, lora_path],
        )
        refresh_btn.click(refresh_lora_weights, inputs=lora_path, outputs=lora_path)
        generate_btn.click(
            generate,
            inputs=[lora_path, prompts, guidance, num_steps, batch_size],
            outputs=[output, gen_status],
        )
        app.load(refresh_lora_weights, inputs=lora_path, outputs=lora_path)

    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
