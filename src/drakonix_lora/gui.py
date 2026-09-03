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


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


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

    # ETA from the last few inter-step deltas rather than total elapsed
    # since training started — the first delta would otherwise fold in
    # model-loading time and badly skew the estimate for a short run.
    recent_step_times: list[float] = []
    last_step_at: float | None = None

    def on_step(step: int, total: int, loss: float) -> None:
        nonlocal last_step_at
        now = time.monotonic()
        if last_step_at is not None:
            recent_step_times.append(now - last_step_at)
            del recent_step_times[:-20]
        last_step_at = now

        if recent_step_times:
            avg_step_time = sum(recent_step_times) / len(recent_step_times)
            eta = _format_duration(avg_step_time * (total - step))
        else:
            eta = "estimating…"

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


def generate(lora_path: str, prompt: str, guidance: float, num_steps: int):
    if not lora_path:
        raise gr.Error("no LoRA weights selected — train one first, or click Refresh")
    if not prompt.strip():
        raise gr.Error("prompt is empty")

    pipeline = _ensure_pipeline()
    _ensure_lora(pipeline, lora_path)

    result = pipeline(
        prompt, guidance_scale=guidance, num_inference_steps=int(num_steps)
    )
    return result.images[0]


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
            prompt = gr.Textbox(label="Prompt", placeholder="pixel art of a fire-breathing dragon")
            guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance scale")
            num_steps = gr.Slider(10, 100, value=30, step=5, label="Inference steps")
            generate_btn = gr.Button("Generate", variant="primary")
            output = gr.Image(label="Generated sprite")

        train_btn.click(
            train,
            inputs=[base_model, captions_dir, rank, steps, force],
            outputs=[train_status, lora_path],
        )
        refresh_btn.click(refresh_lora_weights, inputs=lora_path, outputs=lora_path)
        generate_btn.click(
            generate, inputs=[lora_path, prompt, guidance, num_steps], outputs=output
        )
        app.load(refresh_lora_weights, inputs=lora_path, outputs=lora_path)

    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
