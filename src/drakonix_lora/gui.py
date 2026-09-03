"""GUI for training a pixel-art LoRA and generating sprites from a prompt.

Usage:
    python -m drakonix_lora.gui

STATUS: training is wired up to `train.run_training`. Inference isn't yet
— see README.md "what's left to build". The Generate button still raises
a clear error instead of pretending to work.
"""

from pathlib import Path

import gradio as gr

from .train import (
    LORA_DIR,
    checkpoint_label,
    default_checkpoint_path,
    find_matching_checkpoints,
    run_training,
)


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

    def on_step(step: int, total: int, loss: float) -> None:
        progress(step / total, desc=f"step {step}/{total}  loss={loss:.4f}")

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


def generate(lora_path: str, prompt: str, guidance: float, num_steps: int):
    raise gr.Error(
        "Inference isn't wired up yet — see README.md 'what's left to build'. "
        f"Would generate {prompt!r} using LoRA {lora_path!r}."
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Drakonix Sprite LoRA") as app:
        gr.Markdown("# Drakonix Sprite LoRA")
        gr.Markdown(
            "Training is live. Generation isn't implemented yet — "
            "see README.md for the plan."
        )

        with gr.Tab("Train LoRA"):
            base_model = gr.Textbox(
                label="Base model", value="runwayml/stable-diffusion-v1-5"
            )
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
