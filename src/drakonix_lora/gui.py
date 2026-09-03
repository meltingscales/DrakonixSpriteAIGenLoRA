"""GUI for training a pixel-art LoRA and generating sprites from a prompt.

Usage:
    python -m drakonix_lora.gui

STATUS: structural stub, ported from DrakonixSpriteAIGen's gui.py. The tabs
and controls reflect the plan in README.md, but the actual training/
inference calls into diffusers + peft aren't wired up yet — see README.md
"what's left to build". Both buttons currently raise a clear error instead
of pretending to work.
"""

from pathlib import Path

import gradio as gr

LORA_DIR = Path("lora_weights")


def list_lora_weights() -> list[str]:
    if not LORA_DIR.exists():
        return []
    paths = sorted(LORA_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in paths]


def refresh_lora_weights(current: str):
    choices = list_lora_weights()
    value = current if current in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def train(base_model: str, captions_dir: str, rank: int, steps: int, progress=gr.Progress()):
    raise gr.Error(
        "LoRA training isn't wired up yet — see README.md 'what's left to build'. "
        f"Would train {base_model!r} on {captions_dir!r}, rank={rank}, steps={steps}."
    )


def generate(lora_path: str, prompt: str, guidance: float, num_steps: int):
    raise gr.Error(
        "Inference isn't wired up yet — see README.md 'what's left to build'. "
        f"Would generate {prompt!r} using LoRA {lora_path!r}."
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Drakonix Sprite LoRA") as app:
        gr.Markdown("# Drakonix Sprite LoRA")
        gr.Markdown(
            "**Stub UI** — training/generation aren't implemented yet. "
            "See README.md for the plan."
        )

        with gr.Tab("Train LoRA"):
            base_model = gr.Textbox(
                label="Base model", value="runwayml/stable-diffusion-v1-5"
            )
            captions_dir = gr.Textbox(label="Captioned dataset dir", value="data/captions")
            rank = gr.Slider(1, 128, value=16, step=1, label="LoRA rank")
            steps = gr.Slider(100, 5000, value=1500, step=100, label="Training steps")
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
            train, inputs=[base_model, captions_dir, rank, steps], outputs=train_status
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
