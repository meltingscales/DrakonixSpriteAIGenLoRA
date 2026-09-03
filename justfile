setup:
    uv sync --group dev

data:
    uv run python -m drakonix_lora.fetch_dataset

gui:
    uv run python -m drakonix_lora.gui

test:
    uv run pytest tests/

# TODO once training is wired up (see README.md):
# train captions_dir="data/captions" base="runwayml/stable-diffusion-v1-5" steps="1500":
#     uv run python -m drakonix_lora.train --captions-dir {{captions_dir}} --base-model {{base}} --steps {{steps}}
