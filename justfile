setup:
    uv sync --group dev

data:
    uv run python -m drakonix_lora.fetch_dataset

data-16px:
    uv run python -m drakonix_lora.fetch_dataset_16px

gui:
    uv run python -m drakonix_lora.gui

test:
    uv run pytest tests/

train captions_dir="data/captions" base="runwayml/stable-diffusion-v1-5" steps="1500":
    uv run python -m drakonix_lora.train --captions-dir {{captions_dir}} --base-model {{base}} --steps {{steps}}
