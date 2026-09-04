"""Download a stratified sample of genuine 16x16 sprites for a LoRA
trained on true low-res pixel density, rather than the medium-density
hand-drawn illustrations `fetch_dataset.py` sources (see AI-NOTES.md for
why that density mismatch matters — it's the likely cause of "fuzzy"
generated output even after post-process pixelation).

Usage:
    python -m drakonix_lora.fetch_dataset_16px [--force] [--per-class N]

Requires a Kaggle API token — either ~/.kaggle/access_token (or
KAGGLE_API_TOKEN env var) or the older ~/.kaggle/kaggle.json (or
KAGGLE_USERNAME / KAGGLE_KEY). Get one from kaggle.com/settings > API >
Create New Token.

Source: ebrahimelgazar/pixel-art (Kaggle, Apache-2.0) — 89,400 genuine
16x16 RGB sprites, the same dataset the sibling DrakonixSpriteAIGenVAE
project already trains its 8x8/16x16 VAE on. Ships as sprites.npy +
sprites_labels.npy (row-aligned pair) with only a numeric one-hot label
per row, no semantic class names — we visually sampled each of the 5
classes to write CLASS_CAPTIONS below:
  0: standing RPG character sprites
  1: round slime creatures
  2: round fruit items
  3: armor/wand equipment icons
  4: RPG character sprites holding a weapon

Writes into data/captions_16px/ — deliberately a SEPARATE directory from
data/captions/, not merged: mixing this native-16x16 pixel density with
the medium-density Pixilart illustrations in one LoRA training run would
give the model conflicting size cues about how "chunky" pixel art should
be. Point train.py / the GUI's Captions dataset dir at this folder to
train a distinct LoRA from it.

Sampling is deterministic (fixed seed) and stratified evenly across the
5 classes regardless of their wildly uneven raw sizes (6,000-35,000),
so the LoRA doesn't skew toward whichever class happens to have the
most images. Downloads/extraction are skipped if already present;
pass --force to re-fetch everything.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CAPTIONS_DIR = Path("data/captions_16px")
KAGGLE_RAW_DIR = Path("data/kaggle_raw_16px")
DATASET = "ebrahimelgazar/pixel-art"
TARGET_SIZE = 512
TRIGGER = "pixelart style"
PER_CLASS_DEFAULT = 40

CLASS_CAPTIONS = {
    0: "a standing fantasy RPG character sprite",
    1: "a round slime creature",
    2: "a round fruit game item",
    3: "a colorful armor piece or wand, a game equipment icon",
    4: "a fantasy RPG character sprite holding a weapon",
}


def check_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    have_creds = (
        (kaggle_dir / "access_token").exists()
        or (kaggle_dir / "kaggle.json").exists()
        or os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )
    if not have_creds:
        sys.exit(
            "No Kaggle API credentials found.\n"
            "Get a token at https://www.kaggle.com/settings > API > Create New Token,\n"
            f"then save it to {kaggle_dir / 'access_token'} (chmod 600), "
            "or set KAGGLE_API_TOKEN / KAGGLE_USERNAME+KAGGLE_KEY."
        )


def download_npy_files() -> tuple[Path, Path]:
    sprites_path = KAGGLE_RAW_DIR / "sprites.npy"
    labels_path = KAGGLE_RAW_DIR / "sprites_labels.npy"
    if sprites_path.exists() and labels_path.exists():
        return sprites_path, labels_path

    from kaggle.api.kaggle_api_extended import KaggleApi

    KAGGLE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"downloading {DATASET} -> {KAGGLE_RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=str(KAGGLE_RAW_DIR), unzip=True, quiet=False)
    return sprites_path, labels_path


def to_training_canvas(sprite: np.ndarray, size: int = TARGET_SIZE) -> Image.Image:
    """Nearest-neighbor upscale a true 16x16 sprite to `size` — no
    letterboxing needed since the source is already square. This is the
    whole point of this dataset: huge flat blocks, not blended detail."""
    return Image.fromarray(sprite).resize((size, size), Image.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-fetch and overwrite existing files"
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=PER_CLASS_DEFAULT,
        help=f"sprites to sample per class (default {PER_CLASS_DEFAULT}, "
        f"x5 classes = {PER_CLASS_DEFAULT * 5} total)",
    )
    args = parser.parse_args()

    check_credentials()
    sprites_path, labels_path = download_npy_files()

    sprites = np.load(sprites_path)
    labels = np.load(labels_path)
    class_idx = labels.argmax(axis=1)

    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)

    fetched = skipped = 0
    i = 0
    for k in sorted(CLASS_CAPTIONS):
        indices = np.where(class_idx == k)[0]
        count = min(args.per_class, len(indices))
        chosen = rng.choice(indices, size=count, replace=False)
        for idx in chosen:
            i += 1
            stem = f"sprite16_{i:03d}"
            image_path = CAPTIONS_DIR / f"{stem}.png"
            caption_path = CAPTIONS_DIR / f"{stem}.txt"

            if image_path.exists() and not args.force:
                skipped += 1
                continue

            image = to_training_canvas(sprites[idx])
            image.save(image_path)
            caption_path.write_text(f"{TRIGGER}, {CLASS_CAPTIONS[k]}\n")
            fetched += 1

    print(f"done: {fetched} fetched, {skipped} skipped, {i} total")


if __name__ == "__main__":
    main()
