"""Download and prepare the curated pixel-art style-LoRA dataset.

Usage:
    python -m drakonix_lora.fetch_dataset [--force]

STATUS: sources 33 hand-curated images from bghira/free-to-use-pixelart
(Hugging Face, MIT-licensed metadata, images originally from Pixilart.com).
That dataset is ~7,270 rows total but is dominated by fan art, art trades,
and gift/commission pieces depicting other people's characters or
copyrighted IP (Pokemon, Zelda, Mario, Metal Slug, Soul Eater, Sanrio,
One Punch Man, Kingdom Hearts, Sonic, Hatsune Miku, etc. all turned up
while screening it) — these were manually picked for (a) a clean
flat-shaded sprite look consistent with what we want the LoRA to learn,
and (b) no IP or third-party-character references in their own
title/description. See AI-NOTES.md for the full screening process.

Each image is resized (nearest-neighbor, to preserve hard pixel edges) and
letterboxed onto a white 512x512 canvas — SD1.5's native training
resolution — with a hand-written caption + the `pixelart style` trigger
word, ready to point `train.py` at once that exists (see README.md).

Downloads are skipped if the destination file already exists; pass
--force to re-fetch and overwrite everything.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

from PIL import Image

CAPTIONS_DIR = Path("data/captions")
TARGET_SIZE = 512
TRIGGER = "pixelart style"

# Hotlink protection on art.pixilart.com requires a browser-like UA + referer.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Referer": "https://www.pixilart.com/",
}

# (source title, image URL, caption describing visual content — trigger
# word appended at write time, not stored here).
DATASET = [
    (
        "Highlander",
        "https://art.pixilart.com/sr2e3cecf2e99aws3.png",
        "a bearded warrior in a blue plaid kilt and brown cape holding a "
        "large sword, standing",
    ),
    (
        "The Hornet",
        "https://art.pixilart.com/sr2506f16456baws3.png",
        "a fiery orange and red humanoid figure made of glowing flame "
        "wisps, on a black background",
    ),
    (
        "Band",
        "https://art.pixilart.com/sr2aaf8a6be20aws3.png",
        "four small character portrait icons in a grid: a girl in black "
        "with red accents, a girl with red hair and sparkles, a black cat "
        "creature with yellow eyes, and a purple owl creature wearing "
        "goggles",
    ),
    (
        "Froggie Summoner Badge",
        "https://art.pixilart.com/sr2f5bffe3782aws3.png",
        "a green frog wearing a purple witch hat, sitting inside a glowing "
        "magic circle badge",
    ),
    (
        "mustache",
        "https://art.pixilart.com/sr23880d5a4e3aws3.png",
        "a gray and white cat wearing a small black top hat and a drawn-on "
        "mustache, sitting behind a bowl",
    ),
    (
        "fflower",
        "https://art.pixilart.com/sr2182f1c91d5aws3.png",
        "a glowing orange flame shaped like a flower on a long curling "
        "stem, dark background",
    ),
    (
        "N3s Sprites - Standard Pack 3",
        "https://art.pixilart.com/sr2e202def5e5aws3.png",
        "a game sprite sheet showing a grid of small fantasy creature "
        "icons — dragon, plant, ghost, bird — each next to a colored dot, "
        "on a gray background",
    ),
    (
        "Happy Birthday 'Lotl",
        "https://art.pixilart.com/sr2e6e563d60eaws3.png",
        "a pink axolotl wearing a small party hat, sitting behind a "
        "birthday cake with candles, with balloons and a HAPPY BIRTHDAY "
        "banner, pink background",
    ),
    (
        "Shark with Eyepatch",
        "https://art.pixilart.com/sr2d36479716daws3.png",
        "a cartoon shark head wearing a black eyepatch, big toothy grin, "
        "dark background",
    ),
    (
        "zombo",
        "https://art.pixilart.com/sr280841b5fdfaws3.png",
        "two green zombie characters, an adult holding a small child, flat "
        "cartoon style",
    ),
    (
        "Froggo",
        "https://art.pixilart.com/sr2a5dac18583aws3.png",
        "a light green frog sitting and holding a pink balloon string, "
        "sleepy expression, purple background",
    ),
    (
        "StrawberryFairy",
        "https://art.pixilart.com/sr29b3229a273aws3.png",
        "a fairy girl with pink wings and a strawberry-themed headband, "
        "green dress, framed by a strawberry-pattern border",
    ),
    (
        "Strawber",
        "https://art.pixilart.com/sr203168761ccaws3.png",
        "a large red strawberry with green leaves and sparkle decorations, "
        "blue background",
    ),
    (
        "LumberHouse",
        "https://art.pixilart.com/sr24f9362904aaws3.png",
        "a small two-story wooden and stone cottage with an orange roof, a "
        "tiny character and tree stump in front, green grass, blue sky",
    ),
    (
        "Pink Clouds",
        "https://art.pixilart.com/sr2471dbafe4baws3.png",
        "a small house on a green hill under large pink cloud shapes, "
        "minimalist scene",
    ),
    (
        "My progress",
        "https://art.pixilart.com/sr223fa7815b7aws3.png",
        "a dark forest with tall tree trunks in the foreground and small "
        "colorful character figures in a clearing behind them, moody blue "
        "lighting",
    ),
    (
        "[17] Scorching",
        "https://art.pixilart.com/sr2c1384cb36aaws3.png",
        "a bright orange glowing flower on a stem, dark speckled "
        "background",
    ),
    (
        "Moonrise",
        "https://art.pixilart.com/sr2ec4e5f89b1aws3.png",
        "a large full moon rising over a lake, dark pine trees silhouetted "
        "against a purple night sky, reflection in the water",
    ),
    (
        "landscape 2",
        "https://art.pixilart.com/sr235cae2db2faws3.png",
        "a small house next to trees on a green field under a cloudy sky",
    ),
    (
        "forest",
        "https://art.pixilart.com/sr28f6e21f93eaws3.png",
        "a tree growing on a rocky cliff, covered in green moss and vines",
    ),
    (
        "merry go v2",
        "https://art.pixilart.com/sr2d844512e57aws3.png",
        "a round cartoon character riding a small wooden boat on ocean "
        "waves",
    ),
    (
        "Rings!",
        "https://art.pixilart.com/sr23324207e7faws3.png",
        "a view of colorful planets and a starry galaxy in space",
    ),
    (
        "Dealbreaker",
        "https://art.pixilart.com/sr26ba0377e0eaws3.png",
        "a purple furry mammoth-like creature wearing a green cap, with a "
        "HELLO TRAVELER dialogue box below",
    ),
    (
        "Red's town of the skies",
        "https://art.pixilart.com/sr2cd76e2abd6aws3.png",
        "a snowy floating town at night with glowing lit windows and a red "
        "moon",
    ),
    (
        "Exaleber",
        "https://art.pixilart.com/sr2650cb08586aws3.png",
        "a small creature climbing a glowing crystal tower, purple "
        "mountains and colorful sunset sky",
    ),
    (
        "Kinari",
        "https://art.pixilart.com/sr280abbd147faws3.png",
        "a white-haired girl character wearing a dark outfit with white "
        "gloves and boots, pink background",
    ),
    (
        "Autumn",
        "https://art.pixilart.com/sr2c08beaf7b2aws3.png",
        "a forest path with autumn-colored trees and a small figure "
        "walking under a cloudy sky",
    ),
    (
        "Space",
        "https://art.pixilart.com/sr2b84041e73caws3.png",
        "three colorful alien monster sprites floating in a starry night "
        "sky",
    ),
    (
        "Hiding in the Blue",
        "https://art.pixilart.com/sr2cd38732d6eaws3.png",
        "a small floating island with a tiny pink character standing on "
        "top, blue sky background",
    ),
    (
        "tree house city",
        "https://art.pixilart.com/sr27b67d757afaws3.png",
        "a glowing treehouse with string lights built into a large tree "
        "at night",
    ),
    (
        "corgi",
        "https://art.pixilart.com/sr26b1197bfc4aws3.png",
        "a happy corgi dog with its tongue out, cream and orange fur",
    ),
    (
        "floating islands",
        "https://art.pixilart.com/sr2213355f4f5aws3.png",
        "two floating islands with rocky mountain peaks against a blue "
        "sky",
    ),
    (
        "Starry Night",
        "https://art.pixilart.com/sr284a92a3bas004.png",
        "a night village scene with a tall cactus, swirling stars, and "
        "smiling moon faces in the sky",
    ),
]


def fetch_image(url: str) -> Image.Image:
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=15) as response:
        return Image.open(response).convert("RGBA")


def to_training_canvas(image: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """Nearest-neighbor resize to fit within `size`, letterboxed onto white.

    Nearest-neighbor keeps pixel-art edges hard instead of blurring them —
    matters whether we're upscaling a small sprite or downscaling a large
    one.
    """
    resized = image.copy()
    resized.thumbnail((size, size), Image.NEAREST)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.paste(resized, offset, resized)
    return canvas.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-fetch and overwrite existing files"
    )
    args = parser.parse_args()

    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    fetched = skipped = 0
    for i, (title, url, caption) in enumerate(DATASET, start=1):
        stem = f"sprite_{i:03d}"
        image_path = CAPTIONS_DIR / f"{stem}.png"
        caption_path = CAPTIONS_DIR / f"{stem}.txt"

        if image_path.exists() and not args.force:
            print(f"skip {stem} ({title!r}, already exists)")
            skipped += 1
            continue

        print(f"fetch {stem} ({title!r})")
        image = to_training_canvas(fetch_image(url))
        image.save(image_path)
        caption_path.write_text(f"{TRIGGER}, {caption}\n")
        fetched += 1

    print(f"done: {fetched} fetched, {skipped} skipped, {len(DATASET)} total")
    if fetched == 0 and skipped == 0:
        sys.exit("no images processed — check DATASET is non-empty")


if __name__ == "__main__":
    main()
