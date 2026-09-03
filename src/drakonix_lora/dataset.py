"""Captioned image dataset for LoRA training.

Loads (image, caption) pairs written by `fetch_dataset.py`:
`<captions_dir>/sprite_NNN.png` next to a matching `sprite_NNN.txt`.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # (3, H, W)
    return (tensor - 0.5) / 0.5  # SD VAE expects input in [-1, 1]


class PixelArtCaptionDataset(Dataset):
    """Loads <captions_dir>/*.png + matching *.txt caption files."""

    def __init__(self, captions_dir: str):
        root = Path(captions_dir)
        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(root.glob("*.png")):
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                self.samples.append((image_path, caption_path))
        if not self.samples:
            raise ValueError(
                f"no (image, caption) pairs found under {captions_dir} — "
                "run `just data` first"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path, caption_path = self.samples[idx]
        image = Image.open(image_path)
        caption = caption_path.read_text().strip()
        return _image_to_tensor(image), caption
