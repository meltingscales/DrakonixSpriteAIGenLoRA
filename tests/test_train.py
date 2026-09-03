from pathlib import Path

import pytest

from drakonix_lora.dataset import PixelArtCaptionDataset
from drakonix_lora.train import (
    checkpoint_label,
    dataset_fingerprint,
    default_checkpoint_path,
    find_matching_checkpoints,
)

CAPTIONS_DIR = "data/captions"
_HAS_DATASET = any(Path(CAPTIONS_DIR).glob("*.png")) if Path(CAPTIONS_DIR).exists() else False


def test_checkpoint_label_is_stable_and_collision_avoiding() -> None:
    label_a = checkpoint_label("runwayml/stable-diffusion-v1-5", CAPTIONS_DIR, 16)
    label_b = checkpoint_label("runwayml/stable-diffusion-v1-5", CAPTIONS_DIR, 16)
    label_c = checkpoint_label("runwayml/stable-diffusion-v1-5", CAPTIONS_DIR, 32)

    assert label_a == label_b
    assert label_a != label_c
    assert "runwayml-stable-diffusion-v1-5" in label_a
    assert "r16" in label_a


def test_default_checkpoint_path_uses_lora_dir_and_steps() -> None:
    path = default_checkpoint_path("some-label", 1500)
    assert path.parent.name == "lora_weights"
    assert "some-label_1500steps_" in path.name
    assert path.suffix == ".safetensors"


def test_find_matching_checkpoints_empty_when_dir_missing(tmp_path, monkeypatch) -> None:
    import drakonix_lora.train as train_module

    monkeypatch.setattr(train_module, "LORA_DIR", tmp_path / "does-not-exist")
    assert find_matching_checkpoints("any-label", 1500) == []


@pytest.mark.skipif(not _HAS_DATASET, reason="data/captions is empty — run `just data` first")
def test_pixel_art_caption_dataset_loads_real_data() -> None:
    dataset = PixelArtCaptionDataset(CAPTIONS_DIR)
    assert len(dataset) > 0

    image, caption = dataset[0]
    assert image.shape == (3, 512, 512)
    assert image.min() >= -1.0 and image.max() <= 1.0
    assert caption.strip() != ""
    assert "pixelart style" in caption


def test_pixel_art_caption_dataset_raises_on_empty_dir(tmp_path) -> None:
    with pytest.raises(ValueError):
        PixelArtCaptionDataset(str(tmp_path))


def test_dataset_fingerprint_changes_with_content(tmp_path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    fp1 = dataset_fingerprint(str(tmp_path))

    (tmp_path / "b.png").write_bytes(b"yy")
    fp2 = dataset_fingerprint(str(tmp_path))

    assert fp1 != fp2
    assert len(fp1) == 8
