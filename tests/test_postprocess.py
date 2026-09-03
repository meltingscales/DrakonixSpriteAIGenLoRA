from PIL import Image

from drakonix_lora.postprocess import pixelate, upscale_for_viewing


def _noisy_image(size: int = 512) -> Image.Image:
    import random

    image = Image.new("RGB", (size, size))
    pixels = image.load()
    rng = random.Random(0)
    for x in range(size):
        for y in range(size):
            pixels[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    return image


def test_pixelate_produces_true_size_grid() -> None:
    result = pixelate(_noisy_image(), size=16, colors=8)
    assert result.size == (16, 16)
    assert result.mode == "RGB"


def test_pixelate_respects_color_budget() -> None:
    result = pixelate(_noisy_image(), size=32, colors=4)
    unique_colors = {result.getpixel((x, y)) for x in range(32) for y in range(32)}
    assert len(unique_colors) <= 4


def test_pixelate_no_dither_speckle() -> None:
    # A flat-color source should stay exactly flat after pixelating —
    # dithering would introduce speckled variation even from one input color.
    flat = Image.new("RGB", (512, 512), (100, 150, 200))
    result = pixelate(flat, size=16, colors=4)
    unique_colors = {result.getpixel((x, y)) for x in range(16) for y in range(16)}
    assert len(unique_colors) == 1


def test_upscale_for_viewing_preserves_hard_edges() -> None:
    small = Image.new("RGB", (2, 2))
    small.putpixel((0, 0), (255, 0, 0))
    small.putpixel((1, 0), (0, 255, 0))
    small.putpixel((0, 1), (0, 0, 255))
    small.putpixel((1, 1), (255, 255, 0))

    result = upscale_for_viewing(small, display_size=8)
    assert result.size == (8, 8)
    # nearest-neighbor: each quadrant should be a uniform solid block
    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert result.getpixel((3, 0)) == (255, 0, 0)
    assert result.getpixel((4, 0)) == (0, 255, 0)
    assert result.getpixel((7, 7)) == (255, 255, 0)
