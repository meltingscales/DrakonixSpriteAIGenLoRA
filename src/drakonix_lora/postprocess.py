"""Force diffusion output into an actual low-res pixel-art grid.

Diffusion models render "pixel art style" texture at whatever resolution
they're asked for — they don't natively output a discrete NxN grid, no
matter how well-trained the LoRA is. Getting a genuine 8x8/16x16 sprite
needs a deterministic post-process: average the render down to the target
grid, then collapse it to a limited color palette. This works regardless
of which model or LoRA produced the source image.
"""

from PIL import Image


def pixelate(image: Image.Image, size: int, colors: int) -> Image.Image:
    """Downscale to a true `size`x`size` grid and quantize to `colors`
    colors. LANCZOS (not NEAREST) for the downscale — it averages each
    cell's neighborhood into one representative color instead of picking
    a single, possibly noisy, source pixel. Dithering is disabled so
    quantization gives flat color cells, not speckled noise."""
    small = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return small.quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    ).convert("RGB")


def upscale_for_viewing(image: Image.Image, display_size: int) -> Image.Image:
    """Nearest-neighbor upscale a true-size sprite for on-screen legibility
    without softening its hard pixel edges. Purely a display concern — the
    saved file should stay at the sprite's real size."""
    return image.resize((display_size, display_size), Image.Resampling.NEAREST)
