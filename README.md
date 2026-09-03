# DrakonixSpriteAIGenLoRA

Sibling project to [DrakonixSpriteAIGen](../DrakonixSpriteAIGen) — that repo
is a from-scratch class-conditional VAE (pick a label, get a sprite). This
repo is the answer to "what if we want to generate sprites *from text
descriptions*?": fine-tune an existing pretrained text-to-image model with
[LoRA](https://arxiv.org/abs/2106.09685) instead of training text
understanding from scratch. From-scratch text-to-image needs a captioned
dataset and a text encoder built from nothing — LoRA reuses a model that
already knows what "dragon" or "knight" looks like, and only teaches it a
new *style* (pixel art) cheaply.

## the plan

- **Base model**: Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`
  via [diffusers](https://github.com/huggingface/diffusers)). Smallest/
  oldest capable SD version — lowest VRAM/disk, and most existing pixel-art
  LoRA tutorials and reference LoRAs target it.
- **Training stack**: `diffusers` + [`peft`](https://github.com/huggingface/peft)
  for the LoRA layers, `accelerate` for the training loop, `transformers`
  for the CLIP tokenizer/text encoder that ships with SD1.5.
- **What we don't have yet**: a captioned pixel-art dataset. This is the
  actual blocker, not the training code. LoRA needs (image, caption) pairs;
  DrakonixSpriteAIGen's Kaggle data has only 5 numeric classes, no text.

## what we learned from DrakonixSpriteAIGen that applies here

- **GPU detection can lie.** `torch.cuda.is_available()` returned `True` on
  our ROCm 6.4 / gfx1151 (Strix Halo) setup but segfaulted on the first
  real op. Ported `device.py`'s subprocess-probe pattern as-is — it's
  fully generic, not VAE-specific.
- **LoRA training is realistically GPU-only.** Unlike the tiny 16x16 VAE
  (which trained fine on CPU in minutes), SD1.5 LoRA training is a full
  diffusion U-Net forward/backward at 512x512 — CPU training would take
  hours to days. If `get_device()` reports `cpu` here, that's a hard
  blocker, not just "slower" — check `nvidia-smi`/`rocminfo` before
  starting a training run.
- **Still need to find a working ROCm-compatible PyTorch build.**
  `pyproject.toml` currently pins plain `torch` (CPU/CUDA wheel from
  PyPI) — that will report no GPU at all on this AMD box. In
  DrakonixSpriteAIGen we installed `torch==2.9.1+rocm6.4` from
  `https://download.pytorch.org/whl/rocm6.4`, and `device.py`'s probe
  correctly detected that it segfaults on real ops for our GPU
  (gfx1151/Strix Halo) and fell back to CPU — but that was only tested
  with a trivial op (`zeros + 1`). Whether a ROCm build handles the much
  heavier conv/attention ops in an actual SD1.5 U-Net forward pass is
  untested and needs to be verified before relying on it for training;
  see AI-NOTES.md for what was tried.

## dataset requirements (from current LoRA training practice, 2026)

- **Image count**: 15–30 images is a commonly cited working range for a
  style LoRA (not a per-character concept LoRA — we want "pixel art
  style", not one specific character).
- **Resolution**: match SD1.5's native 512x512 (512x768/768x512 also
  used). Our sprites are natively 8x8/16x16 — upscale with **nearest-
  neighbor** (not bilinear/bicubic) to preserve the blocky pixel-art look;
  smooth upscaling defeats the point.
- **Captions**: every training image needs a caption/tag file (e.g.
  `sprite_001.txt` next to `sprite_001.png`) describing its content, plus
  a consistent trigger word (e.g. `pixelart style`) so the LoRA associates
  that phrase with the learned style at generation time. Caption quality
  matters more than volume here.
- **Training params** (SD1.5 LoRA, typical starting point): learning rate
  1e-4–2e-4, 1500–2500 steps depending on dataset size, batch size 1 (8GB
  VRAM) to 2–4 (12GB+), clip skip 2. Watch sample previews during training
  — stop once generated previews start looking identical to training
  images rather than chasing a fixed step count.

Sources: [Stable Diffusion Art — How to train LoRA models](https://stable-diffusion-art.com/train-lora/),
[SeaArt — dataset creation guide](https://docs.seaart.ai/guide-1/3-advanced-guide/3-2-lora-training-advance/how-to-create-dataset-for-training)

## what's left to build

Roughly in order, since each step's output feeds the next:

0. **Find/verify a working ROCm-compatible PyTorch build for this GPU**
   (or confirm CPU is the only option and budget training time
   accordingly). See "what we learned" above and AI-NOTES.md.
1. **Source or build a captioned pixel-art dataset.** Either find an
   existing CC0/openly-licensed captioned pixel-art set, or caption
   sprites ourselves (manually, or with a vision-captioning model as a
   first pass then human review). 15–30 good examples is the realistic
   starting target, not thousands.
2. **Preprocessing script**: nearest-neighbor upscale sprites to
   512x512(ish), write matching caption `.txt` files into
   `data/captions/`.
3. **Training script** (`src/drakonix_lora/train.py`, not yet written):
   wraps `diffusers`' LoRA training utilities (see
   [diffusers' text-to-image LoRA example](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
   as the reference implementation) with an epoch/step progress callback,
   mirroring how DrakonixSpriteAIGen's `run_training()` reports progress
   to its GUI.
4. **Inference wiring**: load base SD1.5 + LoRA weights via
   `diffusers.StableDiffusionPipeline` + `peft`, generate from a prompt.
5. **Wire both into `gui.py`**, which already has the tab structure/
   controls in place — the Train and Generate buttons currently raise a
   clear "not implemented yet" error instead of pretending to work.

## what's already ported from DrakonixSpriteAIGen

- `justfile` — `just setup` (uv sync), `just gui`, `just test`. A `train`
  recipe is stubbed out (commented) for once step 3 above exists.
- `pyproject.toml` — uv-managed, dependencies swapped for the LoRA stack
  (`diffusers`, `transformers`, `accelerate`, `peft` instead of `kaggle`).
- `src/drakonix_lora/device.py` — GPU-probe-with-subprocess pattern,
  reused verbatim (see "what we learned" above).
- `src/drakonix_lora/gui.py` — Gradio Blocks structure (Train tab +
  Generate tab, checkpoint/weights dropdown with Refresh + on-page-load
  refresh) ported from the same pattern, fields adapted to LoRA
  training/inference params. Buttons are stubbed until steps 3–4 exist.
- `tests/test_gui.py` — smoke test that the GUI builds, same pattern as
  the original repo's approach of actually exercising code rather than
  just reading it.

## setup

```
just setup
```

## GUI (currently a stub)

```
just gui
```

Shows the intended Train/Generate tabs; both actions currently raise a
clear "not implemented yet" error pointing back here.
