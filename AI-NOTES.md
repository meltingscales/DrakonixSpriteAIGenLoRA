# AI notes

Context for whoever (human or AI) picks this repo up next, that didn't fit
in README.md. README is the plan; this is the "here's what we already
tried and why" scratchpad.

## Hardware this was scaffolded on

AMD Ryzen AI Max+ 395 w/ Radeon 8060S ("Strix Halo" APU), gfx1151, 32
logical CPUs, CachyOS (Arch-based) Linux. If you're on different hardware
(especially an actual discrete NVIDIA/AMD GPU), a lot of the ROCm-specific
caution below may not apply to you — just verify with `device.py`'s probe
either way.

## ROCm attempt, in detail (from the sibling DrakonixSpriteAIGen repo)

We installed a ROCm build of PyTorch there and it's worth recording exactly
what happened, since it'll save re-discovery time here:

- Default `pip install torch` / `uv add torch` gives a CUDA-targeting wheel
  (works CPU-only on non-NVIDIA hardware, reports no GPU).
- ROCm wheels exist at `https://download.pytorch.org/whl/rocm6.4` (and
  `rocm6.3`). We installed `torch==2.9.1+rocm6.4`.
- `torch.cuda.is_available()` returned `True` and `torch.version.hip`
  reported `6.4.43484-123eb5128` — looked like it worked.
- A trivial real op (`torch.zeros(4, device='cuda') + 1`) **segfaulted**
  (process exit 139) when run directly. Also saw a non-fatal warning
  printed alongside: `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or
  directory` — didn't chase whether that's related.
- Because a segfault kills the whole process, `try/except` can't catch it.
  `device.py`'s `get_device()` runs the trivial op in a subprocess and
  checks the subprocess's exit code instead — this reliably detects the
  crash and falls back to CPU without taking the caller down with it.

**What's untested**: whether a full SD1.5 U-Net forward/backward (much
heavier conv2d/attention workload than `zeros + 1`) also segfaults, or
whether it's specifically the trivial-op path that's broken and real
workloads are fine, or vice versa. Worth testing directly with a tiny
`diffusers` pipeline call before assuming CPU is the only option — but
budget real time for it, since a hung/crashed multi-GB model load is
annoying to debug.

**Update 2026-09-03: found a working build.** `torch==2.14.0+rocm7.2`
(Python 3.11, from `https://download.pytorch.org/whl/rocm7.2`) works —
`torch.cuda.is_available()` is `True`, `torch.version.hip` reports
`7.2.53211`, and both the trivial op and a heavier conv2d+backward /
multihead-attention+backward workload (closer to actual U-Net shape) ran
clean with no segfault. Matches what upstream issues say: ROCm 7.1 has a
known segfault bug on gfx1151, fixed in 7.2 (see
[pytorch/pytorch#173367](https://github.com/pytorch/pytorch/issues/173367),
[ROCm/legacy-rocm-build#6530](https://github.com/ROCm/legacy-rocm-build/issues/6530)).
Kernel matters too — a regression from 6.18.3 onward is worked around by
this box already running a newer kernel (`7.2.2-1-cachyos`).

Getting `uv` to actually resolve this was its own fight, worth recording:

- `pip install torch` / plain `uv add torch` picks a CUDA wheel off PyPI
  now (PyPI hosts CUDA builds directly), not a ROCm one — you must point
  at the ROCm index explicitly.
- PyTorch's ROCm wheels are now published as "wheel variants" (same
  version string, different builds per accelerator) — the index at
  `download.pytorch.org/whl/rocm7.2` also carries a `triton-rocm` package
  that torch depends on. That package name **also exists as an empty stub
  on PyPI** (`triton-rocm==3.0.0rc1`, no real wheels) — if the ROCm index
  is marked `explicit = true` in `[[tool.uv.index]]`, uv only consults it
  for packages *directly* pinned to it via `[tool.uv.sources]`, and
  apparently doesn't propagate that pin to `torch`'s own transitive
  dependency on `triton-rocm` cleanly — resolution kept finding the
  PyPI stub instead and failing. Dropping `explicit = true` (index is
  just an extra source, not exclusive) fixed it.
- `uv sync` resolves universally across every Python version and platform
  allowed by `requires-python`, not just the active venv's version — so a
  Python-3.14-branch or Windows-branch resolution failure can block `uv
  sync` even though you're running Python 3.11 on Linux. Fixed by capping
  `requires-python = ">=3.10,<3.14"` (cp314 ROCm wheels' triton-rocm dep
  isn't published yet) and adding
  `[tool.uv] environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`
  to stop it from trying to solve for Windows too.
- Final working `pyproject.toml` shape: `torch>=2.14` in `dependencies`,
  `[[tool.uv.index]]` pointing at the rocm7.2 URL, and
  `[tool.uv.sources]` pinning both `torch` and `triton-rocm` to that
  index by name.

## Kaggle dataset gotcha (from the sibling repo, in case a captioned pixel-art
set ever comes from Kaggle too)

The `ebrahimelgazar/pixel-art` dataset ships `sprites.npy` +
`sprites_labels.npy` (aligned by row — guaranteed, since both came from the
same source array) *and* a separately-extracted `images/*.JPEG` folder
whose filenames do **not** reliably correspond to that row order (checked:
pixel diff between `sprites.npy[0]` and `images/image_0.JPEG` was as large
as diff against an unrelated index). Lesson: when a dataset ships both a
raw array format and an extracted-files format, don't assume filename
order matches array order — verify with an actual pixel comparison, or
build directly from whichever format has a documented/guaranteed
correspondence.

## Design decisions carried over from DrakonixSpriteAIGen (and why)

- **uv** for dependency management, **just** for task running — deps live
  in `pyproject.toml`, `uv.lock` is committed, `justfile` recipes wrap
  `uv run`/`uv sync`. Chosen there because the user asked for it directly;
  kept here for consistency between the two repos.
- **Gradio** for the GUI — `gr.Progress()` param on a plain (non-generator)
  function gives a working progress bar without needing to hand-roll
  streaming; `app.load(fn, outputs=...)` re-runs a function on every page
  load, which is how the checkpoint/weights dropdown stays fresh across
  browser sessions without a manual refresh click.
- **Checkpoint/weights auto-naming + duplicate-run warning**: sibling repo
  names checkpoints `<source>_<size>px_<epochs>ep_<timestamp>.pt` and
  refuses to retrain with identical source+size+epochs unless a "Force
  retrain" checkbox is checked (avoids silently wasting a training run
  reproducing the same result). If/when `train.py` gets written here, the
  equivalent would key off something like
  `<base_model_slug>_r<rank>_<steps>steps_<timestamp>.safetensors` and the
  same source+rank+steps+dataset-hash tuple.
- **Verify by actually running things, not just reading code.** Every
  feature in the sibling repo got exercised for real before being called
  done — training runs, sampling, GUI server smoke-tested via
  `gradio_client` hitting the real HTTP API, not just unit-testing pure
  functions. Worth keeping that habit here, especially once real
  `diffusers` training/inference code exists — that stuff fails in ways
  that are easy to miss by inspection alone (wrong dtype, wrong device,
  silently falling back to CPU, etc).

## Open, not-yet-decided

- Where the captioned pixel-art dataset actually comes from. This needs a
  human licensing decision, not something to automate — see README's
  "what's left to build" #1.
- Whether to target SD1.5 long-term or reconsider SDXL once there's a
  working GPU path (SD1.5 was picked for lower resource needs while GPU
  status was still unknown — see README's base model rationale).
