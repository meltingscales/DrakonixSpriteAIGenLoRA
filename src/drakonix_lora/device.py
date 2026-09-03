"""Pick a torch device, verifying GPU compute actually works.

torch.cuda.is_available() can return True while the GPU backend still
segfaults on first real op (observed on ROCm 6.4 + gfx1151/Strix Halo in the
sibling DrakonixSpriteAIGen project). A segfault can't be caught with
try/except since it kills the whole process, so the check runs in a
throwaway subprocess instead.

LoRA training is realistically GPU-only (CPU training of even a small SD1.5
LoRA is impractically slow) — if this reports "cpu", check nvidia-smi /
rocminfo before starting a training run rather than letting it run for
hours on CPU.
"""

import subprocess
import sys

import torch

_PROBE = "import torch; x = torch.zeros(4, device='cuda'); (x + 1).sum().item()"


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, timeout=30
    )
    if result.returncode == 0:
        return torch.device("cuda")

    print(
        f"GPU detected ({torch.cuda.get_device_name(0)}) but a real op failed "
        f"(exit {result.returncode}) — falling back to CPU.",
        file=sys.stderr,
    )
    return torch.device("cpu")
