"""
Shared CUDA device resolution for the detector zoo.

``torch.cuda.is_available()`` only checks whether the driver exposes a CUDA
device.  It does **not** guarantee that the runtime libraries (cuDNN, CUBLAS,
etc.) are fully functional.  This module runs a cheap smoke-test (a tiny
conv1d forward pass) to verify that the CUDA stack actually works end-to-end
before committing to GPU execution.
"""

import torch
import torch.nn.functional as F

_resolved: torch.device | None = None


def get_device() -> torch.device:
    """Return a ``torch.device`` that is verified to work.

    The result is cached after the first call so the probe only runs once per
    process.
    """
    global _resolved
    if _resolved is not None:
        return _resolved

    if not torch.cuda.is_available():
        _resolved = torch.device("cpu")
        return _resolved

    # Probe: a minimal conv1d forward pass exercises cuDNN.
    try:
        x = torch.randn(1, 1, 8, device="cuda")
        w = torch.randn(1, 1, 3, device="cuda")
        F.conv1d(x, w, padding=1)
        torch.cuda.synchronize()
        _resolved = torch.device("cuda")
    except RuntimeError:
        import warnings
        warnings.warn(
            "CUDA is reported as available but a cuDNN operation failed.  "
            "Falling back to CPU.  To use the GPU, ensure the CUDA/cuDNN "
            "runtime libraries are on LD_LIBRARY_PATH.",
            stacklevel=2,
        )
        _resolved = torch.device("cpu")

    return _resolved
