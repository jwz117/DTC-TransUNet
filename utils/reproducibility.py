"""Reproducibility helpers for DTC-TransUNet."""
import random

import numpy as np
import torch


def set_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    This configures cuDNN deterministic behavior when requested. Exact
    reproducibility can still depend on hardware, library versions, and ops.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

