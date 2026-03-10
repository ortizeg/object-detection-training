"""
Seed utilities for reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic cuDNN algorithms
            (slower but bit-reproducible). If False (default), enable
            cuDNN autotuner for ~5-10% speedup on convolutions.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
