"""
Utilities for loading and inspecting checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import omegaconf
import torch
from loguru import logger


def register_safe_globals() -> None:
    """Allow omegaconf containers to be unpickled safely."""
    torch.serialization.add_safe_globals(
        [
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Metadata,
            omegaconf.nodes.AnyNode,
        ]
    )


def get_checkpoint_hparams(checkpoint_path: str | Path) -> dict[str, Any]:
    """
    Load hyperparameters from a PyTorch Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.

    Returns:
        Dictionary containing hyperparameters.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    register_safe_globals()
    logger.debug(f"Loading checkpoint metadata from {path}")

    # Load on CPU to avoid CUDA errors if just peeking metadata
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    return checkpoint.get("hyper_parameters", {})
