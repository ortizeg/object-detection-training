"""
EMA (Exponential Moving Average) callback for PyTorch Lightning.

Maintains a shadow copy of model weights using exponential moving average.
"""

from __future__ import annotations

import copy
from typing import Any

import lightning as L
import torch
from loguru import logger


class EMACallback(L.Callback):
    """
    Exponential Moving Average callback for model weights.

    Maintains EMA weights and uses them for validation/testing.
    """

    def __init__(self, decay: float = 0.9999, warmup_steps: int = 2000):
        """
        Initialize EMA callback.

        Args:
            decay: EMA decay factor.
            warmup_steps: Steps before starting EMA updates.
        """
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.ema_state_dict: dict[str, torch.Tensor] = {}
        self.original_state_dict: dict[str, torch.Tensor] = {}
        self.step_count = 0
        self._ema_applied = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Initialize EMA state dict from model."""
        logger.info(f"Initializing EMA with decay={self.decay}")
        self.ema_state_dict = copy.deepcopy(pl_module.state_dict())
        self.step_count = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after each training batch."""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # During warmup, just copy weights
            self.ema_state_dict = copy.deepcopy(pl_module.state_dict())
            return

        # Compute EMA update
        model_state = pl_module.state_dict()
        for key in self.ema_state_dict.keys():
            if model_state[key].dtype.is_floating_point:
                self.ema_state_dict[key].mul_(self.decay).add_(
                    model_state[key], alpha=1 - self.decay
                )
            else:
                # Non-floating point tensors are copied directly
                self.ema_state_dict[key] = model_state[key].clone()

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Apply EMA weights for validation."""
        if not self.ema_state_dict:
            return

        logger.debug("Applying EMA weights for validation")
        self.original_state_dict = copy.deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self.ema_state_dict)
        self._ema_applied = True

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Restore original weights after validation."""
        if self._ema_applied and self.original_state_dict:
            logger.debug("Restoring original weights after validation")
            pl_module.load_state_dict(self.original_state_dict)
            self._ema_applied = False

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Apply EMA weights for testing."""
        if not self.ema_state_dict:
            return

        logger.debug("Applying EMA weights for testing")
        self.original_state_dict = copy.deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self.ema_state_dict)
        self._ema_applied = True

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Restore original weights after testing."""
        if self._ema_applied and self.original_state_dict:
            logger.debug("Restoring original weights after testing")
            pl_module.load_state_dict(self.original_state_dict)
            self._ema_applied = False

    def state_dict(self) -> dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "ema_state_dict": self.ema_state_dict,
            "step_count": self.step_count,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.ema_state_dict = state_dict.get("ema_state_dict", {})
        self.step_count = state_dict.get("step_count", 0)
        self.decay = state_dict.get("decay", self.decay)
        logger.info(f"Loaded EMA state from checkpoint (step_count={self.step_count})")

    def save_ema_checkpoint(self, filepath: str) -> None:
        """Save EMA weights to a separate checkpoint file."""
        logger.info(f"Saving EMA checkpoint to {filepath}")
        torch.save(self.ema_state_dict, filepath)
