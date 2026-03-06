"""Tests for EMA callback compatibility with DINOXLightningModel.

Verifies state dict synchronization works for both use_dfl=True and
use_dfl=False without key mismatches (covers INFRA-04).
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import torch

from object_detection_training.callbacks.ema import EMACallback
from object_detection_training.models.dinox_lightning import DINOXLightningModel


def _make_model(use_dfl: bool = False) -> DINOXLightningModel:
    """Create a small DINOXLightningModel for testing."""
    return DINOXLightningModel(
        num_classes=2,
        download_pretrained=False,
        depth=0.67,
        width=0.75,
        use_dfl=use_dfl,
        reg_max=16,
    )


def _make_trainer_mock() -> MagicMock:
    """Create a minimal trainer mock for EMA hooks."""
    trainer = MagicMock()
    trainer.global_step = 100
    trainer.current_epoch = 1
    return trainer


def _make_batch_mock() -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Create a minimal batch for on_train_batch_end signature."""
    images = torch.randn(1, 3, 320, 320)
    targets: list[dict[str, torch.Tensor]] = [
        {
            "boxes": torch.tensor([[160.0, 160.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
        }
    ]
    return images, targets


class TestDINOXEMA:
    """EMA callback compatibility with DINOXLightningModel."""

    def test_ema_state_dict_sync_no_dfl(self) -> None:
        """EMA shadow state dict matches model state dict for non-DFL mode."""
        model = _make_model(use_dfl=False)
        ema = EMACallback(decay=0.9999, warmup_steps=0)
        trainer = _make_trainer_mock()

        # Initialize EMA
        ema.on_fit_start(trainer, model)

        # Verify shadow keys match model keys
        model_keys = set(model.state_dict().keys())
        ema_keys = set(ema.ema_state_dict.keys())
        assert model_keys == ema_keys, (
            f"Key mismatch: "
            f"model_only={model_keys - ema_keys}, "
            f"ema_only={ema_keys - model_keys}"
        )

        # Verify shapes match
        model_sd = model.state_dict()
        for key in model_keys:
            assert model_sd[key].shape == ema.ema_state_dict[key].shape

        # Modify model, run EMA update, verify shadow differs
        original_shadow = copy.deepcopy(ema.ema_state_dict)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        batch = _make_batch_mock()
        ema.on_train_batch_end(trainer, model, None, batch, 0)

        # At least one floating-point tensor should differ
        any_differ = any(
            not torch.equal(original_shadow[k], ema.ema_state_dict[k])
            for k in original_shadow
            if original_shadow[k].dtype.is_floating_point
        )
        assert any_differ, "EMA shadow weights should differ after update"

    def test_ema_state_dict_sync_dfl(self) -> None:
        """EMA shadow state dict includes DFL-specific keys."""
        model = _make_model(use_dfl=True)
        ema = EMACallback(decay=0.9999, warmup_steps=0)
        trainer = _make_trainer_mock()

        # Initialize EMA
        ema.on_fit_start(trainer, model)

        model_keys = set(model.state_dict().keys())
        ema_keys = set(ema.ema_state_dict.keys())
        assert model_keys == ema_keys

        # Verify DFL-specific keys are present
        dfl_keys = [k for k in ema_keys if "dfl" in k.lower()]
        assert len(dfl_keys) > 0, "DFL keys should be present in EMA state dict"

        # Verify reg_preds have 68 output channels in state dict
        reg_pred_keys = [k for k in ema_keys if "reg_preds" in k and "weight" in k]
        for key in reg_pred_keys:
            assert ema.ema_state_dict[key].shape[0] == 68, (
                f"reg_preds {key} should have 68 output channels, "
                f"got {ema.ema_state_dict[key].shape[0]}"
            )

        # Verify EMA update works without errors
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        batch = _make_batch_mock()
        ema.on_train_batch_end(trainer, model, None, batch, 0)

    def test_ema_eval_restore(self) -> None:
        """EMA swap-in/swap-out cycle works for DINOXHead parameter set."""
        model = _make_model(use_dfl=True)
        ema = EMACallback(decay=0.9999, warmup_steps=0)
        trainer = _make_trainer_mock()

        # Initialize EMA
        ema.on_fit_start(trainer, model)

        # Run a few updates to diverge shadow from model
        batch = _make_batch_mock()
        for i in range(3):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.on_train_batch_end(trainer, model, None, batch, i)

        # Save model weights before swap
        pre_swap_sd = copy.deepcopy(model.state_dict())
        shadow_sd = copy.deepcopy(ema.ema_state_dict)

        # Swap in EMA for validation
        ema.on_validation_start(trainer, model)

        # Model should now have shadow weights
        current_sd = model.state_dict()
        for key in shadow_sd:
            torch.testing.assert_close(
                current_sd[key],
                shadow_sd[key],
                msg=f"After swap-in, {key} should match shadow",
            )

        # Swap out (restore original)
        ema.on_validation_end(trainer, model)

        # Model should now have original weights
        restored_sd = model.state_dict()
        for key in pre_swap_sd:
            torch.testing.assert_close(
                restored_sd[key],
                pre_swap_sd[key],
                msg=f"After swap-out, {key} should match original",
            )
