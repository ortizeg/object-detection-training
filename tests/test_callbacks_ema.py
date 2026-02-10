"""Tests for the EMA callback."""

from __future__ import annotations

import copy

import torch
from torch import nn

from object_detection_training.callbacks.ema import EMACallback


def _make_model() -> nn.Module:
    """Create a simple model for EMA tests."""
    return nn.Linear(10, 5, bias=False)


class TestEMACallbackInit:
    """Tests for EMACallback initialization."""

    def test_default_parameters(self) -> None:
        ema = EMACallback()
        assert ema.decay == 0.9999
        assert ema.warmup_steps == 2000
        assert ema.step_count == 0
        assert ema._ema_applied is False

    def test_custom_parameters(self) -> None:
        ema = EMACallback(decay=0.99, warmup_steps=100)
        assert ema.decay == 0.99
        assert ema.warmup_steps == 100


class TestEMACallbackFitStart:
    """Tests for on_fit_start hook."""

    def test_initializes_ema_state_dict(self, mock_trainer, mock_pl_module) -> None:  # type: ignore[no-untyped-def]
        ema = EMACallback()
        ema.on_fit_start(mock_trainer, mock_pl_module)

        assert len(ema.ema_state_dict) > 0
        assert ema.step_count == 0


class TestEMACallbackTrainBatch:
    """Tests for on_train_batch_end hook."""

    def test_warmup_copies_weights(self, mock_trainer, mock_pl_module) -> None:  # type: ignore[no-untyped-def]
        ema = EMACallback(warmup_steps=10)
        ema.on_fit_start(mock_trainer, mock_pl_module)

        # During warmup, state dict should be copied directly
        ema.on_train_batch_end(
            mock_trainer, mock_pl_module, None, (torch.zeros(1), []), 0
        )
        assert ema.step_count == 1

    def test_ema_update_after_warmup(self) -> None:
        """After warmup, EMA update uses exponential moving average."""
        model = _make_model()

        ema = EMACallback(decay=0.9, warmup_steps=0)
        ema.ema_state_dict = copy.deepcopy(model.state_dict())
        ema.step_count = 0

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)

        # Create a mock pl_module backed by the real model
        from unittest.mock import MagicMock

        pl_module = MagicMock()
        pl_module.state_dict.return_value = model.state_dict()

        trainer = MagicMock()

        ema.on_train_batch_end(trainer, pl_module, None, (torch.zeros(1), []), 0)

        # EMA should move toward the new weights but not equal them
        for key in ema.ema_state_dict:
            if ema.ema_state_dict[key].dtype.is_floating_point:
                # After one update with decay=0.9, weight = 0.9 * old + 0.1 * 1.0
                assert not torch.equal(ema.ema_state_dict[key], model.state_dict()[key])

    def test_step_count_increments(self, mock_trainer, mock_pl_module) -> None:  # type: ignore[no-untyped-def]
        ema = EMACallback()
        ema.on_fit_start(mock_trainer, mock_pl_module)

        for i in range(5):
            ema.on_train_batch_end(
                mock_trainer, mock_pl_module, None, (torch.zeros(1), []), i
            )

        assert ema.step_count == 5


class TestEMACallbackValidation:
    """Tests for validation hooks (apply and restore EMA weights)."""

    def test_apply_and_restore_weights(self) -> None:
        """EMA weights are applied for validation and restored after."""
        model = _make_model()

        from unittest.mock import MagicMock

        pl_module = MagicMock()
        pl_module.state_dict.return_value = model.state_dict()
        pl_module.load_state_dict = MagicMock()

        trainer = MagicMock()

        ema = EMACallback()
        ema.ema_state_dict = copy.deepcopy(model.state_dict())
        # Modify EMA weights to be different
        for key in ema.ema_state_dict:
            ema.ema_state_dict[key] = ema.ema_state_dict[key] + 1.0

        # Apply EMA weights for validation
        ema.on_validation_start(trainer, pl_module)
        assert ema._ema_applied is True
        assert pl_module.load_state_dict.called

        # Restore original weights after validation
        ema.on_validation_end(trainer, pl_module)
        assert ema._ema_applied is False

    def test_no_apply_when_empty_ema(self, mock_trainer, mock_pl_module) -> None:  # type: ignore[no-untyped-def]
        """No-op when EMA state dict is empty."""
        ema = EMACallback()
        ema.on_validation_start(mock_trainer, mock_pl_module)
        assert ema._ema_applied is False


class TestEMACallbackTest:
    """Tests for test hooks."""

    def test_apply_and_restore_for_test(self) -> None:
        """EMA weights are applied for testing and restored after."""
        model = _make_model()

        from unittest.mock import MagicMock

        pl_module = MagicMock()
        pl_module.state_dict.return_value = model.state_dict()
        pl_module.load_state_dict = MagicMock()

        trainer = MagicMock()

        ema = EMACallback()
        ema.ema_state_dict = copy.deepcopy(model.state_dict())

        ema.on_test_start(trainer, pl_module)
        assert ema._ema_applied is True

        ema.on_test_end(trainer, pl_module)
        assert ema._ema_applied is False


class TestEMACallbackStateDict:
    """Tests for state_dict and load_state_dict."""

    def test_state_dict_roundtrip(self) -> None:
        """State dict save/load roundtrip preserves callback state."""
        ema = EMACallback(decay=0.95, warmup_steps=500)
        ema.step_count = 42
        ema.ema_state_dict = {"weight": torch.tensor([1.0, 2.0, 3.0])}

        state = ema.state_dict()

        ema2 = EMACallback()
        ema2.load_state_dict(state)

        assert ema2.step_count == 42
        assert ema2.decay == 0.95
        torch.testing.assert_close(
            ema2.ema_state_dict["weight"], torch.tensor([1.0, 2.0, 3.0])
        )
