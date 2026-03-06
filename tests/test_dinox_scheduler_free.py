"""Tests for scheduler-free AdamW optimizer integration (Phase 6, OPT-01/02/03).

Validates AdamWScheduleFree creation, param groups, lifecycle hooks for
train/eval mode switching, checkpoint safety, Hydra config, and no-regression
for the SGD+cosine path.
"""

from __future__ import annotations

import sys
from os.path import dirname, join
from unittest.mock import MagicMock, PropertyMock

import hydra
import torch

# Register model configs by importing the wrappers
sys.path.append(join(dirname(__file__), "../src"))
import object_detection_training.models as _  # noqa: F401
from object_detection_training.models.dinox.config import DINOXConfig
from object_detection_training.models.dinox_lightning import DINOXLightningModel

CONF_PATH = "../src/object_detection_training/conf"


def _make_model(
    use_scheduler_free: bool = False,
    **kwargs: object,
) -> DINOXLightningModel:
    """Create a small DINOXLightningModel for testing."""
    defaults = {
        "num_classes": 2,
        "download_pretrained": False,
        "depth": 0.33,
        "width": 0.25,
    }
    defaults.update(kwargs)
    defaults["use_scheduler_free"] = use_scheduler_free
    return DINOXLightningModel(**defaults)  # type: ignore[arg-type]


class TestSchedulerFreeOptimizerCreation:
    """Verify AdamWScheduleFree is created when use_scheduler_free=True."""

    def test_scheduler_free_optimizer_created(self) -> None:
        """AdamWScheduleFree returned when use_scheduler_free=True."""
        from schedulefree import AdamWScheduleFree

        model = _make_model(use_scheduler_free=True)
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        type(mock_trainer).max_epochs = PropertyMock(return_value=10)
        model._trainer = mock_trainer  # type: ignore[assignment]
        config = model.configure_optimizers()
        assert isinstance(config["optimizer"], AdamWScheduleFree)
        assert "lr_scheduler" not in config

    def test_sgd_optimizer_still_works(self) -> None:
        """SGD + LR scheduler returned when use_scheduler_free=False."""
        model = _make_model(use_scheduler_free=False)
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        type(mock_trainer).max_epochs = PropertyMock(return_value=10)
        model._trainer = mock_trainer  # type: ignore[assignment]
        config = model.configure_optimizers()
        assert isinstance(config["optimizer"], torch.optim.SGD)
        assert "lr_scheduler" in config

    def test_scheduler_free_param_groups(self) -> None:
        """AdamWScheduleFree has 3 param groups with correct decay."""
        from schedulefree import AdamWScheduleFree

        model = _make_model(use_scheduler_free=True)
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        type(mock_trainer).max_epochs = PropertyMock(return_value=10)
        model._trainer = mock_trainer  # type: ignore[assignment]
        config = model.configure_optimizers()
        opt = config["optimizer"]
        assert isinstance(opt, AdamWScheduleFree)
        assert len(opt.param_groups) == 3
        # pg0 = BN (no decay), pg1 = weights (with decay), pg2 = biases
        assert opt.param_groups[0]["weight_decay"] == 0.0
        assert opt.param_groups[1]["weight_decay"] > 0
        assert opt.param_groups[2]["weight_decay"] == 0.0

    def test_scheduler_free_warmup_steps(self) -> None:
        """AdamWScheduleFree has warmup_steps configured."""
        model = _make_model(use_scheduler_free=True)
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        type(mock_trainer).max_epochs = PropertyMock(return_value=10)
        model._trainer = mock_trainer  # type: ignore[assignment]
        config = model.configure_optimizers()
        opt = config["optimizer"]
        # warmup_steps is stored in optimizer defaults or param_groups
        warmup = opt.defaults.get(
            "warmup_steps", opt.param_groups[0].get("warmup_steps")
        )
        assert warmup is not None
        assert warmup > 0


class TestSchedulerFreeHooks:
    """Verify lifecycle hooks call optimizer.train()/eval() correctly."""

    def test_scheduler_free_hooks_train_mode(self) -> None:
        """on_train_epoch_start calls optimizer.train()."""
        model = _make_model(use_scheduler_free=True)
        mock_opt = MagicMock()
        mock_opt.train = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.optimizers = [mock_opt]
        type(mock_trainer).current_epoch = PropertyMock(return_value=0)
        model._trainer = mock_trainer  # type: ignore[assignment]
        model.on_train_epoch_start()
        mock_opt.train.assert_called_once()

    def test_scheduler_free_hooks_eval_mode(self) -> None:
        """on_validation_model_eval calls optimizer.eval()."""
        model = _make_model(use_scheduler_free=True)
        mock_opt = MagicMock()
        mock_opt.eval = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.optimizers = [mock_opt]
        model._trainer = mock_trainer  # type: ignore[assignment]
        model.on_validation_model_eval()
        mock_opt.eval.assert_called_once()

    def test_scheduler_free_hooks_noop_when_disabled(self) -> None:
        """Hooks do NOT call train/eval when use_scheduler_free=False."""
        model = _make_model(use_scheduler_free=False)
        mock_opt = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.optimizers = [mock_opt]
        type(mock_trainer).current_epoch = PropertyMock(return_value=0)
        model._trainer = mock_trainer  # type: ignore[assignment]

        model.on_train_epoch_start()
        mock_opt.train.assert_not_called()

        model.on_validation_model_eval()
        mock_opt.eval.assert_not_called()

    def test_scheduler_free_checkpoint_safety(self) -> None:
        """on_save_checkpoint calls optimizer.eval() for weight averaging."""
        model = _make_model(use_scheduler_free=True)
        mock_opt = MagicMock()
        mock_opt.eval = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.optimizers = [mock_opt]
        model._trainer = mock_trainer  # type: ignore[assignment]
        model.on_save_checkpoint({})
        mock_opt.eval.assert_called_once()


class TestSchedulerFreeConfig:
    """Verify Hydra config and DINOXConfig validation."""

    def test_e6_hydra_config_loads(self) -> None:
        """dinox_m_e6 config enables scheduler-free with correct LR."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_e6"],
            )
        assert cfg.models.use_scheduler_free is True
        assert cfg.models.learning_rate == 0.0025

    def test_dinox_config_validates_scheduler_free(self) -> None:
        """DINOXConfig accepts use_scheduler_free=True without error."""
        config = DINOXConfig(use_scheduler_free=True)
        assert config.use_scheduler_free is True
