"""Tests for the object detection training framework."""

from __future__ import annotations

import sys
from os.path import dirname, join

import hydra

# Add src to path to import train script if needed
sys.path.append(join(dirname(__file__), "../src"))

# Register model configs by importing the wrappers
import object_detection_training.models as _  # noqa: F401, E402


def test_hydra_configuration():
    """Verify that we can load the configuration using Hydra."""
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="train")

        # Check basic config structure
        assert "task" in cfg
        assert "models" in cfg
        assert "data" in cfg
        assert "trainer" in cfg
        assert "callbacks" in cfg
        # assert "logging" in cfg  # Removed as logging: null removes the key

        # Check task config (from ConfigStore via @register)
        assert cfg.task._target_ == "object_detection_training.tasks.TrainTask"
        assert cfg.task.name == "object_detection_training"

        # Check model config (default is rfdetr_small)
        assert (
            cfg.models._target_
            == "object_detection_training.models.rfdetr_lightning.RFDETRSmallModel"
        )
        assert cfg.models.num_classes is None

        # Check data config
        assert (
            cfg.data._target_
            == "object_detection_training.data.coco_data_module.COCODataModule"
        )
        assert cfg.data.batch_size == 8

        # Check trainer config
        assert cfg.trainer._target_ == "lightning.Trainer"
        assert cfg.trainer.max_epochs == 300
        assert cfg.trainer.precision == "16-mixed"


def test_hydra_model_override():
    """Test that we can override the model with YOLOX."""
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="train", overrides=["models=yolox_s"])

        # Check YOLOX model config
        assert (
            cfg.models._target_
            == "object_detection_training.models.yolox_lightning.YOLOXSModel"
        )
        assert cfg.models.num_classes is None


def test_hydra_callbacks():
    """Test callback configuration."""
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="train")

        # Check callbacks
        assert "model_checkpoint" in cfg.callbacks
        assert "ema" in cfg.callbacks
        assert "onnx_export" in cfg.callbacks
        assert "model_info" in cfg.callbacks

        # Check EMA config
        assert cfg.callbacks.ema.decay == 0.993
        assert cfg.callbacks.ema.warmup_steps == 0
