"""Tests for DINO-X YAML parameterization via Hydra.

Validates Phase 1 Hydra configs (dinox_m_baseline, dinox_m_dfl) compose
correctly and produce working models. Follows test_yolox_parameterization.py
pattern.
"""

from __future__ import annotations

import sys
from os.path import dirname, join

import hydra
import torch

# Register model configs by importing the wrappers
sys.path.append(join(dirname(__file__), "../src"))
import object_detection_training.models as _  # noqa: F401
from object_detection_training.models.dinox_lightning import DINOXLightningModel

CONF_PATH = "../src/object_detection_training/conf"


class TestHydraConfigCompleteness:
    """Verify DINO-X YAML configs contain all required parameters."""

    def test_dinox_baseline_has_arch_params(self) -> None:
        """dinox_m_baseline has correct architecture parameters."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_baseline"],
            )
        assert cfg.models.depth == 0.67
        assert cfg.models.width == 0.75
        assert cfg.models.depthwise is False
        assert cfg.models.use_dfl is False
        assert cfg.models.checkpoint_name == "yolox_m.pth"

    def test_dinox_dfl_has_arch_params(self) -> None:
        """dinox_m_dfl has correct DFL architecture parameters."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_dfl"],
            )
        assert cfg.models.depth == 0.67
        assert cfg.models.width == 0.75
        assert cfg.models.use_dfl is True
        assert cfg.models.reg_max == 16
        assert cfg.models.dfl_loss_weight == 0.25

    def test_dinox_base_params_present(self) -> None:
        """dinox_base defines common training hyperparameters."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_baseline"],
            )
        base_params = [
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
            "download_pretrained",
            "freeze_backbone_epochs",
            "l1_loss_epoch",
            "iou_loss_type",
            "use_dfl",
            "reg_max",
        ]
        for param in base_params:
            assert param in cfg.models, f"Missing base param '{param}' in dinox config"


class TestHydraOverride:
    """Verify Hydra overrides work for DINO-X configs."""

    def test_override_use_dfl(self) -> None:
        """Override use_dfl on baseline config."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_baseline", "models.use_dfl=true"],
            )
        assert cfg.models.use_dfl is True

    def test_override_reg_max(self) -> None:
        """Override reg_max."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_baseline", "models.reg_max=8"],
            )
        assert cfg.models.reg_max == 8

    def test_override_dfl_loss_weight(self) -> None:
        """Override dfl_loss_weight."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_dfl", "models.dfl_loss_weight=1.5"],
            )
        assert cfg.models.dfl_loss_weight == 1.5


class TestModelInstantiationNoPretrained:
    """Instantiate DINOXLightningModel with download_pretrained=False."""

    def test_instantiate_baseline(self) -> None:
        """Create baseline model and verify forward pass."""
        model = DINOXLightningModel(
            num_classes=2,
            download_pretrained=False,
            depth=0.67,
            width=0.75,
            use_dfl=False,
        )
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "predictions" in out
        assert out["predictions"].shape[0] == 1
        assert out["predictions"].shape[2] == 5 + 2

    def test_instantiate_dfl(self) -> None:
        """Create DFL model and verify forward pass."""
        model = DINOXLightningModel(
            num_classes=2,
            download_pretrained=False,
            depth=0.67,
            width=0.75,
            use_dfl=True,
            reg_max=16,
        )
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert "predictions" in out
        assert out["predictions"].shape[0] == 1
        assert out["predictions"].shape[2] == 5 + 2
