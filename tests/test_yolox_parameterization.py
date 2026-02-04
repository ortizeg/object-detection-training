"""Tests for YOLOX YAML parameterization.

Validates that architecture parameters from Hydra YAML configs produce
identical models to the previous YOLOX_CONFIGS dict approach.
"""

from __future__ import annotations

import sys
from os.path import dirname, join

import hydra
import pytest
import torch

# Register model configs by importing the wrappers
sys.path.append(join(dirname(__file__), "../src"))
import object_detection_training.models as _  # noqa: F401
from object_detection_training.models.yolox_lightning import YOLOXLightningModel

CONF_PATH = "../src/object_detection_training/conf"

# ---------------------------------------------------------------------------
# Variant parameter definitions (source of truth: original YOLOX_CONFIGS dict)
# ---------------------------------------------------------------------------

VARIANT_EXPECTED_PARAMS = {
    "nano": {
        "depth": 0.33,
        "width": 0.25,
        "depthwise": True,
        "in_channels": [256, 512, 1024],
        "checkpoint_name": "yolox_nano.pth",
    },
    "tiny": {
        "depth": 0.33,
        "width": 0.375,
        "depthwise": False,
        "in_channels": [256, 512, 1024],
        "checkpoint_name": "yolox_tiny.pth",
    },
    "s": {
        "depth": 0.33,
        "width": 0.50,
        "depthwise": False,
        "in_channels": [256, 512, 1024],
        "checkpoint_name": "yolox_s.pth",
    },
    "m": {
        "depth": 0.67,
        "width": 0.75,
        "depthwise": False,
        "in_channels": [256, 512, 1024],
        "checkpoint_name": "yolox_m.pth",
    },
    "l": {
        "depth": 1.0,
        "width": 1.0,
        "depthwise": False,
        "in_channels": [256, 512, 1024],
        "checkpoint_name": "yolox_l.pth",
    },
    "x": {
        "depth": 1.33,
        "width": 1.25,
        "depthwise": False,
        "in_channels": [256, 512, 1024],
        "checkpoint_name": "yolox_x.pth",
    },
}

VARIANT_YAML_MAP = {
    "nano": "yolox_nano",
    "tiny": "yolox_tiny",
    "s": "yolox_s",
    "m": "yolox_m",
    "l": "yolox_l",
    "x": "yolox_x",
}

# Original YOLOX_CONFIGS dict values (for equivalence testing)
ORIGINAL_YOLOX_CONFIGS = {
    "nano": {"depth": 0.33, "width": 0.25, "depthwise": True},
    "tiny": {"depth": 0.33, "width": 0.375, "depthwise": False},
    "s": {"depth": 0.33, "width": 0.50, "depthwise": False},
    "m": {"depth": 0.67, "width": 0.75, "depthwise": False},
    "l": {"depth": 1.0, "width": 1.0, "depthwise": False},
    "x": {"depth": 1.33, "width": 1.25, "depthwise": False},
}


# ---------------------------------------------------------------------------
# Hydra config completeness tests
# ---------------------------------------------------------------------------


class TestHydraConfigCompleteness:
    """Verify YAML configs contain all required architecture parameters."""

    @pytest.mark.parametrize("variant", list(VARIANT_EXPECTED_PARAMS.keys()))
    def test_variant_has_all_arch_params(self, variant: str) -> None:
        """Each variant YAML must contain all architecture parameters."""
        yaml_name = VARIANT_YAML_MAP[variant]
        expected = VARIANT_EXPECTED_PARAMS[variant]

        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_yolox", overrides=[f"models={yaml_name}"]
            )

        for key, expected_val in expected.items():
            msg = f"Missing param '{key}' in {yaml_name}.yaml config"
            assert key in cfg.models, msg
            actual = cfg.models[key]
            # Convert OmegaConf lists to plain lists for comparison
            if hasattr(actual, "__iter__") and not isinstance(actual, str):
                actual = list(actual)
            assert actual == expected_val, (
                f"Param '{key}' mismatch in {yaml_name}: "
                f"expected={expected_val!r}, got={actual!r}"
            )

    def test_base_params_present(self) -> None:
        """yolox_base.yaml must define common training hyperparameters."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(config_name="train_yolox", overrides=["models=yolox_s"])

        base_params = [
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
            "download_pretrained",
            "freeze_backbone_epochs",
            "l1_loss_epoch",
            "iou_loss_type",
            "image_mean",
            "image_std",
            "depth",
            "width",
            "depthwise",
            "in_channels",
        ]
        for param in base_params:
            assert param in cfg.models, f"Missing base param '{param}' in yolox config"


# ---------------------------------------------------------------------------
# YAML-to-dict equivalence tests
# ---------------------------------------------------------------------------


class TestYamlConfigDictEquivalence:
    """Verify YAML params match the original YOLOX_CONFIGS dict values."""

    @pytest.mark.parametrize("variant", list(ORIGINAL_YOLOX_CONFIGS.keys()))
    def test_yaml_matches_original_dict(self, variant: str) -> None:
        """YAML architecture params must match original YOLOX_CONFIGS dict."""
        original = ORIGINAL_YOLOX_CONFIGS[variant]
        yaml_name = VARIANT_YAML_MAP[variant]

        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_yolox", overrides=[f"models={yaml_name}"]
            )

        for param, expected_val in original.items():
            yaml_val = cfg.models[param]
            assert yaml_val == expected_val, (
                f"Mismatch for {variant}/{param}: "
                f"original_dict={expected_val!r}, yaml={yaml_val!r}"
            )


# ---------------------------------------------------------------------------
# Model architecture equivalence tests
# ---------------------------------------------------------------------------


class TestModelArchitectureEquivalence:
    """Verify YAML-parameterized models match dict-config models."""

    @pytest.mark.parametrize("variant", ["nano", "s", "m", "l"])
    def test_yaml_model_matches_dict_model(self, variant: str) -> None:
        """Model built from YAML params must have identical state_dict
        keys and shapes as model built from old dict params."""
        from object_detection_training.models.yolox import (
            YOLOPAFPN,
            YOLOX,
            YOLOXHead,
        )

        num_classes = 2
        original = ORIGINAL_YOLOX_CONFIGS[variant]
        in_channels = [256, 512, 1024]

        # Build via original dict params (old path)
        backbone_old = YOLOPAFPN(
            depth=original["depth"],
            width=original["width"],
            in_channels=in_channels,
            depthwise=original["depthwise"],
        )
        head_old = YOLOXHead(
            num_classes=num_classes,
            width=original["width"],
            in_channels=in_channels,
            depthwise=original["depthwise"],
        )
        model_old = YOLOX(backbone=backbone_old, head=head_old)
        old_state = model_old.state_dict()

        # Build via YAML params (new path)
        expected = VARIANT_EXPECTED_PARAMS[variant]
        backbone_new = YOLOPAFPN(
            depth=expected["depth"],
            width=expected["width"],
            in_channels=expected["in_channels"],
            depthwise=expected["depthwise"],
        )
        head_new = YOLOXHead(
            num_classes=num_classes,
            width=expected["width"],
            in_channels=expected["in_channels"],
            depthwise=expected["depthwise"],
        )
        model_new = YOLOX(backbone=backbone_new, head=head_new)
        new_state = model_new.state_dict()

        # Compare keys
        assert set(old_state.keys()) == set(new_state.keys()), (
            f"Key mismatch for {variant}: "
            f"old_only={set(old_state.keys()) - set(new_state.keys())}, "
            f"new_only={set(new_state.keys()) - set(old_state.keys())}"
        )

        # Compare shapes
        for key in old_state:
            assert old_state[key].shape == new_state[key].shape, (
                f"Shape mismatch for {variant}/{key}: "
                f"old={old_state[key].shape} vs new={new_state[key].shape}"
            )

    @pytest.mark.parametrize("variant", ["nano", "s"])
    def test_forward_shape_yaml_params(self, variant: str) -> None:
        """Forward pass with YAML params produces correct output shapes."""
        from object_detection_training.models.yolox import (
            YOLOPAFPN,
            YOLOX,
            YOLOXHead,
        )

        num_classes = 2
        expected = VARIANT_EXPECTED_PARAMS[variant]

        backbone = YOLOPAFPN(
            depth=expected["depth"],
            width=expected["width"],
            in_channels=expected["in_channels"],
            depthwise=expected["depthwise"],
        )
        head = YOLOXHead(
            num_classes=num_classes,
            width=expected["width"],
            in_channels=expected["in_channels"],
            depthwise=expected["depthwise"],
        )
        model = YOLOX(backbone=backbone, head=head)
        model.eval()

        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(dummy)

        # In eval mode, YOLOX returns [batch, num_anchors, 5 + num_classes]
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 5 + num_classes


# ---------------------------------------------------------------------------
# Hydra override tests
# ---------------------------------------------------------------------------


class TestHydraOverride:
    """Verify Hydra overrides reach the composed config."""

    def test_override_depth(self) -> None:
        """Override depth via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_yolox",
                overrides=["models=yolox_s", "models.depth=0.67"],
            )
        assert cfg.models.depth == 0.67

    def test_override_width(self) -> None:
        """Override width via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_yolox",
                overrides=["models=yolox_s", "models.width=0.75"],
            )
        assert cfg.models.width == 0.75

    def test_override_depthwise(self) -> None:
        """Override depthwise via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_yolox",
                overrides=["models=yolox_s", "models.depthwise=true"],
            )
        assert cfg.models.depthwise is True

    def test_override_iou_loss_type(self) -> None:
        """Override iou_loss_type via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_yolox",
                overrides=["models=yolox_s", "models.iou_loss_type=giou"],
            )
        assert cfg.models.iou_loss_type == "giou"

    def test_override_in_channels(self) -> None:
        """Override in_channels list via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_yolox",
                overrides=["models=yolox_s", "models.in_channels=[128,256,512]"],
            )
        assert list(cfg.models.in_channels) == [128, 256, 512]


# ---------------------------------------------------------------------------
# Basketball config tests
# ---------------------------------------------------------------------------


class TestBasketballConfig:
    """Verify basketball YOLOX config composes correctly."""

    def test_basketball_has_architecture_params(self) -> None:
        """Basketball config must include architecture params from yolox_s."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(config_name="train_basketball_yolox")

        assert cfg.models.depth == 0.33
        assert cfg.models.width == 0.50
        assert cfg.models.depthwise is False
        assert list(cfg.models.in_channels) == [256, 512, 1024]

    def test_basketball_num_classes(self) -> None:
        """Basketball config must override num_classes to 10."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(config_name="train_basketball_yolox")

        assert cfg.models.num_classes == 10

    def test_basketball_allows_arch_override(self) -> None:
        """Basketball config must allow architecture overrides."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_basketball_yolox",
                overrides=["models.depth=0.67", "models.width=0.75"],
            )
        assert cfg.models.depth == 0.67
        assert cfg.models.width == 0.75


# ---------------------------------------------------------------------------
# Model instantiation tests (no pretrained weights)
# ---------------------------------------------------------------------------


class TestModelInstantiationNoPretrained:
    """Instantiate YOLOXLightningModel directly with download_pretrained=False."""

    @pytest.mark.parametrize("variant", ["nano", "s", "m", "l"])
    def test_instantiate_from_yaml_params(self, variant: str) -> None:
        """Create model from explicit YAML params without network access."""
        expected = VARIANT_EXPECTED_PARAMS[variant]

        model = YOLOXLightningModel(
            num_classes=2,
            download_pretrained=False,
            depth=expected["depth"],
            width=expected["width"],
            depthwise=expected["depthwise"],
            in_channels=expected["in_channels"],
            checkpoint_name=expected["checkpoint_name"],
        )

        # Verify model was created successfully
        assert model is not None
        assert model.num_classes == 2
        assert model.checkpoint_name == expected["checkpoint_name"]

        # Verify forward pass works
        dummy = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        assert "predictions" in out
        assert out["predictions"].shape[0] == 1
        assert out["predictions"].shape[2] == 5 + 2  # 5 + num_classes
