"""Tests for RFDETR YAML parameterization.

Validates that architecture parameters from Hydra YAML configs produce
identical models to the previous Pydantic config class approach.
"""

from __future__ import annotations

import sys
from os.path import dirname, join

import hydra
import pytest
import torch

# Register model configs by importing the wrappers
sys.path.append(join(dirname(__file__), "../src"))
import object_detection_training.models as _  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Variant parameter definitions (source of truth: Pydantic config classes)
# ---------------------------------------------------------------------------

# These match RFDETRXxxConfig.model_dump() exactly
VARIANT_EXPECTED_PARAMS = {
    "nano": {
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "patch_size": 16,
        "num_windows": 2,
        "dec_layers": 2,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "two_stage": True,
        "bbox_reparam": True,
        "lite_refpoint_refine": True,
        "layer_norm": True,
        "amp": True,
        "group_detr": 13,
        "gradient_checkpointing": False,
        "ia_bce_loss": True,
        "cls_loss_coef": 1.0,
        "segmentation_head": False,
        "mask_downsample_ratio": 4,
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_select": 300,
        "out_feature_indexes": [3, 6, 9, 12],
        "resolution": 384,
        "positional_encoding_size": 24,
        "checkpoint_name": "rf-detr-nano.pth",
    },
    "small": {
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "patch_size": 16,
        "num_windows": 2,
        "dec_layers": 3,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "two_stage": True,
        "bbox_reparam": True,
        "lite_refpoint_refine": True,
        "layer_norm": True,
        "amp": True,
        "group_detr": 13,
        "gradient_checkpointing": False,
        "ia_bce_loss": True,
        "cls_loss_coef": 1.0,
        "segmentation_head": False,
        "mask_downsample_ratio": 4,
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_select": 300,
        "out_feature_indexes": [3, 6, 9, 12],
        "resolution": 512,
        "positional_encoding_size": 32,
        "checkpoint_name": "rf-detr-small.pth",
    },
    "medium": {
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "patch_size": 16,
        "num_windows": 2,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "two_stage": True,
        "bbox_reparam": True,
        "lite_refpoint_refine": True,
        "layer_norm": True,
        "amp": True,
        "group_detr": 13,
        "gradient_checkpointing": False,
        "ia_bce_loss": True,
        "cls_loss_coef": 1.0,
        "segmentation_head": False,
        "mask_downsample_ratio": 4,
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_select": 300,
        "out_feature_indexes": [3, 6, 9, 12],
        "resolution": 576,
        "positional_encoding_size": 36,
        "checkpoint_name": "rf-detr-medium.pth",
    },
    "large": {
        "encoder": "dinov2_windowed_base",
        "hidden_dim": 384,
        "patch_size": 14,
        "num_windows": 4,
        "dec_layers": 3,
        "sa_nheads": 12,
        "ca_nheads": 24,
        "dec_n_points": 4,
        "two_stage": True,
        "bbox_reparam": True,
        "lite_refpoint_refine": True,
        "layer_norm": True,
        "amp": True,
        "group_detr": 13,
        "gradient_checkpointing": False,
        "ia_bce_loss": True,
        "cls_loss_coef": 1.0,
        "segmentation_head": False,
        "mask_downsample_ratio": 4,
        "projector_scale": ["P3", "P5"],
        "num_queries": 300,
        "num_select": 300,
        "out_feature_indexes": [2, 5, 8, 11],
        "resolution": 560,
        "positional_encoding_size": 37,
        "checkpoint_name": "rf-detr-large.pth",
    },
}

VARIANT_YAML_MAP = {
    "nano": "rfdetr_nano",
    "small": "rfdetr_small",
    "medium": "rfdetr_medium",
    "large": "rfdetr_large",
}


# ---------------------------------------------------------------------------
# Hydra config completeness tests
# ---------------------------------------------------------------------------


class TestHydraConfigCompleteness:
    """Verify YAML configs contain all required architecture parameters."""

    @pytest.mark.parametrize("variant", list(VARIANT_EXPECTED_PARAMS.keys()))
    def test_variant_has_all_arch_params(self, variant):
        """Each variant YAML must contain all architecture parameters."""
        yaml_name = VARIANT_YAML_MAP[variant]
        expected = VARIANT_EXPECTED_PARAMS[variant]

        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(config_name="train", overrides=[f"models={yaml_name}"])

        for key, expected_val in expected.items():
            assert key in cfg.models, (
                f"Missing param '{key}' in {yaml_name}.yaml config"
            )
            actual = cfg.models[key]
            # Convert OmegaConf lists to plain lists for comparison
            if hasattr(actual, "__iter__") and not isinstance(actual, str):
                actual = list(actual)
            assert actual == expected_val, (
                f"Param '{key}' mismatch in {yaml_name}: "
                f"expected={expected_val!r}, got={actual!r}"
            )

    def test_base_params_present(self):
        """rfdetr_base.yaml must define common loss/matcher/scheduler params."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(config_name="train", overrides=["models=rfdetr_small"])

        base_params = [
            "set_cost_class",
            "set_cost_bbox",
            "set_cost_giou",
            "bbox_loss_coef",
            "giou_loss_coef",
            "focal_alpha",
            "aux_loss",
            "warmup_start_factor",
            "cosine_eta_min_factor",
            "dim_feedforward",
            "decoder_norm",
            "vit_encoder_num_layers",
            "position_embedding",
            "amp",
            "gradient_checkpointing",
        ]
        for param in base_params:
            assert param in cfg.models, f"Missing base param '{param}' in rfdetr config"


# ---------------------------------------------------------------------------
# YAML-to-Pydantic equivalence tests
# ---------------------------------------------------------------------------


class TestYamlPydanticEquivalence:
    """Verify YAML params match Pydantic config class values."""

    @pytest.mark.parametrize("variant", list(VARIANT_EXPECTED_PARAMS.keys()))
    def test_yaml_matches_pydantic_config(self, variant):
        """YAML architecture params must match Pydantic config model_dump()."""
        import importlib

        config_mod = importlib.import_module(
            "object_detection_training.models.rfdetr.config"
        )

        config_class_map = {
            "nano": "RFDETRNanoConfig",
            "small": "RFDETRSmallConfig",
            "medium": "RFDETRMediumConfig",
            "large": "RFDETRLargeConfig",
        }
        config_cls = getattr(config_mod, config_class_map[variant])
        pydantic_config = config_cls()
        pydantic_dict = pydantic_config.model_dump()

        yaml_name = VARIANT_YAML_MAP[variant]
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(config_name="train", overrides=[f"models={yaml_name}"])

        # Check architecture params that exist in both Pydantic and YAML
        arch_params = [
            "encoder",
            "hidden_dim",
            "dec_layers",
            "patch_size",
            "num_windows",
            "sa_nheads",
            "ca_nheads",
            "dec_n_points",
            "two_stage",
            "bbox_reparam",
            "lite_refpoint_refine",
            "layer_norm",
            "amp",
            "group_detr",
            "gradient_checkpointing",
            "ia_bce_loss",
            "cls_loss_coef",
            "segmentation_head",
            "mask_downsample_ratio",
            "num_queries",
            "num_select",
            "resolution",
            "positional_encoding_size",
        ]

        for param in arch_params:
            if param not in pydantic_dict:
                continue
            pydantic_val = pydantic_dict[param]
            yaml_val = cfg.models[param]
            if hasattr(yaml_val, "__iter__") and not isinstance(yaml_val, str):
                yaml_val = list(yaml_val)
            assert yaml_val == pydantic_val, (
                f"Mismatch for {variant}/{param}: "
                f"pydantic={pydantic_val!r}, yaml={yaml_val!r}"
            )

    @pytest.mark.parametrize("variant", list(VARIANT_EXPECTED_PARAMS.keys()))
    def test_yaml_projector_scale_matches_pydantic(self, variant):
        """projector_scale list must match Pydantic config."""
        import importlib

        config_mod = importlib.import_module(
            "object_detection_training.models.rfdetr.config"
        )
        config_class_map = {
            "nano": "RFDETRNanoConfig",
            "small": "RFDETRSmallConfig",
            "medium": "RFDETRMediumConfig",
            "large": "RFDETRLargeConfig",
        }
        config_cls = getattr(config_mod, config_class_map[variant])
        pydantic_scale = config_cls().projector_scale

        yaml_name = VARIANT_YAML_MAP[variant]
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(config_name="train", overrides=[f"models={yaml_name}"])
        yaml_scale = list(cfg.models.projector_scale)
        assert yaml_scale == pydantic_scale, (
            f"projector_scale mismatch for {variant}: "
            f"pydantic={pydantic_scale}, yaml={yaml_scale}"
        )

    @pytest.mark.parametrize("variant", list(VARIANT_EXPECTED_PARAMS.keys()))
    def test_yaml_out_feature_indexes_matches_pydantic(self, variant):
        """out_feature_indexes list must match Pydantic config."""
        import importlib

        config_mod = importlib.import_module(
            "object_detection_training.models.rfdetr.config"
        )
        config_class_map = {
            "nano": "RFDETRNanoConfig",
            "small": "RFDETRSmallConfig",
            "medium": "RFDETRMediumConfig",
            "large": "RFDETRLargeConfig",
        }
        config_cls = getattr(config_mod, config_class_map[variant])
        pydantic_indexes = config_cls().out_feature_indexes

        yaml_name = VARIANT_YAML_MAP[variant]
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(config_name="train", overrides=[f"models={yaml_name}"])
        yaml_indexes = list(cfg.models.out_feature_indexes)
        assert yaml_indexes == pydantic_indexes, (
            f"out_feature_indexes mismatch for {variant}: "
            f"pydantic={pydantic_indexes}, yaml={yaml_indexes}"
        )


# ---------------------------------------------------------------------------
# Model architecture equivalence tests
# ---------------------------------------------------------------------------


class TestModelArchitectureEquivalence:
    """Verify YAML-parameterized models match Pydantic-config models."""

    @pytest.mark.parametrize("variant", ["small"])
    def test_yaml_model_matches_pydantic_model(self, variant):
        """Model built from YAML params must have identical state_dict
        keys and shapes as model built from Pydantic config."""
        import importlib

        from object_detection_training.models.rfdetr.model_factory import Model

        # Build via Pydantic config (old path)
        config_mod = importlib.import_module(
            "object_detection_training.models.rfdetr.config"
        )
        config_class_map = {
            "nano": "RFDETRNanoConfig",
            "small": "RFDETRSmallConfig",
            "medium": "RFDETRMediumConfig",
            "large": "RFDETRLargeConfig",
        }
        config_cls = getattr(config_mod, config_class_map[variant])
        config = config_cls(pretrain_weights=None, num_classes=2)
        pydantic_model = Model(**config.model_dump())
        pydantic_state = pydantic_model.model.state_dict()

        # Build via YAML params (new path) - must include device
        expected = VARIANT_EXPECTED_PARAMS[variant]
        yaml_params = {k: v for k, v in expected.items() if k != "checkpoint_name"}
        yaml_params["num_classes"] = 2
        yaml_params["pretrain_weights"] = None
        # Match Pydantic config's device auto-detection
        if torch.cuda.is_available():
            yaml_params["device"] = "cuda"
        elif torch.backends.mps.is_available():
            yaml_params["device"] = "mps"
        else:
            yaml_params["device"] = "cpu"
        yaml_model = Model(**yaml_params)
        yaml_state = yaml_model.model.state_dict()

        # Compare keys
        assert set(pydantic_state.keys()) == set(yaml_state.keys()), (
            f"Key mismatch for {variant}: "
            f"pydantic_only={set(pydantic_state.keys()) - set(yaml_state.keys())}, "
            f"yaml_only={set(yaml_state.keys()) - set(pydantic_state.keys())}"
        )

        # Compare shapes
        for key in pydantic_state:
            assert pydantic_state[key].shape == yaml_state[key].shape, (
                f"Shape mismatch for {variant}/{key}: "
                f"pydantic={pydantic_state[key].shape} vs "
                f"yaml={yaml_state[key].shape}"
            )

    @pytest.mark.parametrize("variant", ["small"])
    def test_forward_shape_yaml_params(self, variant):
        """Forward pass with YAML params produces correct output shapes."""
        from object_detection_training.models.rfdetr.lwdetr import build_model
        from object_detection_training.models.rfdetr.model_factory import populate_args

        expected = VARIANT_EXPECTED_PARAMS[variant]
        params = {k: v for k, v in expected.items() if k != "checkpoint_name"}
        params["num_classes"] = 2
        params["pretrain_weights"] = None

        args = populate_args(**params)
        model = build_model(args)
        model.eval()

        resolution = expected["resolution"]
        dummy = torch.randn(1, 3, resolution, resolution)
        with torch.no_grad():
            out = model(dummy)

        assert "pred_logits" in out
        assert "pred_boxes" in out
        assert out["pred_logits"].shape == (1, 300, 3)  # num_classes + 1
        assert out["pred_boxes"].shape == (1, 300, 4)


# ---------------------------------------------------------------------------
# Scheduler params configurable tests
# ---------------------------------------------------------------------------


class TestSchedulerParamsConfigurable:
    """Verify scheduler params are passed through from YAML."""

    def test_warmup_start_factor_stored(self):
        """warmup_start_factor must be stored on the model instance."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(config_name="train", overrides=["models=rfdetr_small"])
        assert cfg.models.warmup_start_factor == 1e-3

    def test_cosine_eta_min_factor_stored(self):
        """cosine_eta_min_factor must be stored on the model instance."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(config_name="train", overrides=["models=rfdetr_small"])
        assert cfg.models.cosine_eta_min_factor == 0.05


# ---------------------------------------------------------------------------
# Hydra override tests
# ---------------------------------------------------------------------------


class TestHydraOverride:
    """Verify Hydra overrides reach the composed config."""

    def test_override_hidden_dim(self):
        """Override hidden_dim via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(
                config_name="train",
                overrides=["models=rfdetr_small", "models.hidden_dim=384"],
            )
        assert cfg.models.hidden_dim == 384

    def test_override_dec_layers(self):
        """Override dec_layers via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(
                config_name="train",
                overrides=["models=rfdetr_small", "models.dec_layers=6"],
            )
        assert cfg.models.dec_layers == 6

    def test_override_scheduler_params(self):
        """Override scheduler params via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(
                config_name="train",
                overrides=[
                    "models=rfdetr_small",
                    "models.warmup_start_factor=0.01",
                    "models.cosine_eta_min_factor=0.1",
                ],
            )
        assert cfg.models.warmup_start_factor == 0.01
        assert cfg.models.cosine_eta_min_factor == 0.1

    def test_override_loss_coef(self):
        """Override loss coefficient via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(
                config_name="train",
                overrides=["models=rfdetr_small", "models.cls_loss_coef=2.0"],
            )
        assert cfg.models.cls_loss_coef == 2.0

    def test_override_projector_scale(self):
        """Override projector_scale list via Hydra CLI."""
        with hydra.initialize(version_base=None, config_path="../conf"):
            cfg = hydra.compose(
                config_name="train",
                overrides=[
                    "models=rfdetr_small",
                    "models.projector_scale=[P3,P5]",
                ],
            )
        assert list(cfg.models.projector_scale) == ["P3", "P5"]
