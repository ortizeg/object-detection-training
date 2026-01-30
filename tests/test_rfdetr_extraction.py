"""Tests for local rfdetr model architecture extraction.

Validates that the local copy of the rfdetr model architecture produces
identical models to the original rfdetr PyPI package.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestImportSmoke:
    """Verify all local rfdetr modules import correctly."""

    def test_import_config(self):
        from object_detection_training.models.rfdetr.config import (
            ModelConfig,
            RFDETRBaseConfig,
            RFDETRLargeConfig,
            RFDETRMediumConfig,
            RFDETRNanoConfig,
            RFDETRSmallConfig,
        )

    def test_import_model_factory(self):
        from object_detection_training.models.rfdetr.model_factory import (
            HOSTED_MODELS,
            Model,
            populate_args,
        )

    def test_import_lwdetr(self):
        from object_detection_training.models.rfdetr.lwdetr import (
            LWDETR,
            MLP,
            PostProcess,
            SetCriterion,
            build_criterion_and_postprocessors,
            build_model,
        )

    def test_import_transformer(self):
        from object_detection_training.models.rfdetr.transformer import (
            Transformer,
            TransformerDecoder,
            TransformerDecoderLayer,
            build_transformer,
        )

    def test_import_matcher(self):
        from object_detection_training.models.rfdetr.matcher import (
            HungarianMatcher,
            build_matcher,
        )

    def test_import_backbone(self):
        from object_detection_training.models.rfdetr.backbone import (
            Joiner,
            build_backbone,
        )

    def test_import_position_encoding(self):
        from object_detection_training.models.rfdetr.position_encoding import (
            build_position_encoding,
        )

    def test_import_segmentation_head(self):
        from object_detection_training.models.rfdetr.segmentation_head import (
            SegmentationHead,
        )

    def test_import_util(self):
        from object_detection_training.models.rfdetr.util.misc import (
            NestedTensor,
            inverse_sigmoid,
        )
        from object_detection_training.models.rfdetr.util.box_ops import (
            box_cxcywh_to_xyxy,
            generalized_box_iou,
        )

    def test_import_ops(self):
        from object_detection_training.models.rfdetr.ops.modules import MSDeformAttn

    def test_import_lightning_model(self):
        from object_detection_training.models.rfdetr_lightning import (
            RFDETRLargeModel,
            RFDETRLightningModel,
            RFDETRMediumModel,
            RFDETRNanoModel,
            RFDETRSmallModel,
        )


# ---------------------------------------------------------------------------
# Config equivalence tests
# ---------------------------------------------------------------------------

class TestConfigEquivalence:
    """Verify config classes produce expected parameters."""

    def test_nano_config(self):
        from object_detection_training.models.rfdetr.config import RFDETRNanoConfig
        cfg = RFDETRNanoConfig()
        assert cfg.hidden_dim == 256
        assert cfg.dec_layers == 2
        assert cfg.resolution == 384
        assert cfg.patch_size == 16
        assert cfg.num_windows == 2
        assert cfg.encoder == "dinov2_windowed_small"

    def test_small_config(self):
        from object_detection_training.models.rfdetr.config import RFDETRSmallConfig
        cfg = RFDETRSmallConfig()
        assert cfg.hidden_dim == 256
        assert cfg.dec_layers == 3
        assert cfg.resolution == 512
        assert cfg.patch_size == 16

    def test_medium_config(self):
        from object_detection_training.models.rfdetr.config import RFDETRMediumConfig
        cfg = RFDETRMediumConfig()
        assert cfg.hidden_dim == 256
        assert cfg.dec_layers == 4
        assert cfg.resolution == 576

    def test_large_config(self):
        from object_detection_training.models.rfdetr.config import RFDETRLargeConfig
        cfg = RFDETRLargeConfig()
        assert cfg.hidden_dim == 384
        assert cfg.dec_layers == 3
        assert cfg.resolution == 560
        assert cfg.encoder == "dinov2_windowed_base"
        assert cfg.sa_nheads == 12
        assert cfg.ca_nheads == 24


# ---------------------------------------------------------------------------
# populate_args tests
# ---------------------------------------------------------------------------

class TestPopulateArgs:
    """Verify populate_args produces correct namespace."""

    def test_populate_args_defaults(self):
        from object_detection_training.models.rfdetr.model_factory import populate_args
        args = populate_args()
        assert args.hidden_dim == 256
        assert args.dec_layers == 3
        assert args.group_detr == 13
        assert args.aux_loss is True

    def test_populate_args_custom(self):
        from object_detection_training.models.rfdetr.model_factory import populate_args
        args = populate_args(
            hidden_dim=384, dec_layers=4, num_classes=20, resolution=640
        )
        assert args.hidden_dim == 384
        assert args.dec_layers == 4
        assert args.num_classes == 20
        assert args.resolution == 640

    def test_populate_args_extra_kwargs(self):
        """Extra kwargs should pass through to Namespace."""
        from object_detection_training.models.rfdetr.model_factory import populate_args
        args = populate_args(
            patch_size=16, num_windows=2, positional_encoding_size=32
        )
        assert args.patch_size == 16
        assert args.num_windows == 2
        assert args.positional_encoding_size == 32


# ---------------------------------------------------------------------------
# Model architecture tests (no pretrained weights)
# ---------------------------------------------------------------------------

class TestModelArchitecture:
    """Test model instantiation without pretrained weights."""

    @pytest.fixture
    def small_args(self):
        from object_detection_training.models.rfdetr.config import RFDETRSmallConfig
        from object_detection_training.models.rfdetr.model_factory import populate_args
        config = RFDETRSmallConfig(pretrain_weights=None, num_classes=2)
        return populate_args(**config.model_dump())

    def test_build_model_smoke(self, small_args):
        """build_model returns an LWDETR nn.Module."""
        from object_detection_training.models.rfdetr.lwdetr import LWDETR, build_model
        model = build_model(small_args)
        assert isinstance(model, LWDETR)

    def test_criterion_weight_dict(self, small_args):
        """Criterion weight_dict has expected keys."""
        from object_detection_training.models.rfdetr.lwdetr import (
            build_criterion_and_postprocessors,
        )
        criterion, postprocess = build_criterion_and_postprocessors(small_args)
        wd = criterion.weight_dict
        assert "loss_ce" in wd
        assert "loss_bbox" in wd
        assert "loss_giou" in wd
        # aux losses for dec_layers - 1 = 2 intermediate layers
        assert "loss_ce_0" in wd
        assert "loss_bbox_1" in wd

    def test_forward_shape(self, small_args):
        """Forward pass returns correct output keys and shapes."""
        from object_detection_training.models.rfdetr.lwdetr import build_model
        model = build_model(small_args)
        model.eval()
        resolution = small_args.resolution
        dummy = torch.randn(1, 3, resolution, resolution)
        with torch.no_grad():
            out = model(dummy)
        assert "pred_logits" in out
        assert "pred_boxes" in out
        # num_queries = 300, num_classes + 1 = 3
        assert out["pred_logits"].shape == (1, 300, 3)
        assert out["pred_boxes"].shape == (1, 300, 4)


# ---------------------------------------------------------------------------
# Checkpoint equivalence tests (parametrized across all 4 variants)
# ---------------------------------------------------------------------------

VARIANT_CONFIGS = {
    "nano": {
        "config_module": "object_detection_training.models.rfdetr.config",
        "config_class": "RFDETRNanoConfig",
        "rfdetr_class": "RFDETRNano",
        "checkpoint_name": "rf-detr-nano.pth",
    },
    "small": {
        "config_module": "object_detection_training.models.rfdetr.config",
        "config_class": "RFDETRSmallConfig",
        "rfdetr_class": "RFDETRSmall",
        "checkpoint_name": "rf-detr-small.pth",
    },
    "medium": {
        "config_module": "object_detection_training.models.rfdetr.config",
        "config_class": "RFDETRMediumConfig",
        "rfdetr_class": "RFDETRMedium",
        "checkpoint_name": "rf-detr-medium.pth",
    },
    "large": {
        "config_module": "object_detection_training.models.rfdetr.config",
        "config_class": "RFDETRLargeConfig",
        "rfdetr_class": "RFDETRLarge",
        "checkpoint_name": "rf-detr-large.pth",
    },
}


@pytest.mark.parametrize("variant", list(VARIANT_CONFIGS.keys()))
class TestCheckpointEquivalence:
    """Verify checkpoint-loaded models are bitwise identical between
    the local code and the rfdetr PyPI package for all 4 sizes."""

    @pytest.fixture
    def setup_models(self, variant, tmp_path):
        """Build model via both local code and rfdetr package, load checkpoint."""
        import importlib
        from pathlib import Path

        info = VARIANT_CONFIGS[variant]

        # Skip if rfdetr package not available (post-removal)
        try:
            rfdetr_pkg = importlib.import_module("rfdetr")
        except ImportError:
            pytest.skip("rfdetr package not installed; skipping equivalence test")

        # Get rfdetr package class
        rfdetr_class = getattr(rfdetr_pkg, info["rfdetr_class"])

        # Download checkpoint to cache
        cache_dir = Path.home() / ".cache" / "rfdetr"
        checkpoint_path = cache_dir / info["checkpoint_name"]
        if not checkpoint_path.exists():
            from object_detection_training.models.rfdetr.model_factory import (
                download_pretrain_weights,
            )
            download_pretrain_weights(info["checkpoint_name"])

        # Build via rfdetr PyPI package
        pkg_wrapper = rfdetr_class(
            pretrain_weights=str(checkpoint_path),
            num_classes=90,
        )
        pkg_model = pkg_wrapper.model.model  # LWDETR nn.Module
        pkg_state = pkg_model.state_dict()

        # Build via local code
        config_mod = importlib.import_module(info["config_module"])
        config_cls = getattr(config_mod, info["config_class"])
        config = config_cls(
            pretrain_weights=str(checkpoint_path),
            num_classes=90,
        )

        from object_detection_training.models.rfdetr.model_factory import Model
        local_model_wrapper = Model(**config.model_dump())
        local_model = local_model_wrapper.model  # LWDETR nn.Module
        local_state = local_model.state_dict()

        return pkg_state, local_state

    def test_identical_keys(self, setup_models, variant):
        """state_dict keys are identical."""
        pkg_state, local_state = setup_models
        assert set(pkg_state.keys()) == set(local_state.keys()), (
            f"Key mismatch for {variant}: "
            f"pkg_only={set(pkg_state.keys()) - set(local_state.keys())}, "
            f"local_only={set(local_state.keys()) - set(pkg_state.keys())}"
        )

    def test_identical_shapes(self, setup_models, variant):
        """All parameter tensors have identical shapes."""
        pkg_state, local_state = setup_models
        for key in pkg_state:
            assert pkg_state[key].shape == local_state[key].shape, (
                f"Shape mismatch for {variant}/{key}: "
                f"pkg={pkg_state[key].shape} vs local={local_state[key].shape}"
            )

    def test_identical_values(self, setup_models, variant):
        """All parameter values are bitwise identical (torch.allclose)."""
        pkg_state, local_state = setup_models
        mismatches = []
        for key in pkg_state:
            if not torch.allclose(pkg_state[key], local_state[key], atol=0, rtol=0):
                diff = (pkg_state[key] - local_state[key]).abs().max().item()
                mismatches.append(f"{key}: max_diff={diff}")
        assert len(mismatches) == 0, (
            f"Value mismatches for {variant}:\n" + "\n".join(mismatches)
        )
