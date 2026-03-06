"""Tests for DINOXConfig Pydantic validation.

Verifies that DINOXConfig validates defaults, bounds, invalid flag
combinations, and immutability.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from object_detection_training.models.dinox import DINOXConfig


class TestDINOXConfigDefaults:
    """Verify default DINOXConfig behaves as YOLOX-M equivalent."""

    def test_all_defaults_off(self) -> None:
        """DINOXConfig() creates valid config with all improvements off."""
        cfg = DINOXConfig()
        assert cfg.use_dfl is False
        assert cfg.use_soft_labels is False
        assert cfg.use_mal is False
        assert cfg.use_dual_head is False
        assert cfg.enable_distillation is False
        assert cfg.use_scheduler_free is False

    def test_frozen_config(self) -> None:
        """DINOXConfig is immutable (frozen Pydantic model)."""
        cfg = DINOXConfig()
        with pytest.raises(ValidationError):
            cfg.use_dfl = True  # type: ignore[misc]


class TestDINOXConfigValidation:
    """Verify DINOXConfig validates input constraints."""

    def test_valid_dfl_config(self) -> None:
        """DINOXConfig with use_dfl=True and reg_max=16 succeeds."""
        cfg = DINOXConfig(use_dfl=True, reg_max=16)
        assert cfg.use_dfl is True
        assert cfg.reg_max == 16

    def test_reg_max_too_low(self) -> None:
        """reg_max=0 raises ValueError."""
        with pytest.raises(ValidationError, match="reg_max"):
            DINOXConfig(reg_max=0)

    def test_reg_max_too_high(self) -> None:
        """reg_max=33 raises ValueError."""
        with pytest.raises(ValidationError, match="reg_max"):
            DINOXConfig(reg_max=33)

    def test_reg_max_lower_bound(self) -> None:
        """reg_max=1 is valid."""
        cfg = DINOXConfig(reg_max=1)
        assert cfg.reg_max == 1

    def test_reg_max_upper_bound(self) -> None:
        """reg_max=32 is valid."""
        cfg = DINOXConfig(reg_max=32)
        assert cfg.reg_max == 32

    def test_mal_requires_soft_labels(self) -> None:
        """use_mal=True without use_soft_labels=True raises ValueError."""
        with pytest.raises(ValidationError, match="use_mal"):
            DINOXConfig(use_mal=True, use_soft_labels=False)

    def test_mal_with_soft_labels_valid(self) -> None:
        """use_mal=True with use_soft_labels=True succeeds."""
        cfg = DINOXConfig(use_mal=True, use_soft_labels=True)
        assert cfg.use_mal is True
        assert cfg.use_soft_labels is True

    def test_dfl_loss_weight_accepts_float(self) -> None:
        """dfl_loss_weight=1.5 succeeds."""
        cfg = DINOXConfig(dfl_loss_weight=1.5)
        assert cfg.dfl_loss_weight == 1.5


class TestDINOXConfigPhase1Equivalence:
    """Verify defaults produce standard YOLOX-M behavior."""

    def test_default_config_matches_yolox(self) -> None:
        """Default DINOXConfig matches standard YOLOX behavior."""
        cfg = DINOXConfig()
        assert cfg.use_dfl is False
        assert cfg.iou_loss_type == "iou"
        assert cfg.assigner == "simota"
