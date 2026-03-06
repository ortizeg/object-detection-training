"""DINOX configuration with feature flags for progressive improvements."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DINOXConfig(BaseModel, frozen=True):
    """Configuration for DINOX model improvements.

    All improvement flags default to False/off, so a default DINOXConfig
    produces a model functionally equivalent to standard YOLOX-M.

    Attributes:
        use_dfl: Enable Distribution Focal Loss regression.
        reg_max: Maximum discrete bin index for DFL (output is reg_max+1 bins).
        dfl_loss_weight: Weight for DFL loss term.
        use_soft_labels: Enable soft label assignment targets.
        soft_label_gamma: Gamma for soft label quality weighting.
        use_log_iou_cost: Enable -log(IoU) regression cost in SimOTA assignment.
        use_mal: Enable Mutual Assistance Learning between heads.
        mal_gamma: Gamma for MAL loss scaling.
        use_dual_head: Enable dual detection head architecture.
        enable_distillation: Enable knowledge distillation from teacher.
        use_scheduler_free: Use scheduler-free optimizer wrapper.
        assigner: Label assignment strategy.
        iou_loss_type: IoU loss variant for regression.
    """

    # Phase 1: DFL
    use_dfl: bool = False
    reg_max: int = 16
    dfl_loss_weight: float = 0.25

    # Phase 2: soft labels
    use_soft_labels: bool = False
    soft_label_gamma: float = 2.0
    use_log_iou_cost: bool = False

    # Future phases: MAL
    use_mal: bool = False
    mal_gamma: float = 1.5

    # Future phases: dual head
    use_dual_head: bool = False
    lambda_o2o: float = 1.0

    # Phase 5: distillation
    enable_distillation: bool = False
    distill_weight: float = 0.5
    distill_layer_indices: list[int] = Field(default_factory=lambda: [3, 7, 11])
    distill_teacher: str = "dinov2_vitb14"

    # Future phases: scheduler-free
    use_scheduler_free: bool = False

    # Assignment and loss configuration
    assigner: Literal["simota", "tal"] = "simota"
    iou_loss_type: Literal["iou", "giou"] = "iou"

    # TAL hyperparameters
    tal_topk: int = 13
    tal_alpha: float = 1.0
    tal_beta: float = 6.0

    @field_validator("reg_max")
    @classmethod
    def validate_reg_max(cls, v: int) -> int:
        """Validate reg_max is in [1, 32]."""
        if v < 1 or v > 32:
            msg = f"reg_max must be in [1, 32], got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_flag_combinations(self) -> DINOXConfig:
        """Validate that dependent flags are enabled together."""
        if self.use_mal and not self.use_soft_labels:
            msg = (
                "use_mal=True requires use_soft_labels=True "
                "(MAL depends on soft label quality scores)"
            )
            raise ValueError(msg)
        if self.use_dual_head and not self.use_soft_labels:
            msg = (
                "use_dual_head=True requires use_soft_labels=True "
                "(dual head depends on soft label targets)"
            )
            raise ValueError(msg)
        return self
