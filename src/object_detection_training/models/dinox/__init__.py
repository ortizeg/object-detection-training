"""DINOX: Enhanced YOLOX with Distribution Focal Loss and progressive improvements."""

from __future__ import annotations

from .config import DINOXConfig
from .dfl import DFLModule, distribution_focal_loss
from .dinox import DINOX
from .dinox_head import DINOXHead
from .hungarian import HungarianAssigner
from .mal import mal_weight, matchability_score
from .tal import TaskAlignedAssigner

__all__ = [
    "DINOX",
    "DFLModule",
    "DINOXConfig",
    "DINOXHead",
    "HungarianAssigner",
    "TaskAlignedAssigner",
    "distribution_focal_loss",
    "mal_weight",
    "matchability_score",
]
