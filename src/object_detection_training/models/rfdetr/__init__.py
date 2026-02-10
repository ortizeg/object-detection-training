from __future__ import annotations

# ------------------------------------------------------------------------
# Copyright (c) 2024 Roboflow. All Rights Reserved.
# Original source: https://github.com/roboflow/rf-detr
# This code is copied for use in this framework with original license.
# ------------------------------------------------------------------------
"""RFDETR datasets and model architecture module."""

from object_detection_training.models.rfdetr.coco import (
    CocoDetection,
    ConvertCoco,
    collate_fn,
)
from object_detection_training.transforms import compute_multi_scale_scales

# Model architecture
from object_detection_training.models.rfdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
)
from object_detection_training.models.rfdetr.model_factory import (
    Model,
    populate_args,
)
from object_detection_training.models.rfdetr.lwdetr import (
    build_criterion_and_postprocessors,
    build_model,
)

__all__ = [
    # COCO
    "CocoDetection",
    "ConvertCoco",
    "collate_fn",
    "compute_multi_scale_scales",
    # Model architecture
    "ModelConfig",
    "RFDETRBaseConfig",
    "RFDETRNanoConfig",
    "RFDETRSmallConfig",
    "RFDETRMediumConfig",
    "RFDETRLargeConfig",
    "Model",
    "populate_args",
    "build_model",
    "build_criterion_and_postprocessors",
]
