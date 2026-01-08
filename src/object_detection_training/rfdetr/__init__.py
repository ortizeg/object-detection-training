# ------------------------------------------------------------------------
# Copyright (c) 2024 Roboflow. All Rights Reserved.
# Original source: https://github.com/roboflow/rf-detr
# This code is copied for use in this framework with original license.
# ------------------------------------------------------------------------
"""RFDETR datasets module."""

from object_detection_training.rfdetr.coco import (
    CocoDetection,
    ConvertCoco,
    collate_fn,
    compute_multi_scale_scales,
    make_coco_transforms,
)
from object_detection_training.rfdetr.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResize,
    RandomSelect,
    RandomSizeCrop,
    SquareResize,
    ToTensor,
)

__all__ = [
    # COCO
    "CocoDetection",
    "ConvertCoco",
    "collate_fn",
    "compute_multi_scale_scales",
    "make_coco_transforms",
    # Transforms
    "Compose",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomResize",
    "RandomSelect",
    "RandomSizeCrop",
    "SquareResize",
    "ToTensor",
]
