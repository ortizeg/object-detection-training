"""RFDETR module."""

from object_detection_training.models.rfdetr.coco import (
    CocoDetection,
    ConvertCoco,
    collate_fn,
    compute_multi_scale_scales,
    make_coco_transforms,
)
from object_detection_training.models.rfdetr.lwdetr import (
    PostProcess,
    build_criterion_and_postprocessors,
    build_model,
)
from object_detection_training.models.rfdetr.transforms import (
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
    # Model
    "build_model",
    "build_criterion_and_postprocessors",
    "PostProcess",
]
