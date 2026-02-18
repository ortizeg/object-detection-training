"""Schemas for object detection training."""

from object_detection_training.schemas.annotation import DetectionAnnotation
from object_detection_training.schemas.crop_manifest import CropMetadata, RLEMask
from object_detection_training.schemas.detection import BoundingBox, Detection
from object_detection_training.schemas.label_mapping import LabelMapping

__all__ = [
    "BoundingBox",
    "CropMetadata",
    "Detection",
    "DetectionAnnotation",
    "LabelMapping",
    "RLEMask",
]
