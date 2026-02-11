"""Inference package for ONNX model inference and annotation."""

from __future__ import annotations

from object_detection_training.inference.annotation import (
    DetectionAnnotationWriter,
)
from object_detection_training.inference.image import ImageLoader
from object_detection_training.inference.inferencer import ONNXInferencer
from object_detection_training.inference.models import (
    BoundingBox,
    Detection,
    DetectionAnnotation,
)
from object_detection_training.inference.postprocess import (
    BasePostProcessor,
    RFDETRPostProcessor,
    YOLOXPostProcessor,
)

__all__ = [
    "BasePostProcessor",
    "BoundingBox",
    "Detection",
    "DetectionAnnotation",
    "DetectionAnnotationWriter",
    "ImageLoader",
    "ONNXInferencer",
    "RFDETRPostProcessor",
    "YOLOXPostProcessor",
]
