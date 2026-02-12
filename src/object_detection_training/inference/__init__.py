"""Inference package for ONNX model inference and annotation."""

from __future__ import annotations

from object_detection_training.inference.inferencer import ONNXInferencer
from object_detection_training.inference.postprocess import (
    BasePostProcessor,
    RFDETRPostProcessor,
    YOLOXPostProcessor,
)
from object_detection_training.io.annotation import (
    DetectionAnnotationWriter,
)
from object_detection_training.io.image import ImageLoader
from object_detection_training.schemas.annotation import DetectionAnnotation
from object_detection_training.schemas.detection import (
    BoundingBox,
    Detection,
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
