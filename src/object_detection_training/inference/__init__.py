"""Inference package for ONNX model inference and annotation."""

from __future__ import annotations

from object_detection_training.inference.base_inferencer import BaseInferencer
from object_detection_training.inference.gemini_inferencer import GeminiInferencer
from object_detection_training.inference.onnx_inferencer import ONNXInferencer
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
    "BaseInferencer",
    "BasePostProcessor",
    "BoundingBox",
    "Detection",
    "DetectionAnnotation",
    "DetectionAnnotationWriter",
    "GeminiInferencer",
    "ImageLoader",
    "ONNXInferencer",
    "RFDETRPostProcessor",
    "YOLOXPostProcessor",
]
