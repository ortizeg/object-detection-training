"""IO utilities for object detection training."""

from object_detection_training.io.annotation import DetectionAnnotationWriter
from object_detection_training.io.image import ImageLoader

__all__ = ["DetectionAnnotationWriter", "ImageLoader"]
