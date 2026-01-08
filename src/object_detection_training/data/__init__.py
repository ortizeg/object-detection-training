"""Data modules for object detection training."""

from object_detection_training.data.base import BaseDataModule
from object_detection_training.data.coco import COCODataModule

__all__ = ["BaseDataModule", "COCODataModule"]
