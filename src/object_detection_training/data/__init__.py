"""Data modules for object detection training."""

from __future__ import annotations

from object_detection_training.data.base import BaseDataModule
from object_detection_training.data.coco_data_module import COCODataModule
from object_detection_training.data.coco_detection_dataset import (
    COCODetectionDataset,
    collate_fn,
    collate_fn_with_image_ids,
)
from object_detection_training.data.dataset_stats import DatasetStatistics
from object_detection_training.data.detection_dataset import (
    DetectionDataset,
    SizeThresholds,
)

__all__ = [
    "BaseDataModule",
    "COCODataModule",
    "DetectionDataset",
    "COCODetectionDataset",
    "DatasetStatistics",
    "SizeThresholds",
    "collate_fn",
    "collate_fn_with_image_ids",
]
