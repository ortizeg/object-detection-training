"""Object detection models."""

from __future__ import annotations

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.models.dinox_lightning import (
    DINOXLightningModel,
    DINOXMBaselineModel,
    DINOXMDFLModel,
    DINOXSBaselineModel,
)
from object_detection_training.models.rfdetr_lightning import (
    RFDETRLargeModel,
    RFDETRLightningModel,
    RFDETRMediumModel,
    RFDETRNanoModel,
    RFDETRSmallModel,
)
from object_detection_training.models.yolox_lightning import (
    YOLOXLightningModel,
    YOLOXLModel,
    YOLOXMModel,
    YOLOXNanoModel,
    YOLOXSModel,
    YOLOXTinyModel,
    YOLOXXModel,
)

__all__ = [
    "BaseDetectionModel",
    "DINOXLightningModel",
    "DINOXMBaselineModel",
    "DINOXMDFLModel",
    "DINOXSBaselineModel",
    "RFDETRLargeModel",
    "RFDETRLightningModel",
    "RFDETRMediumModel",
    "RFDETRNanoModel",
    "RFDETRSmallModel",
    "YOLOXLModel",
    "YOLOXLightningModel",
    "YOLOXMModel",
    "YOLOXNanoModel",
    "YOLOXSModel",
    "YOLOXTinyModel",
    "YOLOXXModel",
]
