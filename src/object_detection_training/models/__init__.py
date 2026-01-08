"""Object detection models."""

from object_detection_training.models.base import BaseDetectionModel
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
    # Base
    "BaseDetectionModel",
    # RFDETR Lightning models
    "RFDETRLightningModel",
    "RFDETRNanoModel",
    "RFDETRSmallModel",
    "RFDETRMediumModel",
    "RFDETRLargeModel",
    # YOLOX Lightning models
    "YOLOXLightningModel",
    "YOLOXNanoModel",
    "YOLOXTinyModel",
    "YOLOXSModel",
    "YOLOXMModel",
    "YOLOXLModel",
    "YOLOXXModel",
]
