"""Pydantic models for detection annotations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from object_detection_training.schemas.detection import Detection


class DetectionAnnotation(BaseModel):
    """Per-image detection annotation in a normalised COCO-like format.

    Each image produces one ``DetectionAnnotation`` that is self-contained:
    it includes the label map so the file is interpretable in isolation.
    """

    filename: str = Field(description="Original image filename")
    categories: dict[int, str] = Field(
        description="Mapping from class index to label name"
    )
    annotations: list[Detection] = Field(default_factory=list)
