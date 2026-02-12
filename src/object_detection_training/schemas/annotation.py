"""Pydantic models for detection annotations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from object_detection_training.schemas.detection import Detection


def _get_current_time() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


class AnnotationInfo(BaseModel):
    """Metadata for the annotation file."""

    annotations_source: str = Field(
        description="Source of the annotations (e.g. model name, 'manual')"
    )
    image_width: int | None = Field(default=None, description="Image width in pixels")
    image_height: int | None = Field(default=None, description="Image height in pixels")
    created_at: str = Field(
        default_factory=_get_current_time,
        description="ISO 8601 timestamp of creation",
    )


class DetectionAnnotation(BaseModel):
    """Per-image detection annotation in a normalised COCO-like format.

    Each image produces one ``DetectionAnnotation`` that is self-contained:
    it includes the label map so the file is interpretable in isolation.
    """

    filename: str = Field(description="Original image filename")
    categories: dict[int, str] = Field(
        description="Mapping from class index to label name"
    )
    info: AnnotationInfo = Field(description="Metadata about the annotation")
    annotations: list[Detection] = Field(default_factory=list)
