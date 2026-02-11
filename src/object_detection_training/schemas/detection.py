"""Pydantic models for detection results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel, frozen=True):
    """Bounding box in top-left x/y, width, height format.

    All values are normalised to the [0, 1] range relative to the
    image dimensions.
    """

    x: float = Field(description="Top-left x (normalised 0-1)")
    y: float = Field(description="Top-left y (normalised 0-1)")
    w: float = Field(description="Width (normalised 0-1)")
    h: float = Field(description="Height (normalised 0-1)")


class Detection(BaseModel, frozen=True):
    """Single detection result."""

    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int
