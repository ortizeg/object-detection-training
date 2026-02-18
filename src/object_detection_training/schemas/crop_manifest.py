"""Pydantic schemas and utilities for masked object crop storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class RLEMask(BaseModel, frozen=True):
    """Run-length encoded binary mask (COCO convention).

    The mask is stored as a flat list of run lengths in row-major order.
    Runs alternate between 0 and 1 values, starting with 0.
    """

    counts: list[int] = Field(..., description="Run-length encoded counts")
    height: int = Field(..., gt=0, description="Mask height in pixels")
    width: int = Field(..., gt=0, description="Mask width in pixels")


class CropMetadata(BaseModel, frozen=True):
    """Metadata for a single extracted object crop.

    Each crop is stored as a `{annotation_id}.png` + `{annotation_id}.json` pair.
    """

    filename: str = Field(
        ..., description="Crop image filename (relative to category dir)"
    )
    mask: RLEMask = Field(..., description="RLE-encoded binary mask")
    bbox_x: float = Field(..., description="Original bbox x in source image")
    bbox_y: float = Field(..., description="Original bbox y in source image")
    bbox_w: float = Field(..., description="Original bbox width in source image")
    bbox_h: float = Field(..., description="Original bbox height in source image")
    source_image: str = Field(..., description="Source image filename")
    annotation_id: int = Field(..., description="Original annotation ID")
    category_name: str = Field(..., description="Category name")

    @classmethod
    def from_json(cls, path: Path | str) -> CropMetadata:
        """Load crop metadata from a JSON file."""
        with open(path) as f:
            return cls.model_validate_json(f.read())

    def save_json(self, path: Path | str) -> None:
        """Save crop metadata to a JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))


def encode_rle(mask: np.ndarray[Any, Any]) -> RLEMask:
    """Encode a binary mask as run-length encoding.

    Args:
        mask: 2D boolean/uint8 numpy array (H, W). Non-zero values are foreground.

    Returns:
        RLEMask with counts starting from background (0) runs.
    """
    h, w = mask.shape[:2]
    flat = mask.flatten().astype(bool)

    if len(flat) == 0:
        return RLEMask(counts=[], height=h, width=w)

    # Find positions where value changes
    changes = np.diff(flat.astype(np.int8))
    change_indices = np.where(changes != 0)[0] + 1

    # Build run lengths
    boundaries = np.concatenate(([0], change_indices, [len(flat)]))
    counts = np.diff(boundaries).tolist()

    # If first pixel is foreground (1), prepend a zero-length background run
    if flat[0]:
        counts = [0, *counts]

    return RLEMask(counts=counts, height=h, width=w)


def decode_rle(rle: RLEMask) -> np.ndarray[Any, Any]:
    """Decode a run-length encoded mask to a binary numpy array.

    Args:
        rle: RLEMask instance.

    Returns:
        2D boolean numpy array of shape (height, width).
    """
    if not rle.counts:
        return np.zeros((rle.height, rle.width), dtype=bool)

    # Alternate between 0 and 1 starting with 0
    flat = np.zeros(rle.height * rle.width, dtype=bool)
    pos = 0
    for i, count in enumerate(rle.counts):
        if i % 2 == 1:  # Odd indices are foreground
            flat[pos : pos + count] = True
        pos += count

    return flat.reshape(rle.height, rle.width)
