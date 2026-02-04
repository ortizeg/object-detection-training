"""Data conversion transforms for bounding boxes and images."""

from __future__ import annotations

from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


class NormalizeBoxCoords(v2.Transform):
    """Normalize bounding box coordinates to ``[0, 1]`` and unwrap to plain tensor.

    Divides ``tv_tensors.BoundingBoxes`` by their canvas size and returns a
    plain ``torch.Tensor``.  After normalization the values are no longer valid
    pixel coordinates, so returning a plain tensor prevents accidental
    geometric transform application downstream.

    Only acts on ``tv_tensors.BoundingBoxes`` — all other inputs pass through
    unchanged.
    """

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt

        h, w = inpt.canvas_size
        # Both XYXY and CXCYWH use the same [w, h, w, h] divisor
        scale = torch.tensor([w, h, w, h], dtype=inpt.dtype, device=inpt.device)
        # Return plain Tensor (not BoundingBoxes) to prevent accidental
        # geometric transforms on normalized coordinates
        return torch.Tensor(inpt / scale)


class ToFloat32Tensor(v2.Transform):
    """Convert PIL images to float32 tensors.

    Wraps ``v2.ToImage`` + ``v2.ToDtype`` into a single transform that can be
    expressed in Hydra YAML (``torch.float32`` is not a valid YAML value).

    Args:
        scale: If ``True``, scale pixel values from ``[0, 255]`` to
            ``[0.0, 1.0]``.  If ``False`` (default), keep 0-255 range
            as float32.
    """

    def __init__(self, scale: bool = False) -> None:
        super().__init__()
        self._to_image = v2.ToImage()
        self._to_dtype = v2.ToDtype(torch.float32, scale=scale)

    def forward(self, *inputs: Any) -> Any:
        outputs = self._to_image(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return self._to_dtype(*outputs)
