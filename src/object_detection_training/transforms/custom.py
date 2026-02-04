"""Custom torchvision v2 transforms for object detection.

Each class subclasses ``torchvision.transforms.v2.Transform`` and uses the
kernel-dispatch API (``transform``) so that ``tv_tensors.BoundingBoxes``,
``tv_tensors.Image``, and plain tensors are handled automatically.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from object_detection_training.models.rfdetr.coco import compute_multi_scale_scales


class MultiScaleResize(v2.Transform):
    """Square resize using the multi-scale scale list from RF-DETR.

    Computes a list of valid square sizes via
    :func:`compute_multi_scale_scales` and randomly picks one each call.
    The image is resized to ``(scale, scale)`` — an exact square — and
    bounding boxes are transformed automatically by v2 dispatch.
    """

    def __init__(
        self,
        base_resolution: int = 560,
        expanded_scales: bool = True,
        patch_size: int = 16,
        num_windows: int = 2,
        skip_random_resize: bool = False,
    ) -> None:
        super().__init__()
        scales = compute_multi_scale_scales(
            base_resolution, expanded_scales, patch_size, num_windows
        )
        if skip_random_resize:
            scales = [scales[-1]]
        self.scales = scales

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        scale = random.choice(self.scales)  # noqa: S311
        return {"size": [scale, scale]}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(F.resize, inpt, size=params["size"])


class MultiScaleRandomResize(v2.Transform):
    """Aspect-ratio-preserving resize from multi-scale list.

    Picks a random size from the computed scale list and resizes the short
    edge to that value (capping the long edge at ``max_size``).  This is
    the v2 equivalent of the legacy ``RandomResize`` transform.
    """

    def __init__(
        self,
        base_resolution: int = 560,
        expanded_scales: bool = True,
        patch_size: int = 16,
        num_windows: int = 2,
        skip_random_resize: bool = False,
        max_size: int = 1333,
    ) -> None:
        super().__init__()
        scales = compute_multi_scale_scales(
            base_resolution, expanded_scales, patch_size, num_windows
        )
        if skip_random_resize:
            scales = [scales[-1]]
        self.scales = scales
        self.max_size = max_size

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        size = random.choice(self.scales)  # noqa: S311
        return {"size": size}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(
            F.resize, inpt, size=[params["size"]], max_size=self.max_size
        )


class RandomSizeCrop(v2.Transform):
    """DETR-style random crop with random dimensions in ``[min_size, max_size]``.

    The crop width and height are independently sampled from
    ``[min_size, min(img_dim, max_size)]``.  The crop region is then randomly
    placed within the image.  Bounding boxes are automatically clipped by
    v2 dispatch.
    """

    def __init__(self, min_size: int = 384, max_size: int = 600) -> None:
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        # Get image spatial size from first spatial input
        orig_h, orig_w = F.get_size(flat_inputs[0])

        crop_w = random.randint(  # noqa: S311
            self.min_size, min(orig_w, self.max_size)
        )
        crop_h = random.randint(  # noqa: S311
            self.min_size, min(orig_h, self.max_size)
        )

        top = random.randint(0, max(0, orig_h - crop_h))  # noqa: S311
        left = random.randint(0, max(0, orig_w - crop_w))  # noqa: S311

        return {
            "top": top,
            "left": left,
            "height": crop_h,
            "width": crop_w,
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(
            F.crop,
            inpt,
            top=params["top"],
            left=params["left"],
            height=params["height"],
            width=params["width"],
        )


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
