"""DETR-style random size crop transform."""

from __future__ import annotations

import random

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


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

    def make_params(self, flat_inputs: list[torch.Tensor]) -> dict[str, int]:
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

    def transform(self, inpt: torch.Tensor, params: dict[str, int]) -> torch.Tensor:
        return self._call_kernel(
            F.crop,
            inpt,
            top=params["top"],
            left=params["left"],
            height=params["height"],
            width=params["width"],
        )
