"""RTMDet-style RandomResize + RandomCrop transform.

Replaces YOLOX RandomAffine with a simpler resize-then-crop strategy that
RTMDet showed produces equal or better results with less computation.
"""

from __future__ import annotations

import random

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class RandomResizeCrop(v2.Transform):
    """Random ratio resize followed by random crop.

    Matches RTMDet's post-mosaic pipeline: ``RandomResize(ratio_range) →
    RandomCrop(crop_size)``.  The image is resized by a random ratio
    (maintaining aspect ratio), padded if smaller than the crop size, then
    randomly cropped to a fixed output size.

    This is simpler and faster than YOLOX's ``RandomAffine`` (which adds
    rotation, shear, and translation) while producing equivalent results
    (RTMDet, Table 7).
    """

    def __init__(
        self,
        crop_size: tuple[int, int] = (640, 640),
        ratio_range: tuple[float, float] = (0.1, 2.0),
        pad_val: int = 114,
    ) -> None:
        """Initialize RandomResizeCrop.

        Args:
            crop_size: Output (height, width) after cropping.
            ratio_range: Min and max scale ratios for the random resize.
                RTMDet uses (0.1, 2.0) for post-mosaic augmentation.
            pad_val: Pixel value for padding when resized image is smaller
                than crop_size.  Default 114 matches YOLOX/RTMDet gray fill.
        """
        super().__init__()
        self.crop_h, self.crop_w = crop_size
        self.ratio_min, self.ratio_max = ratio_range
        self.pad_val = pad_val

    def make_params(self, flat_inputs: list[torch.Tensor]) -> dict[str, float | int]:
        orig_h, orig_w = F.get_size(flat_inputs[0])

        # Sample a random scale ratio
        ratio = random.uniform(self.ratio_min, self.ratio_max)  # noqa: S311

        # Compute new size maintaining aspect ratio
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)

        # Compute padding needed if resized image is smaller than crop
        pad_h = max(self.crop_h - new_h, 0)
        pad_w = max(self.crop_w - new_w, 0)

        # Total size after resize + padding
        total_h = new_h + pad_h
        total_w = new_w + pad_w

        # Random crop position
        top = random.randint(0, max(0, total_h - self.crop_h))  # noqa: S311
        left = random.randint(0, max(0, total_w - self.crop_w))  # noqa: S311

        return {
            "new_h": new_h,
            "new_w": new_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "top": top,
            "left": left,
        }

    def transform(
        self, inpt: torch.Tensor, params: dict[str, float | int]
    ) -> torch.Tensor:
        new_h = int(params["new_h"])
        new_w = int(params["new_w"])
        pad_h = int(params["pad_h"])
        pad_w = int(params["pad_w"])
        top = int(params["top"])
        left = int(params["left"])

        # Resize
        out = self._call_kernel(F.resize, inpt, size=[new_h, new_w])

        # Pad if needed (right and bottom padding)
        if pad_h > 0 or pad_w > 0:
            out = self._call_kernel(
                F.pad, out, padding=[0, 0, pad_w, pad_h], fill=self.pad_val
            )

        # Crop
        out = self._call_kernel(
            F.crop, out, top=top, left=left, height=self.crop_h, width=self.crop_w
        )
        return out  # type: ignore[return-value]
