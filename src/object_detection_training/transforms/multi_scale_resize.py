"""Multi-scale resize transforms for object detection.

Provides square and aspect-ratio-preserving resize transforms that sample
from the RF-DETR multi-scale resolution list.
"""

from __future__ import annotations

import random

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


def compute_multi_scale_scales(
    resolution: int,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> list[int]:
    """Compute valid multi-scale resolutions for transformer patch tokens.

    Returns a list of image sizes that are divisible by ``patch_size * num_windows``
    so that both patching and windowing work correctly.

    Args:
        resolution: Base image resolution.
        expanded_scales: Use a wider range of scale offsets.
        patch_size: Transformer patch size.
        num_windows: Number of windows for windowed attention.

    Returns:
        Sorted list of valid square resolutions.
    """
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = (
        [-3, -2, -1, 0, 1, 2, 3, 4]
        if not expanded_scales
        else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    )
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    # Ensure minimum image size
    return [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]


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

    def make_params(self, flat_inputs: list[torch.Tensor]) -> dict[str, list[int]]:
        scale = random.choice(self.scales)  # noqa: S311
        return {"size": [scale, scale]}

    def transform(
        self, inpt: torch.Tensor, params: dict[str, list[int]]
    ) -> torch.Tensor:
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

    def make_params(self, flat_inputs: list[torch.Tensor]) -> dict[str, int]:
        size = random.choice(self.scales)  # noqa: S311
        return {"size": size}

    def transform(self, inpt: torch.Tensor, params: dict[str, int]) -> torch.Tensor:
        return self._call_kernel(
            F.resize, inpt, size=[params["size"]], max_size=self.max_size
        )
