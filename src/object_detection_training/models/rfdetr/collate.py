from __future__ import annotations

# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Collate utilities for detection training.

Contains NestedTensor and collate_fn for handling variable-size images
from multi-scale training.

Originally from rfdetr.util.misc, copied here for clarity and stability.
"""

import torch
import torchvision
from torch import Tensor


class NestedTensor:
    """Tensor with associated padding mask for variable-size batches."""

    def __init__(self, tensors: Tensor, mask: Tensor | None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        cast_mask = mask.to(device) if mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list: list[list[int]]) -> list[int]:
    """Get maximum value at each position across sublists."""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: list[Tensor]) -> NestedTensor:
    """Create a NestedTensor from a list of tensors with different sizes.

    Pads all tensors to the maximum size and creates a mask indicating
    valid (non-padded) regions.
    """
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # Get maximum size across all tensors
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        # Create padded tensor and mask
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for img, pad_img, m in zip(tensor_list, tensor, mask, strict=True):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3D tensors (C, H, W) are supported")

    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: list[Tensor]) -> NestedTensor:
    """ONNX-compatible version of nested_tensor_from_tensor_list."""
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape), strict=True)]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def collate_fn(batch):
    """Collate function for detection training with variable-size images.

    Handles multi-scale augmentation by padding images to the maximum size
    in the batch and creating attention masks for padded regions.

    Args:
        batch: List of (image, target) tuples where images may have
               different sizes due to multi-scale augmentation.

    Returns:
        Tuple of (NestedTensor, list of targets).
    """
    batch = list(zip(*batch, strict=True))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
