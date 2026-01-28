"""
Utilities for bounding box conversions and manipulations.
"""

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2].

    Args:
        boxes: Tensor of shape [..., 4] in cxcywh format.

    Returns:
        Tensor of shape [..., 4] in xyxy format.
    """
    if boxes.numel() == 0:
        return boxes

    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [cx, cy, w, h].

    Args:
        boxes: Tensor of shape [..., 4] in xyxy format.

    Returns:
        Tensor of shape [..., 4] in cxcywh format.
    """
    if boxes.numel() == 0:
        return boxes

    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)
