"""
Utilities for bounding box conversions and manipulations.
"""

from __future__ import annotations

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


def box_iou_1_to_n(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one XYXY box and N XYXY boxes.

    Args:
        box: Tensor of shape (1, 4) or (4,) in XYXY format.
        boxes: Tensor of shape (N, 4) in XYXY format.

    Returns:
        Tensor of shape (N,) with IoU values.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=box.dtype, device=box.device)

    box = box.view(1, 4)

    # Intersection
    inter_x1 = torch.max(box[:, 0], boxes[:, 0])
    inter_y1 = torch.max(box[:, 1], boxes[:, 1])
    inter_x2 = torch.min(box[:, 2], boxes[:, 2])
    inter_y2 = torch.min(box[:, 3], boxes[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - inter_area
    return inter_area / union.clamp(min=1e-6)


def pad_and_clamp_bbox(
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    img_w: int,
    img_h: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Pad a bounding box by a ratio and clamp to image bounds.

    Expands the box by ``padding_ratio * dimension`` on each side,
    then clamps to ``[0, img_w) x [0, img_h)``.

    Args:
        bbox_x: Top-left x of the original bbox (COCO format).
        bbox_y: Top-left y of the original bbox (COCO format).
        bbox_w: Width of the original bbox.
        bbox_h: Height of the original bbox.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        padding_ratio: Fraction of bbox dimension to add as padding.

    Returns:
        Tuple of ``(x1, y1, x2, y2)`` as clamped integer pixel coordinates.
    """
    pad_w = bbox_w * padding_ratio
    pad_h = bbox_h * padding_ratio

    x1 = max(0, int(bbox_x - pad_w))
    y1 = max(0, int(bbox_y - pad_h))
    x2 = min(img_w, int(bbox_x + bbox_w + pad_w))
    y2 = min(img_h, int(bbox_y + bbox_h + pad_h))

    return x1, y1, x2, y2


def pixel_xyxy_to_normalized_xywh(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy coordinates to normalised xywh (top-left origin).

    Args:
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Tuple of (x, y, w, h) normalised to [0, 1].
    """
    x = x1 / image_width
    y = y1 / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return x, y, w, h


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
