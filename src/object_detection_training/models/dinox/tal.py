"""Task-Aligned Label Assignment (TAL) from TOOD (ICCV 2021).

TAL replaces SimOTA's dynamic-k optimal transport with a simpler top-k
selection based on an alignment metric ``m = cls_score^alpha * IoU^beta``.
It returns the same 5-tuple as SimOTA ``_get_assignments`` for drop-in use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _bboxes_iou(
    bboxes_a: torch.Tensor,
    bboxes_b: torch.Tensor,
    xyxy: bool = True,
) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Duplicated from dinox_head to avoid circular import (dinox_head imports TAL).
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError("Boxes must have 4 columns")

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).to(tl.dtype).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


def _get_in_boxes_info(
    gt_bboxes_per_image: torch.Tensor,
    expanded_strides: torch.Tensor,
    x_shifts: torch.Tensor,
    y_shifts: torch.Tensor,
    total_num_anchors: int,
    num_gt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Determine which anchors are in GT boxes or center regions.

    Duplicated from DINOXHead._get_in_boxes_info so TAL is self-contained.

    Args:
        gt_bboxes_per_image: GT boxes in cxcywh format, shape ``[num_gt, 4]``.
        expanded_strides: Stride per anchor, shape ``[1, total_num_anchors]``.
        x_shifts: Grid x-coords, shape ``[1, total_num_anchors]``.
        y_shifts: Grid y-coords, shape ``[1, total_num_anchors]``.
        total_num_anchors: Total number of anchors.
        num_gt: Number of ground truth boxes.

    Returns:
        Tuple of (is_in_boxes_anchor, is_in_boxes_and_center):
        - is_in_boxes_anchor: bool mask ``[total_num_anchors]``
        - is_in_boxes_and_center: bool mask ``[num_gt, num_candidates]``
    """
    expanded_strides_per_image = expanded_strides[0]
    x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    x_centers = (
        (x_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
        .repeat(num_gt, 1)
    )
    y_centers = (
        (y_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
        .repeat(num_gt, 1)
    )

    # Check 1: inside GT boxes (cxcywh)
    gt_l = (
        (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_r = (
        (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_t = (
        (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_b = (
        (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )

    b_l = x_centers - gt_l
    b_r = gt_r - x_centers
    b_t = y_centers - gt_t
    b_b = gt_b - y_centers
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

    # Check 2: within center radius
    center_radius = 2.5
    gt_centers_l = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_centers_r = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_centers_t = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_centers_b = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)

    c_l = x_centers - gt_centers_l
    c_r = gt_centers_r - x_centers
    c_t = y_centers - gt_centers_t
    c_b = gt_centers_b - y_centers
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0

    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
    is_in_boxes_and_center = (
        is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
    )
    return is_in_boxes_anchor, is_in_boxes_and_center


class TaskAlignedAssigner:
    """Task-aligned label assignment (TAL-01) from TOOD (ICCV 2021).

    Uses alignment metric ``m = cls_score^alpha * IoU^beta`` with top-k
    selection instead of SimOTA's dynamic-k optimal transport.

    The ``assign`` method has the same signature and return contract as
    ``DINOXHead._get_assignments`` for drop-in replacement.

    Args:
        topk: Number of top candidates per GT by alignment metric.
        alpha: Exponent for classification score in alignment metric.
        beta: Exponent for IoU in alignment metric.
        num_classes: Number of object classes.
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        num_classes: int = 80,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    @torch.no_grad()
    def assign(
        self,
        batch_idx: int,
        num_gt: int,
        total_num_anchors: int,
        gt_bboxes_per_image: torch.Tensor,
        gt_classes: torch.Tensor,
        bbox_preds: torch.Tensor,
        cls_preds: torch.Tensor,
        obj_preds: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Task-aligned label assignment.

        Identical signature and return contract as ``DINOXHead._get_assignments``.

        Args:
            batch_idx: Index of current image in the batch (unused, kept for
                signature compatibility).
            num_gt: Number of ground truth objects.
            total_num_anchors: Total number of anchors across all FPN levels.
            gt_bboxes_per_image: GT boxes in cxcywh format, ``[num_gt, 4]``.
            gt_classes: GT class labels, ``[num_gt]``.
            bbox_preds: Decoded box predictions in cxcywh, ``[total_anchors, 4]``.
            cls_preds: Classification logits, ``[total_anchors, C]``.
            obj_preds: Objectness logits, ``[total_anchors, 1]``.
            expanded_strides: Stride per anchor, ``[1, total_anchors]``.
            x_shifts: Grid x-coordinates, ``[1, total_anchors]``.
            y_shifts: Grid y-coordinates, ``[1, total_anchors]``.

        Returns:
            5-tuple matching ``_get_assignments`` contract:
            - gt_matched_classes: ``[num_fg]`` class labels for matched GTs.
            - fg_mask: ``[total_anchors]`` bool mask of foreground anchors.
            - pred_ious_this_matching: ``[num_fg]`` IoU with matched GT.
            - matched_gt_inds: ``[num_fg]`` index of matched GT per fg anchor.
            - num_fg: number of foreground anchors (int).
        """
        device = bbox_preds.device
        gt_bboxes_per_image = gt_bboxes_per_image.to(device)
        gt_classes = gt_classes.to(device)

        # Spatial filtering (same as SimOTA)
        fg_mask, is_in_boxes_and_center = _get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        # Filter to candidate anchors
        bbox_preds_cand = bbox_preds[fg_mask]
        cls_preds_cand = cls_preds[fg_mask]
        num_candidates = bbox_preds_cand.shape[0]

        if num_candidates == 0:
            return (
                torch.zeros(0, device=device),
                torch.zeros(total_num_anchors, dtype=torch.bool, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
                0,
            )

        # Pairwise IoU: [num_gt, num_candidates]
        pair_wise_ious = _bboxes_iou(gt_bboxes_per_image, bbox_preds_cand, xyxy=False)

        # Classification score for each GT class at each candidate
        # cls_sigmoid: [num_candidates, C]
        cls_sigmoid = cls_preds_cand.sigmoid()
        # gt one-hot: [num_gt, C]
        gt_cls_one_hot = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        # bbox_scores[g, a] = cls_sigmoid[a, gt_class[g]]
        # Efficient: (num_candidates, 1, C) * (1, num_gt, C) -> sum over C
        bbox_scores = (cls_sigmoid.unsqueeze(1) * gt_cls_one_hot.unsqueeze(0)).sum(
            -1
        )  # [num_candidates, num_gt]
        bbox_scores = bbox_scores.T  # [num_gt, num_candidates]

        # Alignment metric: m = cls_score^alpha * IoU^beta
        align_metric = bbox_scores.pow(self.alpha) * pair_wise_ious.pow(self.beta)

        # Top-k selection per GT
        topk = min(self.topk, num_candidates)
        _, topk_idxs = align_metric.topk(topk, dim=1)

        # Build matching matrix (vectorized scatter replaces Python loop)
        matching_matrix = torch.zeros_like(align_metric, dtype=torch.uint8)
        matching_matrix.scatter_(1, topk_idxs, 1)

        # Apply spatial constraint
        matching_matrix *= is_in_boxes_and_center.to(matching_matrix.dtype)

        # Check if any matches remain after spatial constraint
        if matching_matrix.sum() == 0:
            return (
                torch.zeros(0, device=device),
                torch.zeros(total_num_anchors, dtype=torch.bool, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
                0,
            )

        # Resolve multi-GT conflicts: keep highest alignment metric
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, best_gt = align_metric[:, anchor_matching_gt > 1].max(dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[best_gt, anchor_matching_gt > 1] = 1

        # Extract results
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = int(fg_mask_inboxes.sum().item())

        # Update fg_mask to only include matched candidates
        fg_mask_new = fg_mask.clone()
        fg_idxs = torch.nonzero(fg_mask, as_tuple=True)[0]
        fg_mask_new[fg_idxs[~fg_mask_inboxes]] = False
        fg_mask = fg_mask_new

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
