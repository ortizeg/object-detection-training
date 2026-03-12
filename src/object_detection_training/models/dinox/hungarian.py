"""Hungarian (one-to-one) label assignment for dual-head training.

Uses scipy's ``linear_sum_assignment`` to find optimal 1:1 matching between
predictions and ground truths. Returns the same 5-tuple contract as
``DINOXHead._get_assignments`` and ``TaskAlignedAssigner.assign`` for drop-in use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _bboxes_iou(
    bboxes_a: torch.Tensor,
    bboxes_b: torch.Tensor,
    xyxy: bool = True,
) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Duplicated from dinox_head/tal to avoid circular import.

    Always runs in float32 to avoid bf16 overflow on area products
    (e.g. 640*640 = 409600 exceeds bf16 max of 65504).
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError("Boxes must have 4 columns")

    bboxes_a = bboxes_a.float()
    bboxes_b = bboxes_b.float()

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
    return area_i / (area_a[:, None] + area_b - area_i + 1e-8)


def _get_in_boxes_info(
    gt_bboxes_per_image: torch.Tensor,
    expanded_strides: torch.Tensor,
    x_shifts: torch.Tensor,
    y_shifts: torch.Tensor,
    total_num_anchors: int,
    num_gt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Determine which anchors are in GT boxes or center regions.

    Duplicated from DINOXHead._get_in_boxes_info so Hungarian is self-contained.

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


class HungarianAssigner:
    """One-to-one label assignment using Hungarian algorithm.

    Computes an alignment metric ``m = cls_score^alpha * IoU^beta`` and uses
    ``scipy.optimize.linear_sum_assignment`` to find optimal 1:1 matching.
    Returns exactly ``num_gt`` foreground anchors (one per GT).

    The ``assign`` method has the same signature and 5-tuple return contract
    as ``DINOXHead._get_assignments`` and ``TaskAlignedAssigner.assign``.

    Args:
        alpha: Exponent for classification score in alignment metric.
        beta: Exponent for IoU in alignment metric.
        num_classes: Number of object classes.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 6.0,
        num_classes: int = 80,
    ) -> None:
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
        """Hungarian (one-to-one) label assignment.

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

        # Early return for zero GT
        if num_gt == 0:
            return (
                torch.zeros(0, device=device),
                torch.zeros(total_num_anchors, dtype=torch.bool, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
                0,
            )

        # Spatial filtering (same as SimOTA/TAL)
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
        cls_sigmoid = cls_preds_cand.sigmoid()
        gt_cls_one_hot = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        # bbox_scores[g, a] = cls_sigmoid[a, gt_class[g]]
        bbox_scores = (cls_sigmoid.unsqueeze(1) * gt_cls_one_hot.unsqueeze(0)).sum(
            -1
        )  # [num_candidates, num_gt]
        bbox_scores = bbox_scores.T  # [num_gt, num_candidates]

        # Alignment metric: m = cls_score^alpha * IoU^beta
        # Cast to float32 — pow(beta=6.0) underflows in bf16 for small IoU values
        align_metric = bbox_scores.float().pow(self.alpha) * pair_wise_ious.float().pow(
            self.beta
        )

        # Cost matrix: negative alignment + large penalty for out-of-box
        # Use 1e4 instead of 1e6 — bf16 max is ~65504, so 1e6 overflows to inf
        cost = -align_metric + 1e4 * (~is_in_boxes_and_center).float()

        # Solve with Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # Filter out matches with infinite cost (no valid candidate for a GT)
        valid = cost[row_ind, col_ind] < 1e3
        row_ind_t = torch.tensor(row_ind, device=device, dtype=torch.long)
        col_ind_t = torch.tensor(col_ind, device=device, dtype=torch.long)
        valid_t = torch.tensor(valid, device=device, dtype=torch.bool)
        row_ind_t = row_ind_t[valid_t]
        col_ind_t = col_ind_t[valid_t]

        num_fg = int(row_ind_t.shape[0])

        if num_fg == 0:
            return (
                torch.zeros(0, device=device),
                torch.zeros(total_num_anchors, dtype=torch.bool, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
                0,
            )

        # Build fg_mask: mark only matched candidate positions
        fg_mask_new = torch.zeros(total_num_anchors, dtype=torch.bool, device=device)
        fg_idxs = torch.nonzero(fg_mask, as_tuple=True)[0]
        matched_anchor_idxs = fg_idxs[col_ind_t]
        fg_mask_new[matched_anchor_idxs] = True

        # Extract matched GT info
        matched_gt_inds = row_ind_t
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = pair_wise_ious[row_ind_t, col_ind_t]

        return (
            gt_matched_classes,
            fg_mask_new,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
