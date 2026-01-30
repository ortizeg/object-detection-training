from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor
from torchvision.ops import box_iou


def compute_detection_curves(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[int | str, dict[str, Any]]:
    """
    Compute Precision-Recall and F1 curves for object detection.

    Args:
        preds: List of prediction dicts (boxes, scores, labels).
        targets: List of target dicts (boxes, labels).
        num_classes: Number of classes.
        iou_threshold: IOU threshold for matching.

    Returns:
        Dictionary containing curve data per class.
    """
    # Group by class
    class_preds: dict[int, list[dict[str, Any]]] = {c: [] for c in range(num_classes)}
    class_gts: dict[int, list[dict[str, Any]]] = {c: [] for c in range(num_classes)}

    # Organize data by class
    for i, (pred, target) in enumerate(zip(preds, targets, strict=True)):
        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
        p_labels = pred["labels"]

        t_boxes = target["boxes"]
        t_labels = target["labels"]

        for c in range(num_classes):
            # Filter preds for class c
            mask = p_labels == c
            if mask.any():
                class_preds[c].append(
                    {"boxes": p_boxes[mask], "scores": p_scores[mask], "image_id": i}
                )

            # Filter targets for class c
            mask = t_labels == c
            if mask.any():
                class_gts[c].append({"boxes": t_boxes[mask], "image_id": i})

    curves: dict[int | str, dict[str, Any]] = {}

    for c in range(num_classes):
        c_preds = class_preds[c]
        c_gts = class_gts[c]

        # Count total ground truths
        n_pos = sum(len(g["boxes"]) for g in c_gts)
        if n_pos == 0:
            continue

        # Organize GTs by image_id for faster matching
        gts_by_image = {g["image_id"]: g["boxes"] for g in c_gts}
        # Keep track of matched GTs to avoid double counting
        seen_gts: dict[Any, set[Any]] = {g["image_id"]: set() for g in c_gts}

        # Use 'scores' effectively as confidence thresholds
        # We need to sort by score descending for PR curve calculation?
        # Torchmetrics PR curve expects predictions and targets.
        # Standard AP matches globally sorted preds to GTs in their respective images.

        # First gather all preds into a single list with metadata
        flat_preds = []
        for p in c_preds:
            img_id = p["image_id"]
            for box, score in zip(p["boxes"], p["scores"], strict=True):
                flat_preds.append(
                    {"box": box, "score": score.item(), "image_id": img_id}
                )

        # Sort by score desc
        flat_preds.sort(key=lambda x: x["score"], reverse=True)

        # Match
        tps_list: list[int] = []
        fps_list: list[int] = []
        scores_list: list[float] = []

        for p in flat_preds:
            img_id = p["image_id"]
            p_box = p["box"].unsqueeze(0)  # [1, 4]
            score = p["score"]
            scores_list.append(score)

            if img_id in gts_by_image:
                gt_boxes = gts_by_image[img_id]  # [N, 4]
                if len(gt_boxes) > 0:
                    ious = box_iou(p_box, gt_boxes).squeeze(0)  # [N]
                    max_iou, max_idx = ious.max(0)
                    max_idx = max_idx.item()

                    if max_iou >= iou_threshold:
                        if max_idx not in seen_gts[img_id]:
                            tps_list.append(1)
                            fps_list.append(0)
                            seen_gts[img_id].add(max_idx)
                        else:
                            # Duplicate detection
                            tps_list.append(0)
                            fps_list.append(1)
                    else:
                        tps_list.append(0)
                        fps_list.append(1)
                else:
                    tps_list.append(0)
                    fps_list.append(1)
            else:
                tps_list.append(0)
                fps_list.append(1)

        # Compute cumulative stats
        tps = np.array(tps_list)
        fps = np.array(fps_list)
        scores = np.array(scores_list)

        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)

        recalls = tp_cumsum / n_pos
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        curves[c] = {
            "precision": precisions,
            "recall": recalls,
            "scores": scores,
            "f1": 2 * (precisions * recalls) / (precisions + recalls + 1e-6),
        }

    # Compute Overall PR Curve (Micro-averaged across all classes)
    all_flat_preds = []
    total_n_pos = 0

    # Collect all predictions and ground truths from all classes
    for c in range(num_classes):
        # We need to gather all preds for this class
        c_preds = class_preds[c]
        c_gts = class_gts[c]
        total_n_pos += sum(len(g["boxes"]) for g in c_gts)

        for p in c_preds:
            img_id = p["image_id"]
            for box, score in zip(p["boxes"], p["scores"], strict=True):
                all_flat_preds.append(
                    {
                        "box": box,
                        "score": score.item(),
                        "image_id": img_id,
                        "class_id": c,
                    }
                )

    if all_flat_preds and total_n_pos > 0:
        # Sort all preds globally by score
        all_flat_preds.sort(key=lambda x: x["score"], reverse=True)

        # Match (class-aware but global sorted)
        # Note: In object detection, we usually aggregate TP/FP counts globally
        # or macro-average. Here we do micro-average (treating every instance as
        # part of one pool).
        all_tps_list: list[int] = []
        all_fps_list: list[int] = []
        all_scores_list: list[float] = []

        # Reset matched GTs globally
        global_seen_gts: dict[int, dict[Any, set[Any]]] = {
            c: {g["image_id"]: set() for g in class_gts[c]} for c in range(num_classes)
        }
        gts_by_class_image = {
            c: {g["image_id"]: g["boxes"] for g in class_gts[c]}
            for c in range(num_classes)
        }

        for p in all_flat_preds:
            c = p["class_id"]
            img_id = p["image_id"]
            p_box = p["box"].unsqueeze(0)
            score = p["score"]
            all_scores_list.append(score)

            matched = False
            if c in gts_by_class_image and img_id in gts_by_class_image[c]:
                gt_boxes = gts_by_class_image[c][img_id]
                if len(gt_boxes) > 0:
                    ious = box_iou(p_box, gt_boxes).squeeze(0)
                    max_iou, max_idx = ious.max(0)
                    max_idx = max_idx.item()

                    if (
                        max_iou >= iou_threshold
                        and max_idx not in global_seen_gts[c][img_id]
                    ):
                        all_tps_list.append(1)
                        all_fps_list.append(0)
                        global_seen_gts[c][img_id].add(max_idx)
                        matched = True

            if not matched:
                all_tps_list.append(0)
                all_fps_list.append(1)

        tps = np.array(all_tps_list)
        fps = np.array(all_fps_list)
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)

        recalls = tp_cumsum / total_n_pos
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        curves["overall"] = {
            "precision": precisions,
            "recall": recalls,
            "scores": np.array(all_scores_list),
            "f1": 2 * (precisions * recalls) / (precisions + recalls + 1e-6),
        }

    return curves
