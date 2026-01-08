from typing import Any, Dict, List

import numpy as np
from torch import Tensor
from torchvision.ops import box_iou


def compute_detection_curves(
    preds: List[Dict[str, Tensor]],
    targets: List[Dict[str, Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
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
    class_preds = {c: [] for c in range(num_classes)}
    class_gts = {c: [] for c in range(num_classes)}

    # Organize data by class
    for i, (pred, target) in enumerate(zip(preds, targets)):
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

    curves = {}

    for c in range(num_classes):
        c_preds = class_preds[c]
        c_gts = class_gts[c]

        if not c_preds:
            continue

        # Count total ground truths
        n_pos = sum(len(g["boxes"]) for g in c_gts)
        if n_pos == 0:
            continue

        # Organize GTs by image_id for faster matching
        gts_by_image = {g["image_id"]: g["boxes"] for g in c_gts}
        # Keep track of matched GTs to avoid double counting
        seen_gts = {g["image_id"]: set() for g in c_gts}

        # Use 'scores' effectively as confidence thresholds
        # We need to sort by score descending for PR curve calculation?
        # Torchmetrics PR curve expects predictions and targets.
        # Standard AP matches globally sorted preds to GTs in their respective images.

        # First gather all preds into a single list with metadata
        flat_preds = []
        for p in c_preds:
            img_id = p["image_id"]
            for box, score in zip(p["boxes"], p["scores"]):
                flat_preds.append(
                    {"box": box, "score": score.item(), "image_id": img_id}
                )

        # Sort by score desc
        flat_preds.sort(key=lambda x: x["score"], reverse=True)

        # Match
        tps = []
        fps = []
        scores = []

        for p in flat_preds:
            img_id = p["image_id"]
            p_box = p["box"].unsqueeze(0)  # [1, 4]
            score = p["score"]
            scores.append(score)

            if img_id in gts_by_image:
                gt_boxes = gts_by_image[img_id]  # [N, 4]
                if len(gt_boxes) > 0:
                    ious = box_iou(p_box, gt_boxes).squeeze(0)  # [N]
                    max_iou, max_idx = ious.max(0)
                    max_idx = max_idx.item()

                    if max_iou >= iou_threshold:
                        if max_idx not in seen_gts[img_id]:
                            tps.append(1)
                            fps.append(0)
                            seen_gts[img_id].add(max_idx)
                        else:
                            # Duplicate detection
                            tps.append(0)
                            fps.append(1)
                    else:
                        tps.append(0)
                        fps.append(1)
                else:
                    tps.append(0)
                    fps.append(1)
            else:
                tps.append(0)
                fps.append(1)

        # Compute cumulative stats
        tps = np.array(tps)
        fps = np.array(fps)
        scores = np.array(scores)

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

    return curves
