"""Multi-method object detection evaluation task.

Compares Gemini VLM, SmolVLM2, and RF-DETR on the same dataset using
``supervision.MeanAveragePrecision`` and precision/recall/F1 metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from loguru import logger
from pydantic import Field
from tqdm import tqdm

from object_detection_training.inference.base_inferencer import BaseInferencer
from object_detection_training.io.image import ImageLoader
from object_detection_training.schemas.detection import Detection
from object_detection_training.tasks.base_task import BaseTask
from object_detection_training.utils.hydra import register

matplotlib.use("Agg")

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

# Basketball player sub-classes that get merged into "player"
_PLAYER_CLASSES = frozenset(
    {
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
    }
)

# Eval label map (after merging): class_id -> name
_EVAL_LABEL_MAP: dict[int, str] = {
    0: "player",
    1: "ball",
    2: "referee",
    3: "rim",
    4: "number",
}

# Reverse lookup for COCO category names -> eval class id
_NAME_TO_EVAL_ID: dict[str, int] = {}
for _id, _name in _EVAL_LABEL_MAP.items():
    _NAME_TO_EVAL_ID[_name] = _id
# All player sub-classes map to 0
for _pc in _PLAYER_CLASSES:
    _NAME_TO_EVAL_ID[_pc] = 0
# ball-in-basket -> ball
_NAME_TO_EVAL_ID["ball-in-basket"] = 1
# COCO-vocabulary aliases (for zero-shot detectors pre-trained on COCO/O365)
_NAME_TO_EVAL_ID["person"] = 0  # COCO "person" -> player
_NAME_TO_EVAL_ID["sports ball"] = 1  # COCO "sports ball" -> ball
_NAME_TO_EVAL_ID["basketball"] = 1  # alternate -> ball
_NAME_TO_EVAL_ID["basketball hoop"] = 3  # -> rim
_NAME_TO_EVAL_ID["hoop"] = 3
_NAME_TO_EVAL_ID["jersey number"] = 4


def _load_coco_gt(
    coco_json_path: Path,
) -> dict[str, sv.Detections]:
    """Parse a COCO annotations JSON and return per-image ground truth.

    Returns:
        Dict mapping image filename -> sv.Detections (xyxy pixel coords).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build category id -> name
    cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in coco["categories"]}

    # Build image id -> (filename, width, height)
    img_info: dict[int, tuple[str, int, int]] = {
        img["id"]: (img["file_name"], img["width"], img["height"])
        for img in coco["images"]
    }

    # Group annotations by image
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    result: dict[str, sv.Detections] = {}
    for img_id, (filename, _img_w, _img_h) in img_info.items():
        anns = anns_by_image.get(img_id, [])
        boxes: list[list[float]] = []
        class_ids: list[int] = []

        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            eval_id = _NAME_TO_EVAL_ID.get(cat_name.lower())
            if eval_id is None:
                continue

            # COCO bbox is [x, y, w, h] in pixels
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            class_ids.append(eval_id)

        if boxes:
            result[filename] = sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int),
            )
        else:
            result[filename] = sv.Detections.empty()

    return result


def _detections_to_sv(
    detections: list[Detection],
    image_width: int,
    image_height: int,
) -> sv.Detections:
    """Convert internal Detection list to supervision Detections."""
    if not detections:
        return sv.Detections.empty()

    boxes: list[list[float]] = []
    class_ids: list[int] = []
    confidences: list[float] = []

    for det in detections:
        # Convert from normalised xywh to pixel xyxy
        x1 = det.bbox.x * image_width
        y1 = det.bbox.y * image_height
        x2 = (det.bbox.x + det.bbox.w) * image_width
        y2 = (det.bbox.y + det.bbox.h) * image_height

        # Remap class id through eval label map
        eval_id = det.class_id  # already mapped by inferencer
        boxes.append([x1, y1, x2, y2])
        class_ids.append(eval_id)
        confidences.append(det.confidence)

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confidences, dtype=np.float32),
    )


def _compute_metrics(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
) -> dict[str, Any]:
    """Compute mAP and per-class AP using supervision."""
    map_metric = sv.metrics.MeanAveragePrecision()  # type: ignore[attr-defined]

    for filename in gt_map:
        gt = gt_map[filename]
        pred = pred_map.get(filename, sv.Detections.empty())
        map_metric.update(predictions=pred, targets=gt)

    result = map_metric.compute()

    return {
        "mAP_50_95": float(result.map50_95),
        "mAP_50": float(result.map50),
        "mAP_75": float(result.map75),
        "per_class_ap50": {
            _EVAL_LABEL_MAP.get(int(cls_id), str(cls_id)): float(
                result.ap_per_class[i][0]  # IoU=0.5 is index 0
            )
            for i, cls_id in enumerate(result.matched_classes)
        },
    }


def _compute_prf1_at_threshold(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
    threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision, recall, F1 at a confidence threshold."""
    tp = 0
    fp = 0
    total_gt = 0

    for filename in gt_map:
        gt = gt_map[filename]
        pred = pred_map.get(filename, sv.Detections.empty())

        total_gt += len(gt)

        if pred.confidence is not None:
            mask = pred.confidence >= threshold
            pred = pred[mask]  # type: ignore[assignment]

        if len(gt) == 0:
            fp += len(pred)
            continue

        if len(pred) == 0:
            continue

        # Compute IoU between predictions and ground truth
        from supervision.detection.utils.iou_and_nms import box_iou_batch

        iou_matrix = box_iou_batch(pred.xyxy, gt.xyxy)

        # Match predictions to ground truth (greedy)
        matched_gt: set[int] = set()
        for pred_idx in range(len(pred)):
            if iou_matrix.shape[1] == 0:
                fp += 1
                continue
            best_gt_idx = int(np.argmax(iou_matrix[pred_idx]))
            best_iou = float(iou_matrix[pred_idx, best_gt_idx])

            pred_cls = int(pred.class_id[pred_idx]) if pred.class_id is not None else -1
            gt_cls = int(gt.class_id[best_gt_idx]) if gt.class_id is not None else -2
            if (
                best_iou >= iou_threshold
                and best_gt_idx not in matched_gt
                and pred_cls == gt_cls
            ):
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def _find_best_threshold(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
    steps: int = 20,
) -> tuple[float, dict[str, float]]:
    """Sweep confidence thresholds on val to find max F1."""
    best_threshold = 0.0
    best_metrics: dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for i in range(1, steps + 1):
        threshold = i / steps
        metrics = _compute_prf1_at_threshold(gt_map, pred_map, threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def _plot_pr_curves(
    results: dict[str, dict[str, Any]],
    split: str,
    output_path: Path,
) -> None:
    """Plot overlaid PR curves for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for method_name, method_results in results.items():
        pr_data = method_results.get(f"{split}_pr_data")
        if pr_data is None:
            continue
        precisions = pr_data["precisions"]
        recalls = pr_data["recalls"]
        ax.plot(recalls, precisions, label=method_name, linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves ({split})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"PR curve saved to {output_path}")


def _compute_pr_curve(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
    steps: int = 20,
) -> dict[str, list[float]]:
    """Compute precision-recall curve data."""
    precisions: list[float] = []
    recalls: list[float] = []

    for i in range(steps + 1):
        threshold = i / steps
        metrics = _compute_prf1_at_threshold(gt_map, pred_map, threshold)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])

    return {"precisions": precisions, "recalls": recalls}


@register(group="task")
class EvalDetectionTask(BaseTask):
    """Evaluate multiple object detection methods on the same dataset.

    Runs Gemini, SmolVLM2, and/or RF-DETR on val/test splits and
    computes mAP, precision, recall, F1, and PR curves.
    """

    name: str = Field(default="eval_detection", description="Task name")

    # Data directories (COCO format with _annotations.coco.json)
    val_dir: Path = Field(
        description="Val directory with images + _annotations.coco.json"
    )
    test_dir: Path = Field(
        description="Test directory with images + _annotations.coco.json"
    )

    # Gemini config
    run_gemini: bool = Field(default=True, description="Run Gemini evaluation")
    gemini_model_name: str = Field(
        default="gemini-3.1-pro-preview",
        description="Gemini model name",
    )
    gemini_classes: list[str] = Field(
        default=["ball", "number", "player", "referee", "rim"],
        description="Classes for Gemini detection",
    )
    gemini_prompt_template: str | None = Field(
        default=None, description="Optional Gemini prompt template"
    )

    # SmolVLM2 config
    run_smolvlm2: bool = Field(default=True, description="Run SmolVLM2 evaluation")
    smolvlm2_model_name: str = Field(
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        description="SmolVLM2 model name",
    )

    # RF-DETR config
    run_rfdetr: bool = Field(default=True, description="Run RF-DETR evaluation")
    onnx_model_path: Path | None = Field(
        default=None, description="Path to RF-DETR ONNX model"
    )
    label_mapping_path: Path | None = Field(
        default=None, description="Path to labels_mapping.json"
    )
    rfdetr_input_size: int = Field(default=560, description="RF-DETR input size")
    rfdetr_confidence_threshold: float = Field(
        default=0.01, description="RF-DETR confidence threshold for raw predictions"
    )

    # YOLOX config
    run_yolox: bool = Field(default=False, description="Run YOLOX evaluation")
    yolox_onnx_model_path: Path | None = Field(
        default=None, description="Path to YOLOX ONNX model"
    )
    yolox_label_mapping_path: Path | None = Field(
        default=None, description="Path to YOLOX labels_mapping.json"
    )
    yolox_input_size: int = Field(default=640, description="YOLOX input size")
    yolox_confidence_threshold: float = Field(
        default=0.01, description="YOLOX confidence threshold for raw predictions"
    )
    yolox_nms_iou_threshold: float = Field(
        default=0.45, description="YOLOX NMS IoU threshold"
    )

    # OmDet-Turbo config
    run_omdet_turbo: bool = Field(
        default=False, description="Run OmDet-Turbo evaluation"
    )
    omdet_turbo_model_name: str = Field(
        default="omlab/omdet-turbo-swin-tiny-hf",
        description="OmDet-Turbo model name",
    )
    omdet_turbo_box_threshold: float = Field(
        default=0.01, description="OmDet-Turbo box confidence threshold"
    )

    # Grounding DINO config
    run_grounding_dino: bool = Field(
        default=False, description="Run Grounding DINO evaluation"
    )
    grounding_dino_model_name: str = Field(
        default="IDEA-Research/grounding-dino-base",
        description="Grounding DINO model name",
    )
    grounding_dino_box_threshold: float = Field(
        default=0.01, description="Grounding DINO box confidence threshold"
    )
    grounding_dino_text_threshold: float = Field(
        default=0.01, description="Grounding DINO text confidence threshold"
    )
    grounding_dino_nms_iou_threshold: float = Field(
        default=0.5, description="Grounding DINO NMS IoU threshold"
    )

    # Florence-2 config
    run_florence2: bool = Field(default=False, description="Run Florence-2 evaluation")
    florence2_model_name: str = Field(
        default="microsoft/Florence-2-large",
        description="Florence-2 model name",
    )
    florence2_task: str = Field(
        default="<OD>",
        description="Florence-2 task prompt (<OD> or <CAPTION_TO_PHRASE_GROUNDING>)",
    )

    # Eval config
    confidence_threshold_steps: int = Field(
        default=20, description="Number of threshold steps for F1 sweep"
    )

    def run(self) -> dict[str, str | None]:
        """Run evaluation across all configured methods."""
        if self.output_dir is None:
            self.output_dir = Path("eval_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load ground truth
        logger.info("Loading ground truth annotations...")
        val_gt = _load_coco_gt(self.val_dir / "_annotations.coco.json")
        test_gt = _load_coco_gt(self.test_dir / "_annotations.coco.json")
        logger.info(f"Val: {len(val_gt)} images, Test: {len(test_gt)} images")

        all_results: dict[str, dict[str, Any]] = {}

        # --- Gemini ---
        if self.run_gemini:
            logger.info("=" * 40 + " Gemini " + "=" * 40)
            inferencer, label_map = self._build_gemini_inferencer()
            all_results["Gemini"] = self._eval_method(
                method_name="Gemini",
                inferencer=inferencer,
                label_map=label_map,
                val_gt=val_gt,
                test_gt=test_gt,
                val_image_dir=self.val_dir,
                test_image_dir=self.test_dir,
            )

        # --- SmolVLM2 ---
        if self.run_smolvlm2:
            logger.info("=" * 40 + " SmolVLM2 " + "=" * 40)
            inferencer, label_map = self._build_smolvlm2_inferencer()
            all_results["SmolVLM2"] = self._eval_method(
                method_name="SmolVLM2",
                inferencer=inferencer,
                label_map=label_map,
                val_gt=val_gt,
                test_gt=test_gt,
                val_image_dir=self.val_dir,
                test_image_dir=self.test_dir,
            )
            # Free GPU memory
            if hasattr(inferencer, "unload"):
                inferencer.unload()

        # --- RF-DETR ---
        if self.run_rfdetr:
            logger.info("=" * 40 + " RF-DETR " + "=" * 40)
            inferencer, label_map = self._build_rfdetr_inferencer()
            all_results["RF-DETR"] = self._eval_method(
                method_name="RF-DETR",
                inferencer=inferencer,
                label_map=label_map,
                val_gt=val_gt,
                test_gt=test_gt,
                val_image_dir=self.val_dir,
                test_image_dir=self.test_dir,
            )

        # --- YOLOX ---
        if self.run_yolox:
            logger.info("=" * 40 + " YOLOX " + "=" * 40)
            inferencer, label_map = self._build_yolox_inferencer()
            all_results["YOLOX"] = self._eval_method(
                method_name="YOLOX",
                inferencer=inferencer,
                label_map=label_map,
                val_gt=val_gt,
                test_gt=test_gt,
                val_image_dir=self.val_dir,
                test_image_dir=self.test_dir,
            )

        # --- OmDet-Turbo ---
        if self.run_omdet_turbo:
            logger.info("=" * 40 + " OmDet-Turbo " + "=" * 40)
            inferencer, label_map = self._build_omdet_turbo_inferencer()
            all_results["OmDet-Turbo"] = self._eval_method(
                method_name="OmDet-Turbo",
                inferencer=inferencer,
                label_map=label_map,
                val_gt=val_gt,
                test_gt=test_gt,
                val_image_dir=self.val_dir,
                test_image_dir=self.test_dir,
                filter_area_outliers=True,
            )
            if hasattr(inferencer, "unload"):
                inferencer.unload()

        # --- Grounding DINO ---
        if self.run_grounding_dino:
            logger.info("=" * 40 + " Grounding DINO " + "=" * 40)
            inferencer, label_map = self._build_grounding_dino_inferencer()
            all_results["Grounding-DINO"] = self._eval_method(
                method_name="Grounding-DINO",
                inferencer=inferencer,
                label_map=label_map,
                val_gt=val_gt,
                test_gt=test_gt,
                val_image_dir=self.val_dir,
                test_image_dir=self.test_dir,
                filter_area_outliers=True,
            )
            if hasattr(inferencer, "unload"):
                inferencer.unload()

        # --- Florence-2 ---
        if self.run_florence2:
            logger.info("=" * 40 + " Florence-2 " + "=" * 40)
            inferencer, label_map = self._build_florence2_inferencer()
            all_results["Florence-2"] = self._eval_method(
                method_name="Florence-2",
                inferencer=inferencer,
                label_map=label_map,
                val_gt=val_gt,
                test_gt=test_gt,
                val_image_dir=self.val_dir,
                test_image_dir=self.test_dir,
                filter_area_outliers=True,
            )
            if hasattr(inferencer, "unload"):
                inferencer.unload()

        # --- Write outputs ---
        self._write_summary_csv(all_results)
        self._write_results_json(all_results)

        # Plot PR curves
        _plot_pr_curves(all_results, "val", self.output_dir / "pr_curves_val.png")
        _plot_pr_curves(all_results, "test", self.output_dir / "pr_curves_test.png")

        logger.info("=" * 60)
        logger.info("Evaluation complete!")
        self._print_summary_table(all_results)

        return {"output_dir": str(self.output_dir)}

    # ------------------------------------------------------------------
    # Method evaluation
    # ------------------------------------------------------------------

    def _eval_method(
        self,
        method_name: str,
        inferencer: BaseInferencer,
        label_map: dict[int, str],
        val_gt: dict[str, sv.Detections],
        test_gt: dict[str, sv.Detections],
        val_image_dir: Path,
        test_image_dir: Path,
        *,
        filter_area_outliers: bool = False,
    ) -> dict[str, Any]:
        """Run a single method on val and test, compute all metrics."""
        if self.output_dir is None:
            msg = "output_dir must be set before running evaluation"
            raise RuntimeError(msg)

        # Run predictions on val
        logger.info(f"[{method_name}] Running predictions on val...")
        val_preds = self._run_predictions(
            inferencer,
            label_map,
            val_gt,
            val_image_dir,
            method_name,
            "val",
            filter_area_outliers=filter_area_outliers,
        )

        # Run predictions on test
        logger.info(f"[{method_name}] Running predictions on test...")
        test_preds = self._run_predictions(
            inferencer,
            label_map,
            test_gt,
            test_image_dir,
            method_name,
            "test",
            filter_area_outliers=filter_area_outliers,
        )

        # Compute mAP on val and test
        logger.info(f"[{method_name}] Computing mAP...")
        val_metrics = _compute_metrics(val_gt, val_preds)
        test_metrics = _compute_metrics(test_gt, test_preds)

        # Find best threshold on val
        best_thresh, val_prf1 = _find_best_threshold(
            val_gt, val_preds, self.confidence_threshold_steps
        )
        logger.info(
            f"[{method_name}] Best val threshold: {best_thresh:.2f} "
            f"(P={val_prf1['precision']:.3f} R={val_prf1['recall']:.3f} "
            f"F1={val_prf1['f1']:.3f})"
        )

        # Apply best threshold to test
        test_prf1 = _compute_prf1_at_threshold(test_gt, test_preds, best_thresh)

        # Compute PR curves
        val_pr = _compute_pr_curve(val_gt, val_preds, self.confidence_threshold_steps)
        test_pr = _compute_pr_curve(
            test_gt, test_preds, self.confidence_threshold_steps
        )

        return {
            "val_mAP_50_95": val_metrics["mAP_50_95"],
            "val_mAP_50": val_metrics["mAP_50"],
            "val_mAP_75": val_metrics["mAP_75"],
            "val_per_class_ap50": val_metrics["per_class_ap50"],
            "val_precision": val_prf1["precision"],
            "val_recall": val_prf1["recall"],
            "val_f1": val_prf1["f1"],
            "val_threshold": best_thresh,
            "test_mAP_50_95": test_metrics["mAP_50_95"],
            "test_mAP_50": test_metrics["mAP_50"],
            "test_mAP_75": test_metrics["mAP_75"],
            "test_per_class_ap50": test_metrics["per_class_ap50"],
            "test_precision": test_prf1["precision"],
            "test_recall": test_prf1["recall"],
            "test_f1": test_prf1["f1"],
            "test_threshold": best_thresh,
            "val_pr_data": val_pr,
            "test_pr_data": test_pr,
        }

    def _run_predictions(
        self,
        inferencer: BaseInferencer,
        label_map: dict[int, str],
        gt_map: dict[str, sv.Detections],
        image_dir: Path,
        method_name: str,
        split: str,
        *,
        filter_area_outliers: bool = False,
    ) -> dict[str, sv.Detections]:
        """Run inferencer on all images in gt_map, return sv.Detections per image."""
        if self.output_dir is None:
            msg = "output_dir must be set before running predictions"
            raise RuntimeError(msg)
        pred_map: dict[str, sv.Detections] = {}
        raw_preds: dict[str, list[dict[str, Any]]] = {}

        for filename in tqdm(gt_map, desc=f"{method_name} {split}", unit="img"):
            img_path = image_dir / filename
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                pred_map[filename] = sv.Detections.empty()
                continue

            loader = ImageLoader(img_path)
            image = loader.read()
            detections = inferencer.predict(
                image,
                image_width=loader.width,
                image_height=loader.height,
            )

            # Remap detection class IDs to eval label map
            remapped = self._remap_detections(detections, label_map)

            # Filter outlier boxes by area (removes large crowd/court boxes)
            if filter_area_outliers:
                remapped = self._filter_area_outliers(remapped)

            sv_dets = _detections_to_sv(remapped, loader.width, loader.height)
            pred_map[filename] = sv_dets

            # Store raw predictions for JSON output
            raw_preds[filename] = [
                {
                    "bbox_xyxy": box.tolist(),
                    "class_id": int(cls_id),
                    "confidence": float(conf) if conf is not None else 1.0,
                }
                for box, cls_id, conf in zip(
                    sv_dets.xyxy,
                    sv_dets.class_id if sv_dets.class_id is not None else [],
                    sv_dets.confidence if sv_dets.confidence is not None else [],
                    strict=False,
                )
            ]

        # Save raw predictions
        pred_path = self.output_dir / f"predictions_{method_name.lower()}_{split}.json"
        with open(pred_path, "w") as f:
            json.dump(raw_preds, f, indent=2)
        logger.info(f"Saved predictions to {pred_path}")

        return pred_map

    @staticmethod
    def _remap_detections(
        detections: list[Detection],
        label_map: dict[int, str],
    ) -> list[Detection]:
        """Remap inferencer class IDs to eval label map IDs.

        Each inferencer may use its own class numbering (e.g. RF-DETR
        uses 10 training classes, Gemini uses whatever order its
        ``classes`` list defines).  This method translates each
        detection's ``class_id`` to the unified eval label map via:

        1. Look up the class *name* from the inferencer's ``label_map``.
        2. Map that name to the eval ID through ``_NAME_TO_EVAL_ID``.
        3. Drop detections whose class has no eval mapping.
        """
        remapped: list[Detection] = []
        for det in detections:
            class_name = label_map.get(det.class_id)
            if class_name is None:
                logger.debug(
                    f"Skipping detection with unknown class_id={det.class_id} "
                    f"(not in label_map)"
                )
                continue

            eval_id = _NAME_TO_EVAL_ID.get(class_name.lower())
            if eval_id is None:
                logger.debug(
                    f"Skipping detection with class {class_name!r} (no eval mapping)"
                )
                continue

            remapped.append(
                Detection(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=eval_id,
                )
            )
        return remapped

    @staticmethod
    def _filter_area_outliers(
        detections: list[Detection],
        max_area_fraction: float = 0.05,
    ) -> list[Detection]:
        """Remove detections whose normalised area exceeds a fraction of the image.

        Zero-shot detectors often produce large spurious boxes covering
        crowd regions or the entire court.  Real player/object boxes are
        typically <3% of the image area, while spurious boxes are >10%.

        Args:
            detections: List of detections with normalised bboxes.
            max_area_fraction: Maximum allowed normalised area (w*h).
                Boxes exceeding this are dropped.  Default 0.05 (5%).
        """
        filtered = [d for d in detections if d.bbox.w * d.bbox.h <= max_area_fraction]
        n_removed = len(detections) - len(filtered)
        if n_removed > 0:
            logger.debug(
                f"Area filter removed {n_removed}/{len(detections)} boxes "
                f"exceeding {max_area_fraction:.0%} of image area"
            )
        return filtered

    # ------------------------------------------------------------------
    # Inferencer builders
    # ------------------------------------------------------------------

    def _build_gemini_inferencer(
        self,
    ) -> tuple[BaseInferencer, dict[int, str]]:
        from object_detection_training.inference.gemini_inferencer import (
            GeminiInferencer,
        )

        label_map = dict(enumerate(self.gemini_classes))
        inferencer = GeminiInferencer(
            model_name=self.gemini_model_name,
            classes=self.gemini_classes,
            prompt_template=self.gemini_prompt_template,
        )
        return inferencer, label_map

    def _build_smolvlm2_inferencer(
        self,
    ) -> tuple[BaseInferencer, dict[int, str]]:
        from object_detection_training.inference.smolvlm2_inferencer import (
            SmolVLM2Inferencer,
        )

        classes = list(_EVAL_LABEL_MAP.values())
        label_map = dict(enumerate(classes))
        inferencer = SmolVLM2Inferencer(
            model_name=self.smolvlm2_model_name,
            classes=classes,
        )
        return inferencer, label_map

    def _build_rfdetr_inferencer(
        self,
    ) -> tuple[BaseInferencer, dict[int, str]]:
        if self.onnx_model_path is None:
            msg = "onnx_model_path is required when run_rfdetr=True"
            raise ValueError(msg)

        from object_detection_training.inference.onnx_inferencer import ONNXInferencer
        from object_detection_training.inference.postprocess import RFDETRPostProcessor
        from object_detection_training.schemas.label_mapping import LabelMapping

        # Load label mapping
        if self.label_mapping_path is not None:
            mapping = LabelMapping.from_json(self.label_mapping_path)
            label_map = {int(k): v for k, v in mapping.id_to_name.items()}
        else:
            label_map = dict(_EVAL_LABEL_MAP)

        post_processor = RFDETRPostProcessor(
            label_map=label_map,
            confidence_threshold=self.rfdetr_confidence_threshold,
        )

        inferencer = ONNXInferencer(
            model_path=self.onnx_model_path,
            post_processor=post_processor,
            input_height=self.rfdetr_input_size,
            input_width=self.rfdetr_input_size,
        )
        return inferencer, label_map

    def _build_yolox_inferencer(
        self,
    ) -> tuple[BaseInferencer, dict[int, str]]:
        if self.yolox_onnx_model_path is None:
            msg = "yolox_onnx_model_path is required when run_yolox=True"
            raise ValueError(msg)

        from object_detection_training.inference.onnx_inferencer import ONNXInferencer
        from object_detection_training.inference.postprocess import YOLOXPostProcessor
        from object_detection_training.schemas.label_mapping import LabelMapping

        # Load label mapping
        if self.yolox_label_mapping_path is not None:
            mapping = LabelMapping.from_json(self.yolox_label_mapping_path)
            label_map = {int(k): v for k, v in mapping.id_to_name.items()}
        else:
            label_map = dict(_EVAL_LABEL_MAP)

        post_processor = YOLOXPostProcessor(
            label_map=label_map,
            confidence_threshold=self.yolox_confidence_threshold,
            nms_iou_threshold=self.yolox_nms_iou_threshold,
            model_input_size=(self.yolox_input_size, self.yolox_input_size),
        )

        # YOLOX uses raw 0-255 float input (no ImageNet normalization)
        inferencer = ONNXInferencer(
            model_path=self.yolox_onnx_model_path,
            post_processor=post_processor,
            input_height=self.yolox_input_size,
            input_width=self.yolox_input_size,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
        )
        return inferencer, label_map

    def _build_omdet_turbo_inferencer(
        self,
    ) -> tuple[BaseInferencer, dict[int, str]]:
        from object_detection_training.inference.omdet_turbo_inferencer import (
            OmDetTurboInferencer,
        )

        classes = [
            "person",
            "sports ball",
            "referee",
            "basketball hoop",
            "jersey number",
        ]
        label_map = dict(enumerate(classes))
        inferencer = OmDetTurboInferencer(
            model_name=self.omdet_turbo_model_name,
            classes=classes,
            box_threshold=self.omdet_turbo_box_threshold,
        )
        return inferencer, label_map

    def _build_grounding_dino_inferencer(
        self,
    ) -> tuple[BaseInferencer, dict[int, str]]:
        from object_detection_training.inference.grounding_dino_inferencer import (
            GroundingDINOInferencer,
        )

        classes = [
            "person",
            "sports ball",
            "referee",
            "basketball hoop",
            "jersey number",
        ]
        label_map = dict(enumerate(classes))
        inferencer = GroundingDINOInferencer(
            model_name=self.grounding_dino_model_name,
            classes=classes,
            box_threshold=self.grounding_dino_box_threshold,
            text_threshold=self.grounding_dino_text_threshold,
            nms_iou_threshold=self.grounding_dino_nms_iou_threshold,
        )
        return inferencer, label_map

    def _build_florence2_inferencer(
        self,
    ) -> tuple[BaseInferencer, dict[int, str]]:
        from object_detection_training.inference.florence2_inferencer import (
            Florence2Inferencer,
        )

        classes = [
            "person",
            "sports ball",
            "referee",
            "basketball hoop",
            "jersey number",
        ]
        label_map = dict(enumerate(classes))
        caption = self.gemini_prompt_template or ""
        inferencer = Florence2Inferencer(
            model_name=self.florence2_model_name,
            classes=classes,
            caption=caption,
            task=self.florence2_task,
        )
        return inferencer, label_map

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write_summary_csv(self, all_results: dict[str, dict[str, Any]]) -> None:
        if self.output_dir is None:
            return
        csv_path = self.output_dir / "summary.csv"
        fieldnames = [
            "Method",
            "Split",
            "mAP@50:95",
            "mAP@50",
            "mAP@75",
            "Precision",
            "Recall",
            "F1",
            "Threshold",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for method, res in all_results.items():
                for split in ["val", "test"]:
                    writer.writerow(
                        {
                            "Method": method,
                            "Split": split,
                            "mAP@50:95": f"{res[f'{split}_mAP_50_95']:.4f}",
                            "mAP@50": f"{res[f'{split}_mAP_50']:.4f}",
                            "mAP@75": f"{res[f'{split}_mAP_75']:.4f}",
                            "Precision": f"{res[f'{split}_precision']:.4f}",
                            "Recall": f"{res[f'{split}_recall']:.4f}",
                            "F1": f"{res[f'{split}_f1']:.4f}",
                            "Threshold": f"{res[f'{split}_threshold']:.2f}",
                        }
                    )

        logger.info(f"Summary CSV saved to {csv_path}")

    def _write_results_json(self, all_results: dict[str, dict[str, Any]]) -> None:
        if self.output_dir is None:
            return
        json_path = self.output_dir / "results.json"

        # Remove non-serialisable PR curve data
        serialisable = {}
        for method, res in all_results.items():
            serialisable[method] = {
                k: v for k, v in res.items() if not k.endswith("_pr_data")
            }

        with open(json_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"Results JSON saved to {json_path}")

    def _print_summary_table(self, all_results: dict[str, dict[str, Any]]) -> None:
        header = (
            f"{'Method':<10} | {'Split':<5} | "
            f"{'mAP@50:95':>9} | {'mAP@50':>6} | "
            f"{'mAP@75':>6} | {'Precision':>9} | "
            f"{'Recall':>6} | {'F1':>6} | "
            f"{'Threshold':>9}"
        )
        sep = "-" * len(header)
        logger.info(sep)
        logger.info(header)
        logger.info(sep)

        for method, res in all_results.items():
            for split in ["val", "test"]:
                row = (
                    f"{method:<10} | {split:<5} | "
                    f"{res[f'{split}_mAP_50_95']:>9.4f} | "
                    f"{res[f'{split}_mAP_50']:>6.4f} | "
                    f"{res[f'{split}_mAP_75']:>6.4f} | "
                    f"{res[f'{split}_precision']:>9.4f} | "
                    f"{res[f'{split}_recall']:>6.4f} | "
                    f"{res[f'{split}_f1']:>6.4f} | "
                    f"{res[f'{split}_threshold']:>9.2f}"
                )
                logger.info(row)
        logger.info(sep)
