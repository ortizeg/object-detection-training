"""Grounding DINO zero-shot object detection inferencer."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image
from transformers import (  # type: ignore[attr-defined]
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from object_detection_training.inference.base_inferencer import BaseInferencer
from object_detection_training.schemas.detection import BoundingBox, Detection
from object_detection_training.utils.boxes import pixel_xyxy_to_normalized_xywh


class GroundingDINOInferencer(BaseInferencer):
    """Run zero-shot object detection using Grounding DINO.

    Uses the HuggingFace ``AutoModelForZeroShotObjectDetection`` API
    with a period-separated class prompt (e.g. ``"player . ball . referee"``).

    Args:
        model_name: HuggingFace model ID.
        classes: Ordered list of class names (index = class ID).
        box_threshold: Minimum confidence for box filtering.
        text_threshold: Minimum confidence for text/class filtering.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).
    """

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-base",
        classes: list[str] | None = None,
        box_threshold: float = 0.01,
        text_threshold: float = 0.01,
        nms_iou_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # Build normalised lookup: lower-cased class name -> class index
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

        # Build period-separated prompt
        self._text_prompt = " . ".join(self.classes) + " ." if self.classes else ""

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        logger.info(f"Loading Grounding DINO model {model_name} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single BGR image."""
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        # BGR -> RGB PIL
        pil_img = Image.fromarray(image[..., ::-1])

        try:
            inputs = self._processor(
                images=pil_img,
                text=self._text_prompt,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]

            detections = self._convert_results(results, w, h)
            return self._nms(detections)

        except Exception:
            logger.exception("Grounding DINO inference failed")
            return []

    def unload(self) -> None:
        """Free GPU memory."""
        del self._model
        del self._processor
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        logger.info("Grounding DINO model unloaded")

    # Classes that should only appear on small boxes (fraction of image area).
    # If a box exceeds this threshold, fall back to the next matching class.
    _SMALL_OBJECT_CLASSES: frozenset[str] = frozenset({"jersey number"})
    _SMALL_OBJECT_MAX_AREA: float = 0.01  # 1% of image area

    def _resolve_label(
        self,
        label: str,
        box_area_fraction: float = 0.0,
    ) -> int | None:
        """Resolve a Grounding DINO label string to a class ID.

        The HuggingFace processor can return concatenated labels like
        ``"person referee jersey number"`` when multiple phrase tokens
        activate for the same box.  We pick the class whose name appears
        *earliest* in the label string (the primary class), with a
        size-based sanity check for small-object classes.
        """
        normalized = label.lower().strip()

        # Fast path: exact match
        exact = self._name_to_id.get(normalized)
        if exact is not None:
            # Check size sanity for small-object classes
            if (
                normalized in self._SMALL_OBJECT_CLASSES
                and box_area_fraction > self._SMALL_OBJECT_MAX_AREA
            ):
                return None
            return exact

        # Find all class names contained in the label, sorted by position
        matches: list[tuple[int, str, int]] = []  # (position, name, class_id)
        for name, class_id in self._name_to_id.items():
            pos = normalized.find(name)
            if pos != -1:
                matches.append((pos, name, class_id))
        matches.sort()

        # Return the first match that passes the size sanity check
        for _pos, name, class_id in matches:
            if (
                name in self._SMALL_OBJECT_CLASSES
                and box_area_fraction > self._SMALL_OBJECT_MAX_AREA
            ):
                continue
            return class_id
        return None

    def _convert_results(
        self,
        results: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert HF post-processed results to Detection list."""
        boxes = results["boxes"]
        scores = results["scores"]
        # transformers >=4.51: "text" removed, "text_labels" may be None,
        # "labels" returns integer IDs.
        labels = (
            results.get("text")
            or results.get("text_labels")
            or results.get("labels", [])
        )

        detections: list[Detection] = []
        img_area = float(image_width * image_height)
        for box, score, label in zip(boxes, scores, labels, strict=False):
            # Compute normalized box area for size-based label sanity checks
            if isinstance(box, torch.Tensor):
                bx1, by1, bx2, by2 = box.tolist()
            else:
                bx1, by1, bx2, by2 = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )
            box_area_frac = (bx2 - bx1) * (by2 - by1) / img_area if img_area else 0.0

            # Handle both string labels and integer IDs
            if isinstance(label, str):
                class_id = self._resolve_label(label, box_area_frac)
            else:
                # Integer label ID — map through classes list if in range
                label_idx = int(label)
                class_id = label_idx if 0 <= label_idx < len(self.classes) else None
            if class_id is None:
                logger.debug(f"Label {label!r} not in class map - skipping")
                continue

            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(
                bx1, by1, bx2, by2, image_width, image_height
            )

            conf = float(score) if not isinstance(score, float) else score
            detections.append(
                Detection(
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=conf,
                    class_id=class_id,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Per-class greedy NMS on normalized xywh boxes
    # ------------------------------------------------------------------

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """Apply per-class greedy NMS to remove duplicate boxes."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep: list[Detection] = []
        suppressed = [False] * len(dets)

        for i, det_i in enumerate(dets):
            if suppressed[i]:
                continue
            keep.append(det_i)
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                if dets[j].class_id != det_i.class_id:
                    continue
                if self._iou(det_i, dets[j]) > self.nms_iou_threshold:
                    suppressed[j] = True

        return keep

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        """Compute IoU between two detections (normalized xywh boxes)."""
        ax1, ay1 = a.bbox.x, a.bbox.y
        ax2, ay2 = a.bbox.x + a.bbox.w, a.bbox.y + a.bbox.h
        bx1, by1 = b.bbox.x, b.bbox.y
        bx2, by2 = b.bbox.x + b.bbox.w, b.bbox.y + b.bbox.h

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = a.bbox.w * a.bbox.h
        area_b = b.bbox.w * b.bbox.h
        union = area_a + area_b - inter
        return inter / max(union, 1e-9)
