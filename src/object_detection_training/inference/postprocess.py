"""Post-processing strategies for different ONNX detection models.

Each post-processor converts raw ONNX outputs into a list of
:class:`Detection` objects with normalised bounding boxes.

Subclass :class:`BasePostProcessor` to support a new model family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_training.schemas.detection import BoundingBox, Detection


class BasePostProcessor(ABC):
    """Base class for ONNX model post-processing.

    Args:
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
    """

    def __init__(
        self,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
    ) -> None:
        self.label_map = label_map
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert raw ONNX outputs to a list of detections.

        Args:
            outputs: Raw numpy arrays from ``onnxruntime.InferenceSession.run``.
            image_width: Original image width (for normalisation).
            image_height: Original image height (for normalisation).

        Returns:
            Filtered list of :class:`Detection` objects.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_detection(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        confidence: float,
        class_id: int,
    ) -> Detection | None:
        """Create a single Detection with boundary clamping.

        Returns ``None`` when the class id is missing from the label map.
        """
        label = self.label_map.get(class_id)
        if label is None:
            logger.debug(f"Unknown class id {class_id}, skipping")
            return None

        bbox = BoundingBox(
            x=float(np.clip(x, 0.0, 1.0)),
            y=float(np.clip(y, 0.0, 1.0)),
            w=float(np.clip(w, 0.0, 1.0)),
            h=float(np.clip(h, 0.0, 1.0)),
        )
        return Detection(bbox=bbox, confidence=float(confidence), label=label)


class YOLOXPostProcessor(BasePostProcessor):
    """YOLOX-style post-processing.

    Expected ONNX output: single tensor of shape
    ``[batch, num_anchors, 5 + num_classes]`` where columns are
    ``[cx, cy, w, h, obj_conf, cls_0, cls_1, ...]`` in **pixel** coords.

    Applies objectness * class confidence scoring and greedy NMS.
    """

    def __init__(
        self,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        super().__init__(label_map, confidence_threshold)
        self.nms_iou_threshold = nms_iou_threshold

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Decode YOLOX predictions for a single image."""
        # outputs[0] shape: [1, num_anchors, 5+num_classes]
        pred = np.asarray(outputs[0], dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]  # remove batch dim

        # Score = obj_conf * max(class_conf)
        obj_conf = pred[:, 4]
        cls_conf = pred[:, 5:]
        class_ids = cls_conf.argmax(axis=1)
        class_scores = cls_conf[np.arange(len(cls_conf)), class_ids]
        scores = obj_conf * class_scores

        # Threshold
        mask = scores > self.confidence_threshold
        pred = pred[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return []

        # cxcywh (pixel) -> normalised xywh (top-left)
        cx = pred[:, 0] / image_width
        cy = pred[:, 1] / image_height
        w = pred[:, 2] / image_width
        h = pred[:, 3] / image_height
        x1 = cx - w / 2
        y1 = cy - h / 2

        # NMS (greedy, per-class)
        keep = self._nms(x1, y1, w, h, scores, class_ids)

        detections: list[Detection] = []
        for idx in keep:
            det = self._make_detection(
                x=float(x1[idx]),
                y=float(y1[idx]),
                w=float(w[idx]),
                h=float(h[idx]),
                confidence=float(scores[idx]),
                class_id=int(class_ids[idx]),
            )
            if det is not None:
                detections.append(det)

        return detections

    # ------------------------------------------------------------------
    # Greedy NMS (numpy, no torchvision dependency at inference)
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(
        box: npt.NDArray[np.floating[Any]],
        boxes: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute IoU between one box and many boxes (all in xywh)."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
        y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])

        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = box[2] * box[3]
        area_b = boxes[:, 2] * boxes[:, 3]
        union = area_a + area_b - inter
        return inter / np.maximum(union, 1e-6)  # type: ignore[no-any-return]

    def _nms(
        self,
        x: npt.NDArray[np.floating[Any]],
        y: npt.NDArray[np.floating[Any]],
        w: npt.NDArray[np.floating[Any]],
        h: npt.NDArray[np.floating[Any]],
        scores: npt.NDArray[np.floating[Any]],
        class_ids: npt.NDArray[np.integer[Any]],
    ) -> list[int]:
        """Per-class greedy NMS. Returns indices to keep."""
        boxes = np.stack([x, y, w, h], axis=1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while len(order) > 0:
            i = int(order[0])
            keep.append(i)

            if len(order) == 1:
                break

            rest = order[1:]
            # Only suppress within same class
            same_class = class_ids[rest] == class_ids[i]
            ious = self._iou(boxes[i], boxes[rest])
            suppress = same_class & (ious > self.nms_iou_threshold)
            order = rest[~suppress]

        return keep


class RFDETRPostProcessor(BasePostProcessor):
    """RFDETR-style post-processing.

    Expected ONNX outputs:
      - ``outputs[0]``: logits ``[1, num_queries, num_classes]``
      - ``outputs[1]``: boxes  ``[1, num_queries, 4]`` in normalised cxcywh

    Uses sigmoid activation (focal loss convention) and simple thresholding.
    """

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Decode RFDETR predictions for a single image."""
        # Dynamically identify outputs by shape (boxes always have last dim 4)
        out0 = np.asarray(outputs[0], dtype=np.float32)
        out1 = np.asarray(outputs[1], dtype=np.float32)

        if out0.ndim == 3:
            out0 = out0[0]
        if out1.ndim == 3:
            out1 = out1[0]

        if out0.shape[-1] == 4:
            boxes = out0
            logits = out1
        elif out1.shape[-1] == 4:
            boxes = out1
            logits = out0
        else:
            # Fallback (assume logits first) or raise error?
            # If neither has shape 4, we likely have bigger issues.
            # Let's assume logits first as per docstring, but log a warning.
            logger.warning(
                "Could not identify boxes by shape (expected last dim 4). "
                "Assuming outputs[0]=logits."
            )
            logits = out0
            boxes = out1

        # Sigmoid activation
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Per-query best class
        class_ids = probs.argmax(axis=1)
        scores = probs[np.arange(len(probs)), class_ids]

        # Threshold
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        detections: list[Detection] = []
        for i in range(len(scores)):
            cx, cy, bw, bh = boxes[i]
            det = self._make_detection(
                x=float(cx - bw / 2),
                y=float(cy - bh / 2),
                w=float(bw),
                h=float(bh),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
            )
            if det is not None:
                detections.append(det)

        return detections
