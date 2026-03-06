"""OmDet-Turbo zero-shot object detection inferencer."""

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


class OmDetTurboInferencer(BaseInferencer):
    """Run zero-shot object detection using OmDet-Turbo.

    Uses the HuggingFace ``AutoModelForZeroShotObjectDetection`` API
    with a simple class-list input (no free-text prompt support).

    Args:
        model_name: HuggingFace model ID.
        classes: Ordered list of class names (index = class ID).
        box_threshold: Minimum confidence for detections.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).
    """

    def __init__(
        self,
        model_name: str = "omlab/omdet-turbo-swin-tiny-hf",
        classes: list[str] | None = None,
        box_threshold: float = 0.01,
        nms_iou_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.box_threshold = box_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # Build normalised lookup: lower-cased class name -> class index
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

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

        logger.info(f"Loading OmDet-Turbo model {model_name} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(model_name)
        # OmDet-Turbo uses a timm SwinTransformer backbone whose attention
        # mask buffers are registered on the meta device.  We load without
        # device_map, then materialise any remaining meta buffers to CPU
        # before moving to the target device.
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        self._materialize_meta_buffers(self._model)
        if self._device != "cpu":
            self._model = self._model.to(self._device)

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
            # OmDet-Turbo takes a list of class labels per image
            inputs = self._processor(
                images=pil_img,
                text=[self.classes],
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                threshold=self.box_threshold,
                target_sizes=[(h, w)],
            )[0]

            detections = self._convert_results(results, w, h)
            return self._nms(detections)

        except Exception:
            logger.exception("OmDet-Turbo inference failed")
            return []

    @staticmethod
    def _materialize_meta_buffers(model: torch.nn.Module) -> None:
        """Replace any remaining meta-device buffers with real CPU tensors.

        The timm SwinTransformer registers attention-mask buffers on the
        meta device during ``__init__``.  These survive
        ``from_pretrained`` because they are buffers (not parameters) and
        are not stored in the checkpoint.  We walk the module tree and
        replace them with zero-filled tensors on CPU so that
        ``.to(device)`` succeeds.
        """
        n_fixed = 0
        for module in model.modules():
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf is not None and buf.device.type == "meta":
                    real = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                    module.register_buffer(name, real)
                    n_fixed += 1
        if n_fixed > 0:
            logger.debug(f"Materialised {n_fixed} meta buffer(s) to CPU")

    def unload(self) -> None:
        """Free GPU memory."""
        del self._model
        del self._processor
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        logger.info("OmDet-Turbo model unloaded")

    def _convert_results(
        self,
        results: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert HF post-processed results to Detection list."""
        boxes = results["boxes"]
        scores = results["scores"]
        # transformers >=4.51: "text" may be removed; "text_labels" can be
        # None; fall back to "labels" (integer IDs).
        labels = (
            results.get("text")
            or results.get("text_labels")
            or results.get("labels", [])
        )

        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if isinstance(label, str):
                class_id = self._name_to_id.get(label.lower().strip())
            else:
                label_idx = int(label)
                class_id = label_idx if 0 <= label_idx < len(self.classes) else None
            if class_id is None:
                logger.debug(f"Label {label!r} not in class map - skipping")
                continue

            if isinstance(box, torch.Tensor):
                x1, y1, x2, y2 = box.tolist()
            else:
                x1, y1, x2, y2 = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )

            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(
                x1, y1, x2, y2, image_width, image_height
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
