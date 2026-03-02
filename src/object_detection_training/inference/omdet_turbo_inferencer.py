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
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.box_threshold = box_threshold

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
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
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

            return self._convert_results(results, w, h)

        except Exception:
            logger.exception("OmDet-Turbo inference failed")
            return []

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
        labels = results["text"]

        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            class_id = self._name_to_id.get(label.lower().strip())
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
