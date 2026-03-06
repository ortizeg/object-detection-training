"""Florence-2 zero-shot object detection inferencer."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image
from transformers import (  # type: ignore[attr-defined]
    AutoModelForCausalLM,
    AutoProcessor,
)

from object_detection_training.inference.base_inferencer import BaseInferencer
from object_detection_training.schemas.detection import BoundingBox, Detection
from object_detection_training.utils.boxes import pixel_xyxy_to_normalized_xywh


class Florence2Inferencer(BaseInferencer):
    """Run zero-shot object detection using Florence-2.

    Supports two task modes:
    - ``<OD>``: General object detection (no caption needed).
    - ``<CAPTION_TO_PHRASE_GROUNDING>``: Caption-guided phrase grounding.

    Florence-2 does not produce confidence scores, so all detections
    are assigned ``default_confidence``.

    Args:
        model_name: HuggingFace model ID.
        classes: Ordered list of class names (index = class ID).
        caption: Caption text for ``<CAPTION_TO_PHRASE_GROUNDING>`` task.
        task: Florence-2 task prompt (``"<OD>"`` or
            ``"<CAPTION_TO_PHRASE_GROUNDING>"``).
        default_confidence: Confidence assigned to all detections.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        classes: list[str] | None = None,
        caption: str = "",
        task: str = "<OD>",
        default_confidence: float = 1.0,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.caption = caption
        self._task = task
        self.default_confidence = default_confidence

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

        logger.info(f"Loading Florence-2 model {model_name} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
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
            prompt = self._task if self._task == "<OD>" else self._task + self.caption
            inputs = self._processor(
                text=prompt,
                images=pil_img,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )

            generated_text: str = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self._processor.post_process_generation(
                generated_text,
                task=self._task,
                image_size=(w, h),
            )

            return self._convert_results(parsed, w, h)

        except Exception:
            logger.exception("Florence-2 inference failed")
            return []

    def unload(self) -> None:
        """Free GPU memory."""
        del self._model
        del self._processor
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        logger.info("Florence-2 model unloaded")

    def _convert_results(
        self,
        parsed: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert Florence-2 post-processed results to Detection list."""
        task_result = parsed.get(self._task, {})
        if not isinstance(task_result, dict):
            return []

        boxes = task_result.get("bboxes", [])
        labels = task_result.get("labels", [])

        detections: list[Detection] = []
        for box, label in zip(boxes, labels, strict=False):
            class_id = self._name_to_id.get(label.lower().strip())
            if class_id is None:
                logger.debug(f"Label {label!r} not in class map - skipping")
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(
                x1, y1, x2, y2, image_width, image_height
            )

            detections.append(
                Detection(
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=self.default_confidence,
                    class_id=class_id,
                )
            )

        return detections
