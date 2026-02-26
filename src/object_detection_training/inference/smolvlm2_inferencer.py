"""SmolVLM2 model inference engine for zero-shot object detection."""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from object_detection_training.inference.base_inferencer import BaseInferencer
from object_detection_training.schemas.detection import BoundingBox, Detection


class _SmolVLM2BBox(BaseModel):
    """Bounding box in 0-1000 corner coordinates for VLM responses."""

    x_min: int = Field(description="Left edge (0-1000)")
    y_min: int = Field(description="Top edge (0-1000)")
    x_max: int = Field(description="Right edge (0-1000)")
    y_max: int = Field(description="Bottom edge (0-1000)")


class _SmolVLM2Detection(BaseModel):
    """VLM response schema for a single detection."""

    bbox: _SmolVLM2BBox
    label: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SmolVLM2Inferencer(BaseInferencer):
    """Run zero-shot object detection using SmolVLM2.

    SmolVLM2 has no native bounding-box grounding, so detection quality
    will likely be poor.  This establishes a baseline for future
    fine-tuning.

    The model is prompted to return JSON bounding boxes in a 0-1000
    coordinate system (same format as Gemini).  Responses are parsed
    with the shared ``GeminiDetection`` Pydantic schema.

    Args:
        model_name: HuggingFace model ID.
        classes: Full ordered list of class names (index = class ID).
        prompt_template: Optional custom prompt template.
        device: Device string (``"cuda"``, ``"cpu"``, or ``"auto"``).
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        classes: list[str] | None = None,
        prompt_template: str | None = None,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []

        # Build normalised lookup: lower-cased class name -> class index
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        logger.info(f"Loading SmolVLM2 model {model_name} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        ).to(self._device)

        class_list = ", ".join(self.classes) if self.classes else "all objects"
        self._prompt = prompt_template or (
            f"Detect all instances of: {class_list}. "
            "Return a JSON array where each element has: "
            '"bbox" with "x_min", "y_min", "x_max", "y_max" (integers 0-1000), '
            '"label" (string), and "confidence" (float 0-1). '
            "Return ONLY the JSON array, no other text."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single image.

        Args:
            image: BGR uint8 image (OpenCV convention).
            image_width: Image width in pixels (for coordinate normalisation).
            image_height: Image height in pixels (for coordinate normalisation).

        Returns:
            List of Detection objects with class IDs matching ``self.classes``.
        """
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        # Convert BGR -> RGB for PIL
        rgb_image = Image.fromarray(image[..., ::-1])

        try:
            # Build chat-style messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rgb_image},
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ]

            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=1024)

            # Decode only the newly generated tokens
            prompt_len = inputs["input_ids"].shape[-1]
            generated_text: str = self._processor.decode(
                generated_ids[0][prompt_len:], skip_special_tokens=True
            )

            logger.debug(f"SmolVLM2 raw output: {generated_text[:500]}")
            return self._parse_response(generated_text, w, h)

        except Exception:
            logger.exception("SmolVLM2 inference failed")
            return []

    def unload(self) -> None:
        """Free GPU memory by deleting the model and processor."""
        del self._model
        del self._processor
        if self._device == "cuda":
            torch.cuda.empty_cache()
        logger.info("SmolVLM2 model unloaded")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_label(self, raw_label: str) -> int | None:
        """Map a label string to a class index.

        Matching strategy (first match wins):
        1. Exact match (case-insensitive).
        2. Substring containment (label in class name or vice-versa).
        """
        label = raw_label.lower().strip()

        # 1. Exact match
        if label in self._name_to_id:
            return self._name_to_id[label]

        # 2. Substring match (prefer shortest class name that contains label)
        candidates: list[tuple[int, str]] = []
        for name, idx in self._name_to_id.items():
            if label in name or name in label:
                candidates.append((idx, name))

        if candidates:
            candidates.sort(key=lambda t: len(t[1]))
            return candidates[0][0]

        return None

    def _map_detections(
        self,
        raw_dets: list[_SmolVLM2Detection],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert parsed detections to internal Detection format."""
        results: list[Detection] = []
        for det in raw_dets:
            class_id = self._resolve_label(det.label)
            if class_id is None:
                logger.warning(
                    f"Label {det.label!r} not in class map "
                    f"{list(self._name_to_id.keys())} - skipping",
                )
                continue

            # Convert from 0-1000 xyxy to 0-1 xywh
            x = det.bbox.x_min / 1000.0
            y = det.bbox.y_min / 1000.0
            w = (det.bbox.x_max - det.bbox.x_min) / 1000.0
            h = (det.bbox.y_max - det.bbox.y_min) / 1000.0

            results.append(
                Detection(
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    confidence=det.confidence,
                    class_id=class_id,
                )
            )
        return results

    def _parse_response(
        self,
        text: str,
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Parse JSON response from SmolVLM2 output."""
        # Try to extract JSON array from text
        json_str = self._extract_json(text)
        if json_str is None:
            logger.warning("No JSON array found in SmolVLM2 response")
            return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse SmolVLM2 JSON: {json_str[:200]}")
            return []

        if not isinstance(data, list):
            logger.error(f"Expected JSON list, got {type(data).__name__}")
            return []

        raw_dets: list[_SmolVLM2Detection] = []
        for item in data:
            try:
                raw_dets.append(_SmolVLM2Detection.model_validate(item))
            except Exception:
                logger.debug(f"Skipping unparseable item: {item}")

        return self._map_detections(raw_dets, image_width, image_height)

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract the first JSON array from text."""
        # Find first '[' and matching ']'
        start = text.find("[")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None
