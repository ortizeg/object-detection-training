"""Gemini model inference engine."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import numpy.typing as npt
from google import genai
from google.genai.types import GenerateContentConfig
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from object_detection_training.inference.base_inferencer import BaseInferencer
from object_detection_training.schemas.detection import BoundingBox, Detection


class GeminiBBox(BaseModel):
    """Bounding box in top-left x/y, width, height format normalized to [0, 1]."""

    x: float = Field(description="Top-left x (normalised 0-1)")
    y: float = Field(description="Top-left y (normalised 0-1)")
    w: float = Field(description="Width (normalised 0-1)")
    h: float = Field(description="Height (normalised 0-1)")


class GeminiDetection(BaseModel):
    """Single detection result from Gemini."""

    bbox: GeminiBBox
    label: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GeminiInferencer(BaseInferencer):
    """Run inference on images using Google's Gemini models.

    The ``classes`` list defines the full label map (index → name). Gemini
    may return only a subset of these labels; each returned label is mapped
    to its index in ``classes`` via case-insensitive matching.

    Args:
        model_name: Name of the Gemini model to use (e.g., 'gemini-2.5-pro').
        classes: Full ordered list of class names (defines class IDs by index).
        prompt_template: Optional custom prompt template.
    """

    _MAX_RETRIES: int = 5
    _INITIAL_BACKOFF: float = 5.0

    def __init__(
        self,
        model_name: str,
        classes: list[str],
        prompt_template: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.classes = classes

        # Build a normalised lookup: lower-cased class name → class index
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(classes)
        }

        # Configure API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            msg = (
                "GOOGLE_API_KEY not set. "
                "Export it before running: export GOOGLE_API_KEY=<key>"
            )
            raise RuntimeError(msg)

        self._client = genai.Client(api_key=api_key)
        self._config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[GeminiDetection],
        )

        self._prompt = prompt_template or (
            f"Detect all instances of: {', '.join(classes)}. "
            "Return bounding boxes normalised to [0, 1] in (x, y, w, h) "
            "format where x,y is the top-left corner."
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
        """Run inference on a single image with retry on transient errors.

        Args:
            image: BGR uint8 image (OpenCV convention).
            image_width: Original width (unused, kept for interface compat).
            image_height: Original height (unused, kept for interface compat).

        Returns:
            List of Detection objects with class IDs matching ``self.classes``.
        """
        # Convert BGR → RGB for PIL
        rgb_image = Image.fromarray(image[..., ::-1])

        backoff = self._INITIAL_BACKOFF
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                contents: list[str | Image.Image] = [self._prompt, rgb_image]
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,  # type: ignore[arg-type]
                    config=self._config,
                )

                if response.parsed:
                    parsed: list[GeminiDetection] = response.parsed  # type: ignore[assignment]
                    return self._map_detections(parsed)

                if response.text:
                    return self._parse_text_fallback(response.text)

                logger.warning("Gemini returned an empty response.")
                return []

            except Exception as exc:
                exc_str = str(exc)
                is_retryable = any(
                    code in exc_str for code in ("503", "429", "UNAVAILABLE")
                )
                if is_retryable and attempt < self._MAX_RETRIES:
                    logger.warning(
                        "Attempt %d/%d failed (%s). Retrying in %.0fs...",
                        attempt,
                        self._MAX_RETRIES,
                        exc_str[:80],
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2  # exponential backoff
                    continue

                logger.exception("Gemini inference failed after %d attempts", attempt)
                return []

        return []  # unreachable, but keeps mypy happy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_label(self, raw_label: str) -> int | None:
        """Map a Gemini label string to a class index.

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
            # Pick the shortest matching class name (most specific)
            candidates.sort(key=lambda t: len(t[1]))
            return candidates[0][0]

        return None

    def _map_detections(self, gemini_dets: list[GeminiDetection]) -> list[Detection]:
        """Convert a list of ``GeminiDetection`` to internal ``Detection``."""
        results: list[Detection] = []
        for det in gemini_dets:
            class_id = self._resolve_label(det.label)
            if class_id is None:
                logger.warning(
                    "Label %r not in class map %s — skipping",
                    det.label,
                    list(self._name_to_id.keys()),
                )
                continue

            bbox = BoundingBox(
                x=_clamp01(det.bbox.x),
                y=_clamp01(det.bbox.y),
                w=_clamp01(det.bbox.w),
                h=_clamp01(det.bbox.h),
            )
            results.append(
                Detection(bbox=bbox, confidence=det.confidence, class_id=class_id)
            )
        return results

    def _parse_text_fallback(self, text: str) -> list[Detection]:
        """Parse raw JSON text when ``response.parsed`` is unavailable."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini JSON response: %s", text[:200])
            return []

        if not isinstance(data, list):
            logger.error("Expected a JSON list, got %s", type(data).__name__)
            return []

        gemini_dets: list[GeminiDetection] = []
        for item in data:
            try:
                gemini_dets.append(GeminiDetection.model_validate(item))
            except Exception:
                logger.debug("Skipping unparseable item: %s", item)

        return self._map_detections(gemini_dets)


def _clamp01(value: float) -> float:
    """Clamp *value* to [0, 1]."""
    return max(0.0, min(1.0, value))
