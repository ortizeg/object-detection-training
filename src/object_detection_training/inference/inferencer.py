"""ONNX model inference engine.

Provides :class:`ONNXInferencer` which loads an ONNX model via
``onnxruntime`` and exposes ``predict`` / ``predict_batch`` methods
that combine preprocessing, session execution, and post-processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]
from loguru import logger

from object_detection_training.inference.models import Detection
from object_detection_training.inference.postprocess import BasePostProcessor


class ONNXInferencer:
    """Run inference on images using an ONNX model.

    Args:
        model_path: Path to the ``.onnx`` model file.
        post_processor: Strategy for decoding raw model outputs.
        input_height: Model input height.
        input_width: Model input width.
        batch_size: Number of images per forward pass (currently used
            for documentation; actual batching is handled by
            ``predict_batch``).
        image_mean: Per-channel mean for normalisation (BGR order).
        image_std: Per-channel std for normalisation (BGR order).
    """

    def __init__(
        self,
        model_path: Path | str,
        post_processor: BasePostProcessor,
        input_height: int = 640,
        input_width: int = 640,
        batch_size: int = 1,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self.post_processor = post_processor
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.image_mean = np.array(
            image_mean or [0.485, 0.456, 0.406], dtype=np.float32
        )
        self.image_std = np.array(image_std or [0.229, 0.224, 0.225], dtype=np.float32)

        logger.info(f"Loading ONNX model from {self._model_path}")
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=providers or ort.get_available_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            f"ONNX session ready "
            f"(input={self._input_name}, "
            f"providers={self._session.get_providers()})"
        )

    # ------------------------------------------------------------------
    # Pre-processing (overridable)
    # ------------------------------------------------------------------

    def preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating[Any]]:
        """Resize, normalise, and convert HWC -> CHW float32.

        Override this method for custom pre-processing pipelines.

        Args:
            image: BGR uint8 image (OpenCV default).

        Returns:
            Float32 array of shape ``[1, 3, H, W]``.
        """
        # Resize
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # BGR -> RGB, uint8 -> float32
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0

        # Normalise
        rgb = (rgb - self.image_mean) / self.image_std

        # HWC -> CHW, add batch dim
        chw = np.transpose(rgb, (2, 0, 1))
        batch: npt.NDArray[np.floating[Any]] = np.expand_dims(chw, axis=0).astype(
            np.float32
        )
        return batch

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run full inference pipeline on a single image.

        Args:
            image: BGR uint8 image.
            image_width: Original width (defaults to ``image.shape[1]``).
            image_height: Original height (defaults to ``image.shape[0]``).

        Returns:
            List of :class:`Detection` objects.
        """
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        input_tensor = self.preprocess(image)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        return self.post_processor(outputs, w, h)

    def predict_batch(
        self,
        images: list[npt.NDArray[np.uint8]],
        image_sizes: list[tuple[int, int]] | None = None,
    ) -> list[list[Detection]]:
        """Run inference on multiple images sequentially.

        Args:
            images: List of BGR uint8 images.
            image_sizes: Optional ``(width, height)`` per image.

        Returns:
            List of detection lists, one per image.
        """
        results: list[list[Detection]] = []
        for idx, img in enumerate(images):
            if image_sizes is not None:
                w, h = image_sizes[idx]
            else:
                w, h = int(img.shape[1]), int(img.shape[0])
            results.append(self.predict(img, image_width=w, image_height=h))
        return results
