"""Base class for all inferencers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from object_detection_training.schemas.detection import Detection


class BaseInferencer(ABC):
    """Abstract base class for object detection inferencers."""

    @abstractmethod
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single image.

        Args:
            image: BGR uint8 image.
            image_width: Original width (optional).
            image_height: Original height (optional).

        Returns:
            List of detections.
        """
        pass
