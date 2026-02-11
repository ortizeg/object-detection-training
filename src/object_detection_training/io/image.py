"""Image loading abstraction for inference.

Wraps OpenCV image I/O behind a clean interface so that callers
do not depend on ``cv2`` directly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger


class ImageLoader:
    """Load and inspect an image file via OpenCV.

    Attributes:
        path: Absolute or relative path to the image file.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            msg = f"Image file not found: {self._path}"
            raise FileNotFoundError(msg)

        # Cache dimensions on construction so they are always available.
        self._image: npt.NDArray[np.uint8] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> npt.NDArray[np.uint8]:
        """Read the image as a BGR ``uint8`` numpy array (OpenCV default).

        The result is cached so subsequent calls are free.
        """
        if self._image is None:
            img = cv2.imread(str(self._path))
            if img is None:
                msg = f"OpenCV failed to read image: {self._path}"
                raise OSError(msg)
            self._image = np.asarray(img, dtype=np.uint8)
            logger.debug(f"Loaded image {self.filename} ({self.width}x{self.height})")
        return self._image

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.read().shape[1])

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.read().shape[0])

    @property
    def filename(self) -> str:
        """Basename of the image file."""
        return self._path.name
