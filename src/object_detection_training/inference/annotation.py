"""Write DetectionAnnotation objects to JSON files.

Each image produces one JSON file that is fully self-contained
(includes the label map, image dimensions, and all detections).
"""

from __future__ import annotations

from pathlib import Path

import orjson
from loguru import logger

from object_detection_training.inference.models import DetectionAnnotation


class DetectionAnnotationWriter:
    """Serialize :class:`DetectionAnnotation` instances to JSON on disk.

    Args:
        output_dir: Directory where annotation files are written.
    """

    def __init__(self, output_dir: Path | str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, annotation: DetectionAnnotation) -> Path:
        """Write a single annotation file.

        The output filename is derived from the image filename with a
        ``.json`` extension.

        Args:
            annotation: Detection annotation to persist.

        Returns:
            Path to the written JSON file.
        """

        stem = Path(annotation.filename).stem
        out_path = self._output_dir / f"{stem}.json"

        # orjson produces bytes; use model_dump for clean serialization
        data = annotation.model_dump()
        out_path.write_bytes(
            orjson.dumps(
                data,
                option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
            )
        )

        logger.debug(
            f"Wrote annotation for {annotation.filename} "
            f"({len(annotation.annotations)} detections) -> {out_path}"
        )
        return out_path
