"""Tests for visualization utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from object_detection_training.schemas.annotation import (
    AnnotationInfo,
    DetectionAnnotation,
)
from object_detection_training.schemas.detection import BoundingBox, Detection
from object_detection_training.utils.visualization import (
    draw_detections,
    save_annotated_image,
)


class TestVisualization:
    """Tests for visualization utilities."""

    def test_draw_detections_empty(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotation = DetectionAnnotation(
            filename="test.jpg",
            categories={0: "person"},
            info=AnnotationInfo(annotations_source="test"),
            annotations=[],
        )
        result = draw_detections(image, annotation)
        np.testing.assert_array_equal(result, image)

    @patch("object_detection_training.utils.visualization.sv.BoxAnnotator")
    @patch("object_detection_training.utils.visualization.sv.LabelAnnotator")
    def test_draw_detections_with_boxes(
        self, mock_label_annotator_cls: MagicMock, mock_box_annotator_cls: MagicMock
    ) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotation = DetectionAnnotation(
            filename="test.jpg",
            categories={0: "person"},
            info=AnnotationInfo(annotations_source="test"),
            annotations=[
                Detection(
                    bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
                    confidence=0.9,
                    class_id=0,
                )
            ],
        )

        mock_box_annotator = MagicMock()
        mock_box_annotator.annotate.return_value = image.copy()
        mock_box_annotator_cls.return_value = mock_box_annotator

        mock_label_annotator = MagicMock()
        mock_label_annotator.annotate.return_value = image.copy()
        mock_label_annotator_cls.return_value = mock_label_annotator

        result = draw_detections(image, annotation)

        assert mock_box_annotator.annotate.called
        assert mock_label_annotator.annotate.called
        assert result.shape == image.shape

    @patch("object_detection_training.utils.visualization.draw_detections")
    @patch("object_detection_training.utils.visualization.cv2.imwrite")
    def test_save_annotated_image(
        self, mock_imwrite: MagicMock, mock_draw: MagicMock, tmp_path: Path
    ) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotation = MagicMock(spec=DetectionAnnotation)
        output_path = tmp_path / "output.jpg"

        mock_draw.return_value = image

        save_annotated_image(image, annotation, output_path)

        mock_draw.assert_called_once_with(image, annotation)
        mock_imwrite.assert_called_once_with(str(output_path), image)
