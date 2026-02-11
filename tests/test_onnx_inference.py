"""Tests for the ONNX inference package.

Covers Pydantic models, ImageLoader, post-processors, annotation writer,
ONNXInferencer, and ONNXInferenceTask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from object_detection_training.inference.annotation import (
    DetectionAnnotationWriter,
)
from object_detection_training.inference.image import ImageLoader
from object_detection_training.inference.models import (
    BoundingBox,
    Detection,
    DetectionAnnotation,
)
from object_detection_training.inference.postprocess import (
    RFDETRPostProcessor,
    YOLOXPostProcessor,
)

# =========================================================================
# Pydantic model tests
# =========================================================================


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_creation(self) -> None:
        bbox = BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4)
        assert bbox.x == pytest.approx(0.1)
        assert bbox.w == pytest.approx(0.3)

    def test_frozen(self) -> None:
        bbox = BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4)
        with pytest.raises(Exception):  # noqa: B017
            bbox.x = 0.5  # type: ignore[misc]


class TestDetection:
    """Tests for Detection model."""

    def test_creation(self) -> None:
        det = Detection(
            bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
            confidence=0.95,
            label="person",
        )
        assert det.confidence == pytest.approx(0.95)
        assert det.label == "person"

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            Detection(
                bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
                confidence=1.5,
                label="x",
            )

    def test_frozen(self) -> None:
        det = Detection(
            bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
            confidence=0.5,
            label="a",
        )
        with pytest.raises(Exception):  # noqa: B017
            det.label = "b"  # type: ignore[misc]


class TestDetectionAnnotation:
    """Tests for DetectionAnnotation model."""

    def test_creation(self) -> None:
        ann = DetectionAnnotation(
            image_filename="test.jpg",
            image_width=640,
            image_height=480,
            label_map={0: "person"},
        )
        assert ann.image_filename == "test.jpg"
        assert ann.detections == []

    def test_with_detections(self) -> None:
        det = Detection(
            bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
            confidence=0.9,
            label="ball",
        )
        ann = DetectionAnnotation(
            image_filename="frame.png",
            image_width=1920,
            image_height=1080,
            label_map={0: "ball"},
            detections=[det],
        )
        assert len(ann.detections) == 1

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            DetectionAnnotation(
                image_filename="x.jpg",
                image_width=0,
                image_height=480,
                label_map={},
            )


# =========================================================================
# ImageLoader tests
# =========================================================================


class TestImageLoader:
    """Tests for ImageLoader."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ImageLoader(tmp_path / "nonexistent.jpg")

    @patch("object_detection_training.inference.image.cv2")
    def test_read_returns_image(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        img_path = tmp_path / "test.jpg"
        img_path.touch()

        fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = fake_img

        loader = ImageLoader(img_path)
        result = loader.read()

        mock_cv2.imread.assert_called_once_with(str(img_path))
        assert result.shape == (480, 640, 3)

    @patch("object_detection_training.inference.image.cv2")
    def test_properties(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        img_path = tmp_path / "frame.png"
        img_path.touch()

        mock_cv2.imread.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)

        loader = ImageLoader(img_path)
        assert loader.width == 1280
        assert loader.height == 720
        assert loader.filename == "frame.png"

    @patch("object_detection_training.inference.image.cv2")
    def test_read_failure(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        img_path = tmp_path / "bad.jpg"
        img_path.touch()
        mock_cv2.imread.return_value = None

        loader = ImageLoader(img_path)
        with pytest.raises(OSError, match="failed to read"):
            loader.read()


# =========================================================================
# Post-processor tests
# =========================================================================

LABEL_MAP = {0: "person", 1: "ball"}


class TestYOLOXPostProcessor:
    """Tests for YOLOXPostProcessor."""

    def _make_predictions(
        self,
        num_anchors: int = 5,
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Create fake YOLOX output: [1, num_anchors, 7] (5+2 classes)."""
        pred = np.zeros((1, num_anchors, 7), dtype=np.float32)
        # anchor 0: high confidence person at centre
        pred[0, 0, :] = [320, 240, 50, 100, 0.9, 0.95, 0.05]
        # anchor 1: low confidence (should be filtered)
        pred[0, 1, :] = [100, 100, 30, 30, 0.1, 0.5, 0.5]
        return [pred]

    def test_basic_decoding(self) -> None:
        pp = YOLOXPostProcessor(
            LABEL_MAP, confidence_threshold=0.2, nms_iou_threshold=0.5
        )
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)

        assert len(dets) >= 1
        assert all(isinstance(d, Detection) for d in dets)
        # First detection should be "person"
        assert dets[0].label == "person"

    def test_threshold_filters(self) -> None:
        pp = YOLOXPostProcessor(LABEL_MAP, confidence_threshold=0.99)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        # Very high threshold should filter everything
        assert len(dets) == 0

    def test_empty_input(self) -> None:
        pp = YOLOXPostProcessor(LABEL_MAP)
        empty = [np.zeros((1, 0, 7), dtype=np.float32)]
        assert pp(empty, 640, 480) == []

    def test_normalised_coordinates(self) -> None:
        pp = YOLOXPostProcessor(LABEL_MAP, confidence_threshold=0.2)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        for d in dets:
            assert 0.0 <= d.bbox.x <= 1.0
            assert 0.0 <= d.bbox.y <= 1.0
            assert 0.0 <= d.bbox.w <= 1.0
            assert 0.0 <= d.bbox.h <= 1.0


class TestRFDETRPostProcessor:
    """Tests for RFDETRPostProcessor."""

    def _make_predictions(
        self,
        num_queries: int = 3,
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Create fake RFDETR outputs: logits + boxes."""
        # logits: [1, num_queries, 2] (2 classes)
        logits = np.full((1, num_queries, 2), -10.0, dtype=np.float32)
        # Make query 0 a confident person
        logits[0, 0, 0] = 5.0  # sigmoid(5) ≈ 0.993

        # boxes: [1, num_queries, 4] in normalised cxcywh
        boxes = np.zeros((1, num_queries, 4), dtype=np.float32)
        boxes[0, 0, :] = [0.5, 0.5, 0.1, 0.2]

        return [logits, boxes]

    def test_basic_decoding(self) -> None:
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)

        assert len(dets) == 1
        assert dets[0].label == "person"
        assert dets[0].confidence > 0.9

    def test_threshold_filters(self) -> None:
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.999)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        # sigmoid(5) ≈ 0.993 < 0.999
        assert len(dets) == 0

    def test_box_coordinates_normalised(self) -> None:
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        for d in dets:
            assert 0.0 <= d.bbox.x <= 1.0
            assert 0.0 <= d.bbox.y <= 1.0

    def test_swapped_outputs(self) -> None:
        """Test robustness when outputs are [boxes, logits] (swapped order)."""
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        # Normal order: [logits, boxes]
        logits, boxes = self._make_predictions()
        # Swapped order
        swapped = [boxes, logits]
        dets = pp(swapped, image_width=640, image_height=480)

        assert len(dets) == 1
        assert dets[0].label == "person"


# =========================================================================
# DetectionAnnotationWriter tests
# =========================================================================


class TestDetectionAnnotationWriter:
    """Tests for DetectionAnnotationWriter."""

    def test_write_creates_json(self, tmp_path: Path) -> None:
        writer = DetectionAnnotationWriter(tmp_path / "annotations")
        ann = DetectionAnnotation(
            image_filename="frame_001.jpg",
            image_width=640,
            image_height=480,
            label_map={0: "person"},
            detections=[
                Detection(
                    bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
                    confidence=0.9,
                    label="person",
                )
            ],
        )
        out_path = writer.write(ann)

        assert out_path.exists()
        assert out_path.suffix == ".json"
        assert out_path.stem == "frame_001"

        data = json.loads(out_path.read_text())
        assert data["image_filename"] == "frame_001.jpg"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["bbox"]["x"] == pytest.approx(0.1)

    def test_write_creates_output_dir(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "output"
        writer = DetectionAnnotationWriter(out_dir)
        ann = DetectionAnnotation(
            image_filename="test.jpg",
            image_width=100,
            image_height=100,
            label_map={},
        )
        writer.write(ann)
        assert out_dir.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Write and read back, verify data integrity."""
        writer = DetectionAnnotationWriter(tmp_path)
        original = DetectionAnnotation(
            image_filename="img.png",
            image_width=1920,
            image_height=1080,
            label_map={0: "person", 1: "ball"},
            detections=[
                Detection(
                    bbox=BoundingBox(x=0.5, y=0.5, w=0.1, h=0.1),
                    confidence=0.75,
                    label="ball",
                ),
            ],
        )
        out_path = writer.write(original)
        loaded = DetectionAnnotation.model_validate_json(out_path.read_text())
        assert loaded == original


# =========================================================================
# ONNXInferencer tests (mocked onnxruntime)
# =========================================================================


class TestONNXInferencer:
    """Tests for ONNXInferencer with mocked ONNX session."""

    @patch("object_detection_training.inference.inferencer.ort")
    def test_predict_pipeline(self, mock_ort: MagicMock) -> None:
        """predict() calls preprocess -> session.run -> postprocess."""
        # Setup mock session
        session_mock = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "images"
        session_mock.get_inputs.return_value = [input_mock]
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]

        # Return RFDETR-style outputs
        logits = np.full((1, 2, 2), -10.0, dtype=np.float32)
        logits[0, 0, 0] = 5.0
        boxes = np.zeros((1, 2, 4), dtype=np.float32)
        boxes[0, 0, :] = [0.5, 0.5, 0.1, 0.2]
        session_mock.run.return_value = [logits, boxes]

        mock_ort.InferenceSession.return_value = session_mock
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        from object_detection_training.inference.inferencer import (
            ONNXInferencer,
        )

        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        inferencer = ONNXInferencer(model_path="model.onnx", post_processor=pp)

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image)

        session_mock.run.assert_called_once()
        assert len(dets) == 1
        assert dets[0].label == "person"

    @patch("object_detection_training.inference.inferencer.ort")
    def test_predict_batch(self, mock_ort: MagicMock) -> None:
        """predict_batch processes multiple images."""
        session_mock = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "images"
        session_mock.get_inputs.return_value = [input_mock]
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]
        session_mock.run.return_value = [
            np.full((1, 1, 2), -10.0, dtype=np.float32),
            np.zeros((1, 1, 4), dtype=np.float32),
        ]
        mock_ort.InferenceSession.return_value = session_mock
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        from object_detection_training.inference.inferencer import (
            ONNXInferencer,
        )

        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        inferencer = ONNXInferencer(model_path="model.onnx", post_processor=pp)

        images = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        results = inferencer.predict_batch(images)
        assert len(results) == 2


# =========================================================================
# ONNXInferenceTask tests
# =========================================================================


class TestONNXInferenceTask:
    """Tests for ONNXInferenceTask."""

    def test_init_defaults(self, tmp_path: Path) -> None:
        from object_detection_training.tasks import ONNXInferenceTask

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        task = ONNXInferenceTask(
            model_path=model_path,
            image_dir=img_dir,
            label_map={0: "person"},
        )
        assert task.name == "inference_onnx"
        assert task.post_processor_type == "rfdetr"
        assert task.confidence_threshold == pytest.approx(0.25)

    def test_invalid_post_processor_type(self, tmp_path: Path) -> None:
        from object_detection_training.tasks import ONNXInferenceTask

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        task = ONNXInferenceTask(
            model_path=model_path,
            image_dir=img_dir,
            label_map={0: "person"},
            post_processor_type="unknown",
        )
        with pytest.raises(ValueError, match="Unknown post_processor_type"):
            task.run()
