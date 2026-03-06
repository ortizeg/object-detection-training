"""Tests for Grounding DINO inferencer with mocked transformers model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from object_detection_training.inference.grounding_dino_inferencer import (
    GroundingDINOInferencer,
)
from object_detection_training.schemas.detection import BoundingBox, Detection


@pytest.fixture()
def _mock_transformers():
    """Patch transformers model and processor for all tests."""
    with (
        patch(
            "object_detection_training.inference.grounding_dino_inferencer"
            ".AutoProcessor"
        ) as mock_proc_cls,
        patch(
            "object_detection_training.inference.grounding_dino_inferencer"
            ".AutoModelForZeroShotObjectDetection"
        ) as mock_model_cls,
        patch(
            "object_detection_training.inference.grounding_dino_inferencer.torch"
        ) as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.Tensor = torch.Tensor

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_processor, mock_model, mock_torch


class TestGroundingDINOInferencer:
    """Tests for GroundingDINOInferencer."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_initialization(self) -> None:
        inferencer = GroundingDINOInferencer(
            model_name="test-model",
            classes=["player", "ball"],
            device="cpu",
        )
        assert inferencer.model_name == "test-model"
        assert inferencer.classes == ["player", "ball"]
        assert inferencer._device == "cpu"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_text_prompt_format(self) -> None:
        inferencer = GroundingDINOInferencer(
            classes=["player", "ball", "referee"],
            device="cpu",
        )
        assert inferencer._text_prompt == "player . ball . referee ."

    @pytest.mark.usefixtures("_mock_transformers")
    def test_name_to_id_mapping(self) -> None:
        inferencer = GroundingDINOInferencer(
            classes=["player", "ball", "referee"],
            device="cpu",
        )
        assert inferencer._name_to_id == {
            "player": 0,
            "ball": 1,
            "referee": 2,
        }

    def test_predict_with_results(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        # Setup processor inputs
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        # Setup model output
        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        # Setup post-processing results
        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.85]),
                "text": ["player"],
            }
        ]

        inferencer = GroundingDINOInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.85)
        assert dets[0].bbox.x == pytest.approx(100.0 / 640)
        assert dets[0].bbox.y == pytest.approx(200.0 / 480)

        # Verify input_ids was passed to post-processing
        call_kwargs = mock_processor.post_process_grounded_object_detection.call_args
        assert "input_ids" in call_kwargs.kwargs

    def test_predict_unknown_label_skipped(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.8]),
                "text": ["alien"],
            }
        ]

        inferencer = GroundingDINOInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_with_text_labels_key(self, _mock_transformers) -> None:
        """Test that results with 'text_labels' key (transformers >=4.51) work."""
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.75]),
                "text_labels": ["ball"],
            }
        ]

        inferencer = GroundingDINOInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 1
        assert dets[0].confidence == pytest.approx(0.75)

    def test_predict_with_integer_labels(self, _mock_transformers) -> None:
        """Test that integer label IDs (transformers >=4.51 'labels' key) work."""
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.75]),
                "labels": [1],
            }
        ]

        inferencer = GroundingDINOInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 1
        assert dets[0].confidence == pytest.approx(0.75)

    def test_predict_handles_exception(self, _mock_transformers) -> None:
        mock_processor, _mock_model, _mock_torch = _mock_transformers

        mock_processor.side_effect = RuntimeError("boom")

        inferencer = GroundingDINOInferencer(
            classes=["player"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []


class TestResolveLabelConcatenated:
    """Tests for _resolve_label handling concatenated multi-class labels."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_exact_match(self) -> None:
        inferencer = GroundingDINOInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        assert inferencer._resolve_label("person") == 0
        assert inferencer._resolve_label("sports ball") == 1
        assert inferencer._resolve_label("jersey number") == 4

    @pytest.mark.usefixtures("_mock_transformers")
    def test_concatenated_label_picks_first_match(self) -> None:
        """'person referee jersey number' should resolve to person (first)."""
        inferencer = GroundingDINOInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        assert inferencer._resolve_label("person referee jersey number") == 0
        assert inferencer._resolve_label("person referee basketball hoop") == 0

    @pytest.mark.usefixtures("_mock_transformers")
    def test_concatenated_label_without_person(self) -> None:
        """'basketball hoop jersey number' should resolve to basketball hoop."""
        inferencer = GroundingDINOInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        # basketball hoop appears at position 0, jersey number at position 16
        assert inferencer._resolve_label("basketball hoop jersey number") == 3

    @pytest.mark.usefixtures("_mock_transformers")
    def test_no_match_returns_none(self) -> None:
        inferencer = GroundingDINOInferencer(
            classes=["person", "ball"],
            device="cpu",
        )
        assert inferencer._resolve_label("alien spaceship") is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_case_insensitive(self) -> None:
        inferencer = GroundingDINOInferencer(
            classes=["person", "ball"],
            device="cpu",
        )
        assert inferencer._resolve_label("Person") == 0
        assert inferencer._resolve_label("  BALL  ") == 1


class TestResolveLabelSizeCheck:
    """Tests for size-based sanity check on small-object classes."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_small_box_keeps_jersey_number(self) -> None:
        """A small box labeled 'jersey number' is kept."""
        inferencer = GroundingDINOInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        # 0.5% of image area — under the 1% threshold
        assert inferencer._resolve_label("jersey number", box_area_fraction=0.005) == 4

    @pytest.mark.usefixtures("_mock_transformers")
    def test_large_box_rejects_jersey_number(self) -> None:
        """A large box labeled only 'jersey number' is rejected (returns None)."""
        inferencer = GroundingDINOInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        # 2% of image area — over the 1% threshold
        assert (
            inferencer._resolve_label("jersey number", box_area_fraction=0.02) is None
        )

    @pytest.mark.usefixtures("_mock_transformers")
    def test_large_box_concatenated_falls_through_to_person(self) -> None:
        """A large box with 'person referee jersey number' skips jersey number."""
        inferencer = GroundingDINOInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        # Large box: jersey number is skipped, falls through to person
        result = inferencer._resolve_label(
            "jersey number person", box_area_fraction=0.02
        )
        assert result == 0  # person

    @pytest.mark.usefixtures("_mock_transformers")
    def test_person_not_affected_by_size_check(self) -> None:
        """Person class is never rejected by size check."""
        inferencer = GroundingDINOInferencer(
            classes=["person", "ball"],
            device="cpu",
        )
        assert inferencer._resolve_label("person", box_area_fraction=0.5) == 0


class TestNMS:
    """Tests for per-class greedy NMS."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_removes_overlapping_same_class(self) -> None:
        """Overlapping boxes of same class: lower confidence is suppressed."""
        inferencer = GroundingDINOInferencer(
            classes=["person"],
            nms_iou_threshold=0.5,
            device="cpu",
        )
        dets = [
            Detection(
                bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
                confidence=0.9,
                class_id=0,
            ),
            Detection(
                bbox=BoundingBox(x=0.12, y=0.12, w=0.2, h=0.3),
                confidence=0.7,
                class_id=0,
            ),
        ]
        result = inferencer._nms(dets)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_keeps_different_classes(self) -> None:
        """Overlapping boxes of different classes are both kept."""
        inferencer = GroundingDINOInferencer(
            classes=["person", "ball"],
            nms_iou_threshold=0.5,
            device="cpu",
        )
        dets = [
            Detection(
                bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
                confidence=0.9,
                class_id=0,
            ),
            Detection(
                bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
                confidence=0.7,
                class_id=1,
            ),
        ]
        result = inferencer._nms(dets)
        assert len(result) == 2

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_keeps_non_overlapping(self) -> None:
        """Non-overlapping boxes of same class are both kept."""
        inferencer = GroundingDINOInferencer(
            classes=["person"],
            nms_iou_threshold=0.5,
            device="cpu",
        )
        dets = [
            Detection(
                bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
                confidence=0.9,
                class_id=0,
            ),
            Detection(
                bbox=BoundingBox(x=0.5, y=0.5, w=0.1, h=0.1),
                confidence=0.7,
                class_id=0,
            ),
        ]
        result = inferencer._nms(dets)
        assert len(result) == 2

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_empty_list(self) -> None:
        inferencer = GroundingDINOInferencer(
            classes=["person"],
            device="cpu",
        )
        assert inferencer._nms([]) == []

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_single_detection(self) -> None:
        inferencer = GroundingDINOInferencer(
            classes=["person"],
            device="cpu",
        )
        det = Detection(
            bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
            confidence=0.9,
            class_id=0,
        )
        result = inferencer._nms([det])
        assert len(result) == 1
        assert result[0] is det
