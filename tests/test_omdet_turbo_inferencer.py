"""Tests for OmDet-Turbo inferencer with mocked transformers model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from object_detection_training.inference.omdet_turbo_inferencer import (
    OmDetTurboInferencer,
)
from object_detection_training.schemas.detection import BoundingBox, Detection


@pytest.fixture()
def _mock_transformers():
    """Patch transformers model and processor for all tests."""
    with (
        patch(
            "object_detection_training.inference.omdet_turbo_inferencer.AutoProcessor"
        ) as mock_proc_cls,
        patch(
            "object_detection_training.inference.omdet_turbo_inferencer"
            ".AutoModelForZeroShotObjectDetection"
        ) as mock_model_cls,
        patch(
            "object_detection_training.inference.omdet_turbo_inferencer.torch"
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
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_processor, mock_model, mock_torch


class TestOmDetTurboInferencer:
    """Tests for OmDetTurboInferencer."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_initialization(self) -> None:
        inferencer = OmDetTurboInferencer(
            model_name="test-model",
            classes=["player", "ball"],
            device="cpu",
        )
        assert inferencer.model_name == "test-model"
        assert inferencer.classes == ["player", "ball"]
        assert inferencer._device == "cpu"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_name_to_id_mapping(self) -> None:
        inferencer = OmDetTurboInferencer(
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
        mock_processor.return_value = mock_inputs

        # Setup model output
        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        # Setup post-processing results
        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.9]),
                "text": ["player"],
            }
        ]

        inferencer = OmDetTurboInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.9)
        assert dets[0].bbox.x == pytest.approx(100.0 / 640)
        assert dets[0].bbox.y == pytest.approx(200.0 / 480)
        assert dets[0].bbox.w == pytest.approx(200.0 / 640)
        assert dets[0].bbox.h == pytest.approx(200.0 / 480)

    def test_predict_unknown_label_skipped(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
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

        inferencer = OmDetTurboInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_handles_exception(self, _mock_transformers) -> None:
        mock_processor, _mock_model, _mock_torch = _mock_transformers

        mock_processor.side_effect = RuntimeError("boom")

        inferencer = OmDetTurboInferencer(
            classes=["player"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_empty_results(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([]).reshape(0, 4),
                "scores": torch.tensor([]),
                "text": [],
            }
        ]

        inferencer = OmDetTurboInferencer(
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

        inferencer = OmDetTurboInferencer(
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

        inferencer = OmDetTurboInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 1
        assert dets[0].confidence == pytest.approx(0.75)


class TestMaterializeMetaBuffers:
    """Tests for _materialize_meta_buffers static method."""

    def test_replaces_meta_buffers(self) -> None:
        """Meta-device buffers are replaced with real CPU tensors."""
        model = torch.nn.Module()
        meta_buf = torch.empty(3, 4, device="meta")
        model.register_buffer("attn_mask", meta_buf)

        OmDetTurboInferencer._materialize_meta_buffers(model)

        buf = model.attn_mask
        assert buf.device.type == "cpu"
        assert buf.shape == (3, 4)

    def test_leaves_cpu_buffers_unchanged(self) -> None:
        """CPU buffers are not modified."""
        model = torch.nn.Module()
        cpu_buf = torch.ones(2, 2)
        model.register_buffer("mask", cpu_buf)

        OmDetTurboInferencer._materialize_meta_buffers(model)

        assert torch.equal(model.mask, cpu_buf)

    def test_handles_no_buffers(self) -> None:
        """Model with no buffers does not raise."""
        model = torch.nn.Module()
        OmDetTurboInferencer._materialize_meta_buffers(model)


class TestNMS:
    """Tests for per-class greedy NMS."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_removes_overlapping_same_class(self) -> None:
        """Overlapping boxes of same class: lower confidence is suppressed."""
        inferencer = OmDetTurboInferencer(
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
        inferencer = OmDetTurboInferencer(
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
        inferencer = OmDetTurboInferencer(
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
        inferencer = OmDetTurboInferencer(
            classes=["person"],
            device="cpu",
        )
        assert inferencer._nms([]) == []

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_single_detection(self) -> None:
        inferencer = OmDetTurboInferencer(
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
