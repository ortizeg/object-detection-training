"""Tests for OmDet-Turbo inferencer with mocked transformers model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from object_detection_training.inference.omdet_turbo_inferencer import (
    OmDetTurboInferencer,
)
from object_detection_training.schemas.detection import Detection


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
