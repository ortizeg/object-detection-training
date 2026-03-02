"""Tests for Florence-2 inferencer with mocked transformers model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from object_detection_training.inference.florence2_inferencer import (
    Florence2Inferencer,
)
from object_detection_training.schemas.detection import Detection


@pytest.fixture()
def _mock_transformers():
    """Patch transformers model and processor for all tests."""
    with (
        patch(
            "object_detection_training.inference.florence2_inferencer.AutoProcessor"
        ) as mock_proc_cls,
        patch(
            "object_detection_training.inference.florence2_inferencer"
            ".AutoModelForCausalLM"
        ) as mock_model_cls,
        patch(
            "object_detection_training.inference.florence2_inferencer.torch"
        ) as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_processor, mock_model, mock_torch


class TestFlorence2Inferencer:
    """Tests for Florence2Inferencer."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_initialization(self) -> None:
        inferencer = Florence2Inferencer(
            model_name="test-model",
            classes=["player", "ball"],
            caption="detect basketball objects",
            device="cpu",
        )
        assert inferencer.model_name == "test-model"
        assert inferencer.classes == ["player", "ball"]
        assert inferencer.caption == "detect basketball objects"
        assert inferencer._device == "cpu"
        assert inferencer._task == "<OD>"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_initialization_with_custom_task(self) -> None:
        inferencer = Florence2Inferencer(
            classes=["player"],
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            device="cpu",
        )
        assert inferencer._task == "<CAPTION_TO_PHRASE_GROUNDING>"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_default_confidence(self) -> None:
        inferencer = Florence2Inferencer(
            classes=["player"],
            default_confidence=0.5,
            device="cpu",
        )
        assert inferencer.default_confidence == 0.5

    def test_predict_with_od_task(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="tensor_value")
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["generated text"]

        mock_processor.post_process_generation.return_value = {
            "<OD>": {
                "bboxes": [[100.0, 200.0, 300.0, 400.0]],
                "labels": ["player"],
            }
        }

        inferencer = Florence2Inferencer(
            classes=["player", "ball"],
            task="<OD>",
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == 1.0
        assert dets[0].bbox.x == pytest.approx(100.0 / 640)
        assert dets[0].bbox.y == pytest.approx(200.0 / 480)
        assert dets[0].bbox.w == pytest.approx(200.0 / 640)
        assert dets[0].bbox.h == pytest.approx(200.0 / 480)

        # Verify prompt was just the task token
        call_kwargs = mock_processor.call_args
        assert call_kwargs[1]["text"] == "<OD>"

    def test_predict_with_caption_grounding_task(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="tensor_value")
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["generated text"]

        mock_processor.post_process_generation.return_value = {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[100.0, 200.0, 300.0, 400.0]],
                "labels": ["player"],
            }
        }

        inferencer = Florence2Inferencer(
            classes=["player", "ball"],
            caption="detect players",
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == 1.0

        # Verify prompt includes caption
        call_kwargs = mock_processor.call_args
        assert call_kwargs[1]["text"] == "<CAPTION_TO_PHRASE_GROUNDING>detect players"

    def test_predict_unknown_label_skipped(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="tensor_value")
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["text"]

        mock_processor.post_process_generation.return_value = {
            "<OD>": {
                "bboxes": [[10.0, 20.0, 30.0, 40.0]],
                "labels": ["alien"],
            }
        }

        inferencer = Florence2Inferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_handles_exception(self, _mock_transformers) -> None:
        mock_processor, _mock_model, _mock_torch = _mock_transformers

        mock_processor.side_effect = RuntimeError("boom")

        inferencer = Florence2Inferencer(
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
        mock_inputs.__getitem__ = MagicMock(return_value="tensor_value")
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["text"]

        mock_processor.post_process_generation.return_value = {
            "<OD>": {
                "bboxes": [],
                "labels": [],
            }
        }

        inferencer = Florence2Inferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_no_task_key(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="tensor_value")
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["text"]

        # Missing task key in response
        mock_processor.post_process_generation.return_value = {}

        inferencer = Florence2Inferencer(
            classes=["player"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []
