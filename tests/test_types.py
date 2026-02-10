"""Tests for shared type definitions."""

from __future__ import annotations

import torch

from object_detection_training.types import (
    CurveData,
    DetectionBatch,
    DetectionPrediction,
    DetectionTarget,
    EMAState,
    ModelStats,
    ONNXExportState,
    VisualizationSample,
)


class TestDetectionTarget:
    """Tests for DetectionTarget TypedDict."""

    def test_construction_with_all_fields(self) -> None:
        target: DetectionTarget = {
            "boxes": torch.zeros(3, 4),
            "labels": torch.zeros(3, dtype=torch.long),
            "image_id": torch.tensor([1]),
            "area": torch.zeros(3),
            "iscrowd": torch.zeros(3, dtype=torch.long),
            "orig_size": torch.tensor([480, 640]),
            "size": torch.tensor([480, 640]),
        }
        assert target["boxes"].shape == (3, 4)
        assert target["labels"].shape == (3,)

    def test_construction_partial(self) -> None:
        """DetectionTarget supports total=False (optional fields)."""
        target: DetectionTarget = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        assert "image_id" not in target


class TestDetectionPrediction:
    """Tests for DetectionPrediction TypedDict."""

    def test_construction(self) -> None:
        pred: DetectionPrediction = {
            "boxes": torch.rand(5, 4),
            "scores": torch.rand(5),
            "labels": torch.randint(0, 3, (5,)),
        }
        assert pred["boxes"].shape == (5, 4)
        assert pred["scores"].shape == (5,)
        assert pred["labels"].shape == (5,)


class TestDetectionBatch:
    """Tests for DetectionBatch type alias."""

    def test_batch_is_tuple(self) -> None:
        images = torch.rand(2, 3, 640, 640)
        targets: list[DetectionTarget] = [
            {"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.long)},
            {"boxes": torch.zeros(2, 4), "labels": torch.zeros(2, dtype=torch.long)},
        ]
        batch: DetectionBatch = (images, targets)
        assert len(batch) == 2
        assert batch[0].shape[0] == 2


class TestVisualizationSample:
    """Tests for VisualizationSample TypedDict."""

    def test_construction(self) -> None:
        sample: VisualizationSample = {
            "image": torch.rand(3, 480, 640),
            "target": {
                "boxes": torch.zeros(1, 4),
                "labels": torch.zeros(1, dtype=torch.long),
            },
            "image_id": 42,
        }
        assert sample["image_id"] == 42


class TestModelStats:
    """Tests for ModelStats TypedDict."""

    def test_construction(self) -> None:
        stats: ModelStats = {
            "total_params": 1000000,
            "trainable_params": 900000,
            "model_size_mb": 3.5,
        }
        assert stats["total_params"] == 1000000


class TestEMAState:
    """Tests for EMAState TypedDict."""

    def test_construction(self) -> None:
        state: EMAState = {
            "ema_state_dict": {"weight": torch.zeros(5)},
            "step_count": 100,
            "decay": 0.999,
        }
        assert state["step_count"] == 100


class TestONNXExportState:
    """Tests for ONNXExportState TypedDict."""

    def test_construction(self) -> None:
        state: ONNXExportState = {"exported_checkpoints": ["model.onnx"]}
        assert len(state["exported_checkpoints"]) == 1


class TestCurveData:
    """Tests for CurveData TypedDict."""

    def test_construction(self) -> None:
        import numpy as np

        data: CurveData = {
            "precision": np.array([1.0, 0.8, 0.6]),
            "recall": np.array([0.3, 0.6, 1.0]),
            "scores": np.array([0.9, 0.7, 0.5]),
            "f1": np.array([0.46, 0.69, 0.75]),
        }
        assert len(data["precision"]) == 3
