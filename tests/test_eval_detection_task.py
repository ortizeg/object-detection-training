"""Tests for the EvalDetectionTask."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from object_detection_training.schemas.detection import BoundingBox, Detection
from object_detection_training.tasks.eval_detection_task import (
    EvalDetectionTask,
    _compute_pr_curve,
    _compute_prf1_at_threshold,
    _detections_to_sv,
    _find_best_threshold,
    _load_coco_gt,
)


def _make_coco_json(path: Path, num_images: int = 2) -> None:
    """Create a minimal COCO annotations file."""
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "player"},
            {"id": 2, "name": "ball"},
            {"id": 3, "name": "referee"},
            {"id": 4, "name": "player-in-possession"},
        ],
    }

    ann_id = 1
    for i in range(num_images):
        img_name = f"img_{i:03d}.jpg"
        coco["images"].append(
            {"id": i + 1, "file_name": img_name, "width": 640, "height": 480}
        )
        # Add a player annotation
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 300],
            }
        )
        ann_id += 1
        # Add a player-in-possession annotation (should merge to player)
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 4,
                "bbox": [300, 100, 150, 250],
            }
        )
        ann_id += 1
        # Add a ball annotation
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 2,
                "bbox": [50, 50, 30, 30],
            }
        )
        ann_id += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f)


class TestLoadCocoGt:
    """Tests for _load_coco_gt."""

    def test_basic_loading(self, tmp_path: Path) -> None:
        coco_path = tmp_path / "_annotations.coco.json"
        _make_coco_json(coco_path, num_images=2)

        gt = _load_coco_gt(coco_path)
        assert len(gt) == 2
        assert "img_000.jpg" in gt

        # Both player and player-in-possession should be class_id=0
        dets = gt["img_000.jpg"]
        assert len(dets) == 3  # 2 players + 1 ball
        player_ids = dets.class_id[dets.class_id == 0]
        assert len(player_ids) == 2  # Both mapped to "player"

    def test_empty_images(self, tmp_path: Path) -> None:
        coco_path = tmp_path / "_annotations.coco.json"
        coco = {
            "images": [
                {"id": 1, "file_name": "empty.jpg", "width": 640, "height": 480}
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "player"}],
        }
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        gt = _load_coco_gt(coco_path)
        assert len(gt["empty.jpg"]) == 0


class TestDetectionsToSv:
    """Tests for _detections_to_sv."""

    def test_empty_list(self) -> None:
        sv_dets = _detections_to_sv([], 640, 480)
        assert len(sv_dets) == 0

    def test_conversion(self) -> None:
        dets = [
            Detection(
                bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
                confidence=0.9,
                class_id=0,
            )
        ]
        sv_dets = _detections_to_sv(dets, 640, 480)
        assert len(sv_dets) == 1
        # x1 = 0.1 * 640 = 64
        assert sv_dets.xyxy[0][0] == pytest.approx(64.0)
        # y1 = 0.2 * 480 = 96
        assert sv_dets.xyxy[0][1] == pytest.approx(96.0)
        assert sv_dets.class_id is not None
        assert sv_dets.class_id[0] == 0


class TestComputePrf1:
    """Tests for _compute_prf1_at_threshold."""

    def test_perfect_detection(self) -> None:
        import supervision as sv

        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.9], dtype=np.float32),
            )
        }
        metrics = _compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_no_predictions(self) -> None:
        import supervision as sv

        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred: dict[str, sv.Detections] = {}
        metrics = _compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_threshold_filters(self) -> None:
        import supervision as sv

        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.3], dtype=np.float32),
            )
        }
        # Threshold below confidence -> detection kept
        metrics = _compute_prf1_at_threshold(gt, pred, threshold=0.2)
        assert metrics["recall"] == pytest.approx(1.0)

        # Threshold above confidence -> detection filtered
        metrics = _compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["recall"] == pytest.approx(0.0)


class TestFindBestThreshold:
    """Tests for _find_best_threshold."""

    def test_finds_optimal(self) -> None:
        import supervision as sv

        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.8], dtype=np.float32),
            )
        }
        threshold, metrics = _find_best_threshold(gt, pred, steps=10)
        # Best threshold should be <= 0.8 (to capture the detection)
        assert threshold <= 0.8
        assert metrics["f1"] > 0


class TestComputePrCurve:
    """Tests for _compute_pr_curve."""

    def test_returns_correct_length(self) -> None:
        import supervision as sv

        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.8], dtype=np.float32),
            )
        }
        pr = _compute_pr_curve(gt, pred, steps=5)
        assert len(pr["precisions"]) == 6  # 0 to 5 inclusive
        assert len(pr["recalls"]) == 6


class TestEvalDetectionTask:
    """Tests for EvalDetectionTask."""

    def test_init(self, tmp_path: Path) -> None:
        task = EvalDetectionTask(
            val_dir=tmp_path / "val",
            test_dir=tmp_path / "test",
            run_gemini=False,
            run_smolvlm2=False,
            run_rfdetr=False,
        )
        assert task.name == "eval_detection"

    @patch("object_detection_training.tasks.eval_detection_task.ImageLoader")
    def test_run_rfdetr_only(self, mock_loader_cls: MagicMock, tmp_path: Path) -> None:
        """Run eval with only RF-DETR (mocked ONNX inferencer)."""
        # Setup directories and COCO json
        val_dir = tmp_path / "val"
        test_dir = tmp_path / "test"
        val_dir.mkdir()
        test_dir.mkdir()
        _make_coco_json(val_dir / "_annotations.coco.json", num_images=2)
        _make_coco_json(test_dir / "_annotations.coco.json", num_images=2)

        # Create dummy images
        for d in [val_dir, test_dir]:
            for i in range(2):
                (d / f"img_{i:03d}.jpg").touch()

        # Mock image loader
        mock_loader = MagicMock()
        mock_loader.width = 640
        mock_loader.height = 480
        mock_loader.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_loader_cls.return_value = mock_loader

        output_dir = tmp_path / "output"

        task = EvalDetectionTask(
            val_dir=val_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            run_gemini=False,
            run_smolvlm2=False,
            run_rfdetr=True,
            onnx_model_path=tmp_path / "model.onnx",
        )

        # Mock the RFDETR inferencer builder
        mock_inferencer = MagicMock()
        mock_inferencer.predict.return_value = [
            Detection(
                bbox=BoundingBox(x=0.15, y=0.2, w=0.3, h=0.6),
                confidence=0.9,
                class_id=0,
            ),
        ]

        with patch.object(
            task, "_build_rfdetr_inferencer", return_value=mock_inferencer
        ):
            result = task.run()

        assert output_dir.exists()
        assert (output_dir / "summary.csv").exists()
        assert (output_dir / "results.json").exists()
        assert result["output_dir"] == str(output_dir)

        # Check results JSON
        with open(output_dir / "results.json") as f:
            results_data = json.load(f)
        assert "RF-DETR" in results_data

    def test_run_no_methods(self, tmp_path: Path) -> None:
        """Run eval with no methods enabled produces empty outputs."""
        val_dir = tmp_path / "val"
        test_dir = tmp_path / "test"
        val_dir.mkdir()
        test_dir.mkdir()
        _make_coco_json(val_dir / "_annotations.coco.json", num_images=1)
        _make_coco_json(test_dir / "_annotations.coco.json", num_images=1)

        output_dir = tmp_path / "output"
        task = EvalDetectionTask(
            val_dir=val_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            run_gemini=False,
            run_smolvlm2=False,
            run_rfdetr=False,
        )

        task.run()
        assert (output_dir / "summary.csv").exists()
        assert (output_dir / "results.json").exists()

    def test_rfdetr_requires_model_path(self, tmp_path: Path) -> None:
        val_dir = tmp_path / "val"
        test_dir = tmp_path / "test"
        val_dir.mkdir()
        test_dir.mkdir()
        _make_coco_json(val_dir / "_annotations.coco.json", num_images=1)
        _make_coco_json(test_dir / "_annotations.coco.json", num_images=1)

        task = EvalDetectionTask(
            val_dir=val_dir,
            test_dir=test_dir,
            output_dir=tmp_path / "output",
            run_gemini=False,
            run_smolvlm2=False,
            run_rfdetr=True,
            onnx_model_path=None,
        )

        with pytest.raises(ValueError, match="onnx_model_path is required"):
            task.run()
