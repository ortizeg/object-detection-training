"""Tests for detection metrics curves."""

from __future__ import annotations

import torch

from object_detection_training.metrics.curves import compute_detection_curves


class TestComputeDetectionCurvesSingleClass:
    """Tests for single-class detection curves."""

    def test_perfect_detection(self) -> None:
        """Perfect detection produces recall=1, precision=1."""
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        curves = compute_detection_curves(preds, targets, num_classes=1)

        assert 0 in curves
        assert curves[0]["recall"][-1] == 1.0
        assert curves[0]["precision"][-1] > 0.99

    def test_no_detections(self) -> None:
        """No predictions produces empty curves."""
        preds = [
            {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        curves = compute_detection_curves(preds, targets, num_classes=1)

        # With no preds, the class may not have curve data
        # (no flat_preds, so empty tps/fps)
        assert 0 not in curves or len(curves[0]["recall"]) == 0

    def test_false_positive(self) -> None:
        """Detection far from GT produces precision < 1."""
        preds = [
            {
                "boxes": torch.tensor([[200.0, 200.0, 300.0, 300.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        curves = compute_detection_curves(
            preds, targets, num_classes=1, iou_threshold=0.5
        )

        assert 0 in curves
        # It's a false positive, so precision should be 0
        assert curves[0]["precision"][-1] < 0.01


class TestComputeDetectionCurvesMultiClass:
    """Tests for multi-class detection curves."""

    def test_two_classes(self) -> None:
        """Curves are computed separately for each class."""
        preds = [
            {
                "boxes": torch.tensor(
                    [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]
                ),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor(
                    [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]
                ),
                "labels": torch.tensor([0, 1]),
            }
        ]

        curves = compute_detection_curves(preds, targets, num_classes=2)

        assert 0 in curves
        assert 1 in curves

    def test_overall_curve(self) -> None:
        """Overall micro-averaged curve is computed."""
        preds = [
            {
                "boxes": torch.tensor(
                    [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]
                ),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor(
                    [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]
                ),
                "labels": torch.tensor([0, 1]),
            }
        ]

        curves = compute_detection_curves(preds, targets, num_classes=2)

        assert "overall" in curves
        assert len(curves["overall"]["recall"]) > 0


class TestComputeDetectionCurvesMultipleImages:
    """Tests with multiple images."""

    def test_multiple_images(self) -> None:
        """Curves handle multiple images correctly."""
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
                "scores": torch.tensor([0.7]),
                "labels": torch.tensor([0]),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
                "labels": torch.tensor([0]),
            },
        ]

        curves = compute_detection_curves(preds, targets, num_classes=1)

        assert 0 in curves
        assert curves[0]["recall"][-1] == 1.0

    def test_curve_data_has_expected_keys(self) -> None:
        """Each curve entry has precision, recall, scores, f1."""
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        curves = compute_detection_curves(preds, targets, num_classes=1)

        assert 0 in curves
        curve = curves[0]
        assert "precision" in curve
        assert "recall" in curve
        assert "scores" in curve
        assert "f1" in curve
