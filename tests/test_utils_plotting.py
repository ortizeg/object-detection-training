"""Tests for plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from object_detection_training.utils.plotting import (
    plot_pr_curve,
    save_detection_curves_plots,
)


class TestPlotPrCurve:
    """Tests for plot_pr_curve function."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """PR curve plot is saved to disk."""
        recall = np.linspace(0, 1, 50)
        precision = 1 - recall  # simple descending precision
        save_path = tmp_path / "pr_curve.png"

        plot_pr_curve(recall, precision, "person", save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_with_ap_value(self, tmp_path: Path) -> None:
        """PR curve renders correctly when AP is provided."""
        recall = np.linspace(0, 1, 50)
        precision = np.ones_like(recall) * 0.8
        save_path = tmp_path / "pr_ap.png"

        plot_pr_curve(recall, precision, "car", save_path, ap=0.85)

        assert save_path.exists()

    def test_without_ap_value(self, tmp_path: Path) -> None:
        """PR curve renders correctly without AP."""
        recall = np.linspace(0, 1, 50)
        precision = np.ones_like(recall)
        save_path = tmp_path / "pr_no_ap.png"

        plot_pr_curve(recall, precision, "ball", save_path, ap=None)

        assert save_path.exists()


class TestSaveDetectionCurvesPlots:
    """Tests for save_detection_curves_plots function."""

    def test_saves_per_class_plots(self, tmp_path: Path) -> None:
        """One PNG is created per class in curves_data."""
        curves_data = {
            0: {
                "precision": np.linspace(1, 0.5, 20),
                "recall": np.linspace(0, 1, 20),
                "scores": np.linspace(0.9, 0.1, 20),
                "f1": np.ones(20) * 0.7,
            },
            1: {
                "precision": np.linspace(1, 0.3, 20),
                "recall": np.linspace(0, 1, 20),
                "scores": np.linspace(0.8, 0.05, 20),
                "f1": np.ones(20) * 0.5,
            },
        }
        class_names = ["person", "ball"]

        save_detection_curves_plots(curves_data, class_names, tmp_path)

        assert (tmp_path / "pr_curve_person.png").exists()
        assert (tmp_path / "pr_curve_ball.png").exists()

    def test_saves_overall_plot(self, tmp_path: Path) -> None:
        """Overall curve is saved with 'overall' filename."""
        curves_data = {
            "overall": {
                "precision": np.linspace(1, 0.5, 20),
                "recall": np.linspace(0, 1, 20),
                "scores": np.linspace(0.9, 0.1, 20),
                "f1": np.ones(20) * 0.6,
            },
        }

        save_detection_curves_plots(curves_data, None, tmp_path)

        assert (tmp_path / "pr_curve_overall.png").exists()

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "plots"
        curves_data = {
            0: {
                "precision": np.ones(10),
                "recall": np.linspace(0, 1, 10),
                "scores": np.ones(10),
                "f1": np.ones(10),
            },
        }

        save_detection_curves_plots(curves_data, ["class_0"], output_dir)

        assert output_dir.exists()

    def test_with_prefix(self, tmp_path: Path) -> None:
        """Prefix is prepended to filenames."""
        curves_data = {
            0: {
                "precision": np.ones(10),
                "recall": np.linspace(0, 1, 10),
                "scores": np.ones(10),
                "f1": np.ones(10),
            },
        }

        save_detection_curves_plots(
            curves_data, ["person"], tmp_path, prefix="epoch10_"
        )

        assert (tmp_path / "epoch10_pr_curve_person.png").exists()
