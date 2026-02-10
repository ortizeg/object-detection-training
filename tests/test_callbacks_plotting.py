"""Tests for the TrainingHistoryPlotter callback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import torch

from object_detection_training.callbacks.plotting import TrainingHistoryPlotter


class TestTrainingHistoryPlotterInit:
    """Tests for initialization."""

    def test_default_output_dir(self) -> None:
        plotter = TrainingHistoryPlotter()
        assert plotter.output_dir == Path("outputs")

    def test_custom_output_dir(self) -> None:
        plotter = TrainingHistoryPlotter(output_dir="my_dir")
        assert plotter.output_dir == Path("my_dir")

    def test_initial_history_keys(self) -> None:
        plotter = TrainingHistoryPlotter()
        expected_keys = {"train_loss", "val_loss", "val_mAP", "val_mAP50", "val_mAP75"}
        assert set(plotter.history.keys()) == expected_keys
        for values in plotter.history.values():
            assert values == []


class TestTrainingHistoryPlotterMetrics:
    """Tests for metric collection and plotting."""

    def test_collects_metrics(self, tmp_path: Path) -> None:
        """Metrics are collected and history is appended."""
        plotter = TrainingHistoryPlotter(output_dir=str(tmp_path))

        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "train/loss_epoch": torch.tensor(1.5),
            "val/loss_epoch": torch.tensor(2.0),
            "val/mAP": torch.tensor(0.3),
            "val/mAP_50": torch.tensor(0.5),
            "val/mAP_75": torch.tensor(0.2),
        }
        pl_module = MagicMock()

        plotter.on_train_epoch_end(trainer, pl_module)

        assert len(plotter.epochs) == 1
        assert plotter.epochs[0] == 0
        assert plotter.history["train_loss"] == [1.5]
        assert plotter.history["val_loss"] == [2.0]
        assert plotter.history["val_mAP"] is not None
        assert abs(plotter.history["val_mAP"][0] - 0.3) < 1e-5

    def test_creates_plot_files(self, tmp_path: Path) -> None:
        """Loss and mAP plot files are created."""
        plotter = TrainingHistoryPlotter(output_dir=str(tmp_path))

        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "train/loss_epoch": torch.tensor(1.5),
            "val/mAP": torch.tensor(0.3),
        }
        pl_module = MagicMock()

        plotter.on_train_epoch_end(trainer, pl_module)

        assert (tmp_path / "loss_history.png").exists()
        assert (tmp_path / "map_history.png").exists()

    def test_handles_missing_metrics(self, tmp_path: Path) -> None:
        """Missing metrics are handled gracefully with None."""
        plotter = TrainingHistoryPlotter(output_dir=str(tmp_path))

        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.callback_metrics = {}  # no metrics available
        pl_module = MagicMock()

        plotter.on_train_epoch_end(trainer, pl_module)

        assert plotter.history["train_loss"] == [None]
        assert plotter.history["val_loss"] == [None]

    def test_multiple_epochs(self, tmp_path: Path) -> None:
        """History accumulates across multiple epochs."""
        plotter = TrainingHistoryPlotter(output_dir=str(tmp_path))
        pl_module = MagicMock()

        for epoch in range(3):
            trainer = MagicMock()
            trainer.current_epoch = epoch
            trainer.callback_metrics = {
                "train/loss_epoch": torch.tensor(1.0 / (epoch + 1)),
            }
            plotter.on_train_epoch_end(trainer, pl_module)

        assert len(plotter.epochs) == 3
        assert len(plotter.history["train_loss"]) == 3
