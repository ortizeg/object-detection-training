"""Tests for the DatasetStatisticsCallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from object_detection_training.callbacks.statistics import DatasetStatisticsCallback


class TestDatasetStatisticsCallbackInit:
    """Tests for initialization."""

    def test_default_output_dir(self) -> None:
        cb = DatasetStatisticsCallback()
        assert cb.output_dir == Path("outputs")

    def test_custom_output_dir(self) -> None:
        cb = DatasetStatisticsCallback(output_dir="stats_out")
        assert cb.output_dir == Path("stats_out")


class TestDatasetStatisticsCallbackNoDatamodule:
    """Tests for graceful handling when no datamodule exists."""

    def test_skips_when_no_datamodule(self) -> None:
        """Callback does nothing when trainer has no datamodule."""
        cb = DatasetStatisticsCallback()
        trainer = MagicMock()
        trainer.datamodule = None
        pl_module = MagicMock()

        # Should not raise
        cb.on_fit_start(trainer, pl_module)

    def test_skips_when_datamodule_not_detection(self, tmp_path: Path) -> None:
        """Callback does nothing when datamodule doesn't implement protocol."""
        cb = DatasetStatisticsCallback(output_dir=str(tmp_path))

        # A plain MagicMock won't match the _DetectionDataModule protocol
        trainer = MagicMock()
        trainer.datamodule = MagicMock(spec=[])  # empty spec
        pl_module = MagicMock()

        # Should not raise
        cb.on_fit_start(trainer, pl_module)


class TestDatasetStatisticsCallbackStateDict:
    """Tests for state dict."""

    def test_empty_state_dict(self) -> None:
        cb = DatasetStatisticsCallback()
        assert cb.state_dict() == {}

    def test_load_state_dict_noop(self) -> None:
        cb = DatasetStatisticsCallback()
        cb.load_state_dict({})  # Should not raise
