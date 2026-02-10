"""Tests for the ModelInfoCallback."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from torch import nn

from object_detection_training.callbacks.model_info import ModelInfoCallback


def _make_module() -> MagicMock:
    """Create a mock module backed by a real nn.Linear."""
    model = nn.Linear(10, 5)
    pl_module = MagicMock()
    pl_module.parameters.return_value = list(model.parameters())
    pl_module.buffers.return_value = list(model.buffers())
    # Ensure num_classes is a real value, not a MagicMock
    pl_module.num_classes = 3
    # Remove compute_model_stats by default (so _compute_basic_stats is used)
    del pl_module.compute_model_stats
    return pl_module


class TestModelInfoCallbackInit:
    """Tests for initialization."""

    def test_default_parameters(self) -> None:
        cb = ModelInfoCallback()
        assert cb.output_dir is None
        assert cb.output_filename == "model_info.json"
        assert cb.input_height == 640
        assert cb.input_width == 640

    def test_custom_parameters(self) -> None:
        cb = ModelInfoCallback(
            output_dir="/custom",
            output_filename="info.json",
            input_height=480,
            input_width=480,
        )
        assert cb.output_dir == Path("/custom")
        assert cb.output_filename == "info.json"
        assert cb.input_height == 480


class TestModelInfoCallbackBasicStats:
    """Tests for _compute_basic_stats."""

    def test_computes_param_count(self) -> None:
        """Basic stats include total and trainable parameter counts."""
        model = nn.Linear(10, 5)
        pl_module = MagicMock()
        pl_module.parameters.return_value = list(model.parameters())
        pl_module.buffers.return_value = list(model.buffers())

        cb = ModelInfoCallback()
        stats = cb._compute_basic_stats(pl_module)

        # Linear(10, 5) has 10*5 + 5 = 55 params
        assert stats["total_params"] == 55
        assert stats["trainable_params"] == 55
        assert "model_size_mb" in stats

    def test_frozen_params_counted_separately(self) -> None:
        """Frozen parameters are counted as total but not trainable."""
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False

        pl_module = MagicMock()
        pl_module.parameters.return_value = list(model.parameters())
        pl_module.buffers.return_value = list(model.buffers())

        cb = ModelInfoCallback()
        stats = cb._compute_basic_stats(pl_module)

        assert stats["total_params"] == 55
        assert stats["trainable_params"] == 5  # only bias


class TestModelInfoCallbackOnFitStart:
    """Tests for on_fit_start hook."""

    def test_saves_model_info_json(self, tmp_path: Path) -> None:
        """Model info JSON is saved to disk."""
        pl_module = _make_module()
        trainer = MagicMock()
        trainer.log_dir = str(tmp_path)
        trainer.loggers = []
        trainer.datamodule = None

        cb = ModelInfoCallback()
        cb.on_fit_start(trainer, pl_module)

        output_path = tmp_path / "model_info.json"
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "total_params" in data
        assert "model_class" in data

    def test_saves_to_custom_output_dir(self, tmp_path: Path) -> None:
        """Model info is saved to custom output directory."""
        custom_dir = tmp_path / "custom"
        pl_module = _make_module()
        trainer = MagicMock()
        trainer.log_dir = str(tmp_path)
        trainer.loggers = []
        trainer.datamodule = None

        cb = ModelInfoCallback(output_dir=str(custom_dir))
        cb.on_fit_start(trainer, pl_module)

        assert (custom_dir / "model_info.json").exists()

    def test_uses_compute_model_stats_if_available(self, tmp_path: Path) -> None:
        """Calls compute_model_stats on module if available."""
        pl_module = _make_module()
        pl_module.compute_model_stats = MagicMock(
            return_value={"total_params": 999, "flops": 1000}
        )
        trainer = MagicMock()
        trainer.log_dir = str(tmp_path)
        trainer.loggers = []
        trainer.datamodule = None

        cb = ModelInfoCallback()
        cb.on_fit_start(trainer, pl_module)

        pl_module.compute_model_stats.assert_called_once()
        assert cb.model_info["total_params"] == 999


class TestModelInfoCallbackStateDict:
    """Tests for state_dict and load_state_dict."""

    def test_state_dict_roundtrip(self) -> None:
        cb = ModelInfoCallback()
        cb.model_info = {"total_params": 100, "model_class": "TestModel"}  # type: ignore[typeddict-item]

        state = cb.state_dict()
        cb2 = ModelInfoCallback()
        cb2.load_state_dict(state)

        assert cb2.model_info["total_params"] == 100


class TestModelInfoCallbackLabelsMapping:
    """Tests for labels mapping export."""

    def test_exports_labels_from_class_names(self, tmp_path: Path) -> None:
        """Labels mapping JSON is exported from datamodule.class_names."""
        pl_module = _make_module()
        trainer = MagicMock()
        trainer.log_dir = str(tmp_path)
        trainer.loggers = []
        trainer.datamodule = MagicMock()
        trainer.datamodule.class_names = ["person", "ball", "hoop"]
        # Ensure train_detection_dataset is not found so fallback path is used
        del trainer.datamodule.train_detection_dataset

        cb = ModelInfoCallback()
        cb.on_fit_start(trainer, pl_module)

        labels_path = tmp_path / "labels_mapping.json"
        assert labels_path.exists()

        with open(labels_path) as f:
            mapping = json.load(f)
        assert mapping["0"] == "person"
        assert mapping["2"] == "hoop"
