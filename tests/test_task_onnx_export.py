"""Tests for the ONNXExportTask."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from object_detection_training.tasks import ONNXExportTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_mock() -> MagicMock:
    """Return a mock LightningModule with export_onnx."""
    import lightning as L

    model = MagicMock(spec=L.LightningModule)
    model.export_onnx = MagicMock(return_value="output/model.onnx")
    return model


def _make_task(tmp_path: Path, **overrides: Any) -> ONNXExportTask:
    """Create an ONNXExportTask with sensible defaults."""
    ckpt = tmp_path / "checkpoint.ckpt"
    ckpt.touch()
    defaults: dict[str, Any] = {
        "model": _make_model_mock(),
        "checkpoint_path": ckpt,
        "output_dir": tmp_path / "output",
        "output_path": Path("model.onnx"),
    }
    defaults.update(overrides)
    return ONNXExportTask(**defaults)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestONNXExportTaskInit:
    """Tests for ONNXExportTask initialization and validation."""

    def test_defaults(self, tmp_path: Path) -> None:
        """Default field values are set correctly."""
        task = _make_task(tmp_path)
        assert task.name == "export_onnx"
        assert task.opset_version == 17
        assert task.simplify is True
        assert task.input_height == 640
        assert task.input_width == 640

    def test_custom_opset(self, tmp_path: Path) -> None:
        """Opset version can be overridden."""
        task = _make_task(tmp_path, opset_version=16)
        assert task.opset_version == 16

    def test_forbids_extra_fields(self, tmp_path: Path) -> None:
        """Extra fields are rejected by Pydantic."""
        with pytest.raises(Exception):  # noqa: B017
            _make_task(tmp_path, unknown_field="oops")


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------


class TestONNXExportTaskRun:
    """Tests for the run() method."""

    @patch("object_detection_training.tasks.torch")
    def test_run_calls_export_onnx(self, mock_torch: MagicMock, tmp_path: Path) -> None:
        """run() loads checkpoint and calls model.export_onnx."""
        mock_torch.load.return_value = {"state_dict": {}}

        task = _make_task(tmp_path)
        result = task.run()

        # Verify checkpoint was loaded
        mock_torch.load.assert_called_once()

        # Verify model.load_state_dict was called
        task.model.load_state_dict.assert_called_once_with({})

        # Verify export_onnx was called with correct args
        task.model.export_onnx.assert_called_once()
        call_kwargs = task.model.export_onnx.call_args
        assert "output_path" in call_kwargs.kwargs
        assert call_kwargs.kwargs["opset_version"] == 17
        assert call_kwargs.kwargs["simplify"] is True

        assert "onnx_path" in result

    @patch("object_detection_training.tasks.torch")
    def test_run_handles_raw_state_dict(
        self, mock_torch: MagicMock, tmp_path: Path
    ) -> None:
        """run() handles raw state dicts (no 'state_dict' key)."""
        raw_weights = {"layer.weight": "fake_tensor"}
        mock_torch.load.return_value = raw_weights

        task = _make_task(tmp_path)
        task.run()

        # When there's no 'state_dict' key, the whole dict is the state dict
        task.model.load_state_dict.assert_called_once_with(raw_weights)

    @patch("object_detection_training.tasks.torch")
    def test_run_raises_without_export_onnx(
        self, mock_torch: MagicMock, tmp_path: Path
    ) -> None:
        """run() raises AttributeError if model lacks export_onnx."""
        import lightning as L

        mock_torch.load.return_value = {"state_dict": {}}

        model = MagicMock(spec=L.LightningModule)
        del model.export_onnx  # Ensure it doesn't have export_onnx

        task = _make_task(tmp_path, model=model)

        with pytest.raises(AttributeError, match="does not implement export_onnx"):
            task.run()

    @patch("object_detection_training.tasks.torch")
    def test_run_creates_output_dir(
        self, mock_torch: MagicMock, tmp_path: Path
    ) -> None:
        """run() creates output_dir if it doesn't exist."""
        mock_torch.load.return_value = {"state_dict": {}}
        out_dir = tmp_path / "nested" / "output"

        task = _make_task(tmp_path, output_dir=out_dir)
        task.run()

        assert out_dir.exists()


# ---------------------------------------------------------------------------
# _load_checkpoint tests
# ---------------------------------------------------------------------------


class TestONNXExportTaskLoadCheckpoint:
    """Tests for checkpoint loading."""

    @patch("object_detection_training.tasks.torch")
    def test_registers_safe_globals(
        self, mock_torch: MagicMock, tmp_path: Path
    ) -> None:
        """_load_checkpoint registers omegaconf safe globals."""
        mock_torch.load.return_value = {}

        task = _make_task(tmp_path)
        task._load_checkpoint()

        mock_torch.serialization.add_safe_globals.assert_called_once()

    @patch("object_detection_training.tasks.torch")
    def test_loads_with_cpu_map_location(
        self, mock_torch: MagicMock, tmp_path: Path
    ) -> None:
        """Checkpoint is loaded to CPU regardless of original device."""
        mock_torch.load.return_value = {}

        task = _make_task(tmp_path)
        task._load_checkpoint()

        call_kwargs = mock_torch.load.call_args
        assert call_kwargs.kwargs["map_location"] == "cpu"
