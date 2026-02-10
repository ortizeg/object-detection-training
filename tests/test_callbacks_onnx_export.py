"""Tests for the ONNX export callback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from object_detection_training.callbacks.onnx_export import ONNXExportCallback


class TestONNXExportCallbackInit:
    """Tests for initialization."""

    def test_default_parameters(self) -> None:
        cb = ONNXExportCallback()
        assert cb.output_dir == Path("onnx")
        assert cb.export_best is True
        assert cb.export_final is True
        assert cb.export_all_checkpoints is False
        assert cb.opset_version == 17
        assert cb.simplify is True

    def test_custom_parameters(self) -> None:
        cb = ONNXExportCallback(
            output_dir="custom_onnx",
            export_best=False,
            opset_version=16,
        )
        assert cb.output_dir == Path("custom_onnx")
        assert cb.export_best is False
        assert cb.opset_version == 16


class TestONNXExportCallbackExportModel:
    """Tests for _export_model method."""

    def test_calls_export_onnx(self, tmp_path: Path) -> None:
        """Delegates to pl_module.export_onnx when available."""
        pl_module = MagicMock()
        pl_module.export_onnx = MagicMock(return_value=str(tmp_path / "model.onnx"))

        cb = ONNXExportCallback()
        result = cb._export_model(pl_module, tmp_path / "model.onnx")

        pl_module.export_onnx.assert_called_once_with(
            output_path=str(tmp_path / "model.onnx"),
            input_height=640,
            input_width=640,
            opset_version=17,
            simplify=True,
        )
        assert result is not None

    def test_returns_none_without_export_onnx(self) -> None:
        """Returns None when model doesn't have export_onnx method."""
        pl_module = MagicMock(spec=[])  # empty spec = no methods

        cb = ONNXExportCallback()
        result = cb._export_model(pl_module, Path("model.onnx"))

        assert result is None

    def test_handles_export_exception(self) -> None:
        """Returns None when export_onnx raises an exception."""
        pl_module = MagicMock()
        pl_module.export_onnx.side_effect = RuntimeError("Export failed")

        cb = ONNXExportCallback()
        result = cb._export_model(pl_module, Path("model.onnx"))

        assert result is None


class TestONNXExportCallbackOnTrainEnd:
    """Tests for on_train_end hook."""

    def test_exports_final_model(self, tmp_path: Path) -> None:
        """Final model is exported when export_final is True."""
        pl_module = MagicMock()
        pl_module.export_onnx = MagicMock(return_value="model.onnx")

        trainer = MagicMock()
        trainer.log_dir = str(tmp_path)
        trainer.checkpoint_callback = None

        cb = ONNXExportCallback(export_best=False, export_final=True)
        cb.on_train_end(trainer, pl_module)

        pl_module.export_onnx.assert_called_once()

    def test_skips_final_when_disabled(self, tmp_path: Path) -> None:
        """Final model is NOT exported when export_final is False."""
        pl_module = MagicMock()
        pl_module.export_onnx = MagicMock()

        trainer = MagicMock()
        trainer.log_dir = str(tmp_path)
        trainer.checkpoint_callback = None

        cb = ONNXExportCallback(export_best=False, export_final=False)
        cb.on_train_end(trainer, pl_module)

        pl_module.export_onnx.assert_not_called()


class TestONNXExportCallbackStateDict:
    """Tests for state dict."""

    def test_state_dict_roundtrip(self) -> None:
        cb = ONNXExportCallback()
        cb._exported_checkpoints = ["model_a.onnx", "model_b.onnx"]

        state = cb.state_dict()
        cb2 = ONNXExportCallback()
        cb2.load_state_dict(state)

        assert cb2._exported_checkpoints == ["model_a.onnx", "model_b.onnx"]
