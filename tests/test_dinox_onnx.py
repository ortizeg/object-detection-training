"""Tests for DINO-X ONNX export round-trip.

Verifies ONNX export produces valid models with correct output shapes,
DFL integral baked in, and dynamic batch support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from object_detection_training.models.dinox_lightning import DINOXLightningModel

NUM_CLASSES = 2
INPUT_SIZE = 320


def _make_lightning_model(use_dfl: bool = False) -> DINOXLightningModel:
    """Create a small DINOXLightningModel for testing."""
    return DINOXLightningModel(
        num_classes=NUM_CLASSES,
        download_pretrained=False,
        depth=0.67,
        width=0.75,
        use_dfl=use_dfl,
        reg_max=16,
    )


class TestDINOXOnnxExport:
    """ONNX export round-trip tests."""

    def test_onnx_export_baseline(self, tmp_path: Path) -> None:
        """Export baseline model, verify valid ONNX with correct I/O names."""
        model = _make_lightning_model(use_dfl=False)
        onnx_path = str(tmp_path / "baseline.onnx")
        model.export_onnx(
            onnx_path,
            input_height=INPUT_SIZE,
            input_width=INPUT_SIZE,
            simplify=False,
        )

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        assert len(onnx_model.graph.input) == 1
        assert onnx_model.graph.input[0].name == "input"
        assert len(onnx_model.graph.output) == 1
        assert onnx_model.graph.output[0].name == "output"

    def test_onnx_export_dfl(self, tmp_path: Path) -> None:
        """Export DFL model, verify valid ONNX with correct I/O names."""
        model = _make_lightning_model(use_dfl=True)
        onnx_path = str(tmp_path / "dfl.onnx")
        model.export_onnx(
            onnx_path,
            input_height=INPUT_SIZE,
            input_width=INPUT_SIZE,
            simplify=False,
        )

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        assert onnx_model.graph.input[0].name == "input"
        assert onnx_model.graph.output[0].name == "output"

    def test_onnx_output_shape_matches_pytorch(self, tmp_path: Path) -> None:
        """ONNX runtime output shape matches PyTorch eval output shape."""
        model = _make_lightning_model(use_dfl=True)
        model.eval()

        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            pt_out = model(x)
        pt_shape = pt_out["predictions"].shape

        onnx_path = str(tmp_path / "shape_test.onnx")
        model.export_onnx(
            onnx_path,
            input_height=INPUT_SIZE,
            input_width=INPUT_SIZE,
            simplify=False,
        )

        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {"input": x.numpy()})
        ort_shape = ort_out[0].shape

        assert pt_shape == torch.Size(ort_shape), (
            f"Shape mismatch: PyTorch={pt_shape} vs ONNX={ort_shape}"
        )

    def test_onnx_dynamic_batch(self, tmp_path: Path) -> None:
        """Export with dynamic batch, run with batch_size=2."""
        model = _make_lightning_model(use_dfl=True)
        onnx_path = str(tmp_path / "dynamic.onnx")
        model.export_onnx(
            onnx_path,
            input_height=INPUT_SIZE,
            input_width=INPUT_SIZE,
            simplify=False,
        )

        x = np.random.randn(2, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {"input": x})

        assert ort_out[0].shape[0] == 2
        assert ort_out[0].shape[2] == 5 + NUM_CLASSES

    def test_onnx_dfl_integral_baked_in(self, tmp_path: Path) -> None:
        """DFL model output has 5+C columns (not 4*(reg_max+1)+1+C)."""
        model = _make_lightning_model(use_dfl=True)
        onnx_path = str(tmp_path / "dfl_integral.onnx")
        model.export_onnx(
            onnx_path,
            input_height=INPUT_SIZE,
            input_width=INPUT_SIZE,
            simplify=False,
        )

        x = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {"input": x})

        # Should be 5 + num_classes = 7, NOT 4*(16+1) + 1 + 2 = 71
        assert ort_out[0].shape[2] == 5 + NUM_CLASSES
