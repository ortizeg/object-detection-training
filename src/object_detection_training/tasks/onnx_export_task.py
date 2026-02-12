"""
ONNX Export Task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import omegaconf
import torch
from loguru import logger
from pydantic import Field

from object_detection_training.tasks.base_task import BaseTask
from object_detection_training.utils.hydra import register


@register(group="task")
class ONNXExportTask(BaseTask):
    """Export a trained checkpoint to an optimized ONNX file.

    Loads a PyTorch Lightning checkpoint (or raw state-dict) into the
    specified model class and delegates to
    :pymeth:`BaseDetectionModel.export_onnx` for the actual export.
    """

    name: str = Field(default="export_onnx", description="Task name")

    # Model to export - instantiated via Hydra (same as TrainTask.model)
    model: L.LightningModule = Field(
        description="Model instance (instantiated via Hydra)"
    )

    # Checkpoint / weights path
    checkpoint_path: Path = Field(
        description="Path to a .ckpt (Lightning) or .pt/.pth (raw state-dict) file"
    )

    # Export settings
    output_path: Path = Field(
        default=Path("model.onnx"),
        description="Destination path for the exported ONNX file",
    )
    opset_version: int = Field(default=17, description="ONNX opset version")
    simplify: bool = Field(default=True, description="Simplify the ONNX graph")
    input_height: int | None = Field(
        default=None,
        description="Input image height (inferred from checkpoint if None)",
    )
    input_width: int | None = Field(
        default=None, description="Input image width (inferred from checkpoint if None)"
    )
    num_classes: int | None = Field(
        default=None,
        description=(
            "Number of classes the checkpoint was trained with. "
            "Required when the checkpoint class count differs from the "
            "model default (e.g. 80 for COCO)."
        ),
    )
    dynamic_batch: bool = Field(
        default=True,
        description="Whether to export with dynamic batch size (axis 0)",
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _register_safe_globals() -> None:
        """Allow omegaconf containers to be unpickled safely."""
        torch.serialization.add_safe_globals(
            [
                omegaconf.listconfig.ListConfig,
                omegaconf.dictconfig.DictConfig,
                omegaconf.base.ContainerMetadata,
                omegaconf.base.Metadata,
                omegaconf.nodes.AnyNode,
            ]
        )

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load a checkpoint file and return the raw dict."""
        self._register_safe_globals()
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint: dict[str, Any] = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def run(self) -> dict[str, str | None]:
        """Load checkpoint weights into the model and export to ONNX.

        Returns:
            Dict with ``onnx_path`` pointing to the exported file.
        """
        # Resolve output directory
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            onnx_out = self.output_dir / self.output_path
        else:
            onnx_out = self.output_path

        # Load and apply weights
        checkpoint = self._load_checkpoint()
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        logger.info("Checkpoint weights loaded successfully")

        # Infer dimensions from checkpoint if not provided
        hparams = checkpoint.get("hyper_parameters", {})

        # Use local variables to avoid mutating self if we want preservation,
        # but self is okay here.
        input_height = self.input_height
        input_width = self.input_width

        if input_height is None:
            input_height = hparams.get("input_height", 640)
            logger.info(f"Inferred input_height={input_height} from checkpoint")

        if input_width is None:
            input_width = hparams.get("input_width", 640)
            logger.info(f"Inferred input_width={input_width} from checkpoint")

        # Prepare dynamic axes
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "dets": {0: "batch_size"},
                "labels": {0: "batch_size"},
            }
            logger.info("Exporting with dynamic batch size")

        # Export via BaseDetectionModel.export_onnx
        if not hasattr(self.model, "export_onnx"):
            msg = (
                f"Model {type(self.model).__name__} does not implement export_onnx. "
                "Only BaseDetectionModel subclasses are supported."
            )
            raise AttributeError(msg)

        onnx_path = self.model.export_onnx(  # type: ignore[operator]
            output_path=str(onnx_out),
            input_height=input_height,
            input_width=input_width,
            opset_version=self.opset_version,
            simplify=self.simplify,
            dynamic_axes=dynamic_axes,
        )
        logger.info(f"ONNX model exported to {onnx_path}")

        return {"onnx_path": onnx_path}
