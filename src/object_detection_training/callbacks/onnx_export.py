"""
ONNX export callback for PyTorch Lightning.

Automatically exports models to ONNX format during training.
"""

from pathlib import Path
from typing import List, Optional

import lightning as L
import omegaconf
import torch
from loguru import logger


class ONNXExportCallback(L.Callback):
    """
    Callback to export models to ONNX format.

    Exports the best model, final model, and optionally all kept checkpoints.
    """

    def __init__(
        self,
        output_dir: str = "onnx",
        export_best: bool = True,
        export_final: bool = True,
        export_all_checkpoints: bool = False,
        opset_version: int = 17,
        simplify: bool = True,
        input_height: int = 640,
        input_width: int = 640,
    ):
        """
        Initialize ONNX export callback.

        Args:
            output_dir: Directory to save ONNX models.
            export_best: Export best model checkpoint.
            export_final: Export final model at training end.
            export_all_checkpoints: Export all saved checkpoints.
            opset_version: ONNX opset version.
            simplify: Whether to simplify ONNX model.
            input_height: Input image height for export.
            input_width: Input image width for export.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.export_best = export_best
        self.export_final = export_final
        self.export_all_checkpoints = export_all_checkpoints
        self.opset_version = opset_version
        self.simplify = simplify
        self.input_height = input_height
        self.input_width = input_width

        self._exported_checkpoints: List[str] = []

    def _export_model(
        self,
        pl_module: L.LightningModule,
        output_path: Path,
    ) -> Optional[str]:
        """Export model to ONNX."""
        try:
            if hasattr(pl_module, "export_onnx"):
                return pl_module.export_onnx(
                    output_path=str(output_path),
                    input_height=self.input_height,
                    input_width=self.input_width,
                    opset_version=self.opset_version,
                    simplify=self.simplify,
                )
            else:
                logger.warning(
                    f"Model {type(pl_module).__name__} does not have export_onnx method"
                )
                return None
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            return None

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Export models at the end of training."""
        # Determine output directory from trainer
        log_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
        export_dir = log_dir / self.output_dir
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export final model
        if self.export_final:
            logger.info("Exporting final model to ONNX")
            output_path = export_dir / "model_final.onnx"
            exported = self._export_model(pl_module, output_path)
            if exported:
                self._exported_checkpoints.append(exported)

        # Export best model
        if self.export_best and trainer.checkpoint_callback:
            best_path = trainer.checkpoint_callback.best_model_path
            if best_path:
                try:
                    # Load best checkpoint
                    torch.serialization.add_safe_globals(
                        [
                            omegaconf.listconfig.ListConfig,
                            omegaconf.dictconfig.DictConfig,
                            omegaconf.base.ContainerMetadata,
                            omegaconf.base.Metadata,
                            omegaconf.nodes.AnyNode,
                        ]
                    )

                    logger.info(f"Loading best checkpoint from {best_path}")
                    checkpoint = torch.load(best_path, map_location="cpu")

                    # Handle both Lightning and raw state dicts
                    state_dict = checkpoint.get("state_dict", checkpoint)

                    pl_module.load_state_dict(state_dict)
                    logger.info("Best model loaded successfully for ONNX export")

                    output_path = export_dir / "model_best.onnx"
                    exported = self._export_model(pl_module, output_path)
                    if exported:
                        self._exported_checkpoints.append(exported)
                except Exception as e:
                    logger.error(f"Failed to load or export best checkpoint: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

        # Export all checkpoints
        if self.export_all_checkpoints and trainer.checkpoint_callback:
            checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
            for ckpt_file in checkpoint_dir.glob("*.ckpt"):
                if ckpt_file.stem in ["last", "best"]:
                    continue

                logger.info(f"Exporting checkpoint: {ckpt_file}")
                try:
                    torch.serialization.add_safe_globals(
                        [
                            omegaconf.listconfig.ListConfig,
                            omegaconf.dictconfig.DictConfig,
                            omegaconf.base.ContainerMetadata,
                            omegaconf.base.Metadata,
                            omegaconf.nodes.AnyNode,
                        ]
                    )

                    checkpoint = torch.load(ckpt_file, map_location="cpu")
                    pl_module.load_state_dict(checkpoint["state_dict"])

                    output_path = export_dir / f"{ckpt_file.stem}.onnx"
                    exported = self._export_model(pl_module, output_path)
                    if exported:
                        self._exported_checkpoints.append(exported)
                except Exception as e:
                    logger.error(f"Failed to export {ckpt_file}: {e}")

        logger.info(f"Exported {len(self._exported_checkpoints)} ONNX models")

    def state_dict(self):
        """Return callback state."""
        return {"exported_checkpoints": self._exported_checkpoints}

    def load_state_dict(self, state_dict):
        """Load callback state."""
        self._exported_checkpoints = state_dict.get("exported_checkpoints", [])
