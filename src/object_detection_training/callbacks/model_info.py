"""
Model info callback for PyTorch Lightning.

Computes and logs model statistics at training start.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table


class ModelInfoCallback(L.Callback):
    """
    Callback to compute and save model information.

    Computes parameters, FLOPs, model size, and inference speed at training start.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        output_filename: str = "model_info.json",
        input_height: int = 640,
        input_width: int = 640,
        measure_inference_speed: bool = True,
    ):
        """
        Initialize model info callback.

        Args:
            output_dir: Directory to save the model info JSON.
                        If None, uses trainer.log_dir.
            output_filename: Filename for the model info JSON.
            input_height: Input image height for FLOPs computation.
            input_width: Input image width for FLOPs computation.
            measure_inference_speed: Whether to measure inference speed.
        """
        super().__init__()
        self.output_dir = Path(output_dir) if output_dir else None
        self.output_filename = output_filename
        self.input_height = input_height
        self.input_width = input_width
        self.measure_inference_speed = measure_inference_speed
        self.model_info: dict[str, Any] = {}

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute and save model info at training start."""
        logger.info("Computing model statistics...")

        if hasattr(pl_module, "compute_model_stats"):
            try:
                self.model_info = pl_module.compute_model_stats(
                    input_height=self.input_height,
                    input_width=self.input_width,
                )
            except Exception as e:
                logger.warning(f"Failed to compute model stats: {e}")
                self.model_info = self._compute_basic_stats(pl_module)
        else:
            self.model_info = self._compute_basic_stats(pl_module)

        # Add additional info
        self.model_info["model_class"] = type(pl_module).__name__
        self.model_info["num_classes"] = getattr(pl_module, "num_classes", None)

        # Save to file
        if self.output_dir:
            save_dir = self.output_dir
        elif trainer.log_dir:
            save_dir = Path(trainer.log_dir)
        else:
            save_dir = Path(".")

        output_path = save_dir / self.output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.model_info, f, indent=2)

        logger.info(f"Model info saved to {output_path}")

        # Export labels mapping alongside model_info.json
        self._export_labels_mapping(trainer, save_dir)

        console = Console()
        table = Table(
            title="Model Information",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for k, v in self.model_info.items():
            # Format value for better readability
            if k in ["total_params", "trainable_params"] and isinstance(
                v, (int, float)
            ):
                value_str = f"{v / 1e6:.2f} M"
            elif k == "flops" and isinstance(v, (int, float)):
                if v >= 1e9:
                    value_str = f"{v / 1e9:.2f} G"
                else:
                    value_str = f"{v / 1e6:.2f} M"
            elif k == "model_size_mb" and isinstance(v, (int, float)):
                value_str = f"{v:.2f} MB"
            elif k == "inference_time_ms" and isinstance(v, (int, float)):
                value_str = f"{v:.2f} ms"
            elif k == "fps" and isinstance(v, (int, float)):
                value_str = f"{v:.2f} FPS"
            elif isinstance(v, float):
                value_str = f"{v:,.2f}"
            elif isinstance(v, int):
                value_str = f"{v:,}"
            else:
                value_str = str(v)
            table.add_row(k, value_str)

        logger.info("Model Info:")
        console.print(table)

        # Log to trainer loggers
        for logger_inst in trainer.loggers:
            # WandB: Log as config/metadata
            if (
                hasattr(logger_inst, "experiment")
                and hasattr(logger_inst.experiment, "config")
                and hasattr(logger_inst.experiment.config, "update")
            ):
                # Update config with model info
                # We use the prefix "model_info" to group them
                config_update = {
                    f"model_info/{k}": v for k, v in self.model_info.items()
                }
                # Allow update even if config was set
                logger_inst.experiment.config.update(
                    config_update, allow_val_change=True
                )

            # Others: Log as hyperparams
            elif hasattr(logger_inst, "log_hyperparams"):
                logger_inst.log_hyperparams(self.model_info)

    def _compute_basic_stats(self, pl_module: L.LightningModule) -> dict[str, Any]:
        """Compute basic model statistics."""
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )

        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in pl_module.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in pl_module.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "input_shape": [1, 3, self.input_height, self.input_width],
        }

    def state_dict(self):
        """Return callback state."""
        return {"model_info": self.model_info}

    def load_state_dict(self, state_dict):
        """Load callback state."""
        self.model_info = state_dict.get("model_info", {})

    def _export_labels_mapping(self, trainer: L.Trainer, save_dir: Path) -> None:
        """Export labels mapping JSON from the datamodule.

        Creates a labels_mapping.json file that maps contiguous label IDs
        to class names, useful for interpreting model predictions.

        Args:
            trainer: The Lightning trainer instance.
            save_dir: Directory to save the labels mapping file.
        """
        try:
            datamodule = trainer.datamodule

            # Try using new train_detection_dataset if available
            if hasattr(datamodule, "train_detection_dataset"):
                detection_dataset = datamodule.train_detection_dataset
                if detection_dataset is not None:
                    labels_path = save_dir / "labels_mapping.json"
                    detection_dataset.export_labels_mapping(labels_path)
                    logger.info(f"Labels mapping saved to {labels_path}")
                    return

            # Fallback: use class_names from datamodule directly
            if hasattr(datamodule, "class_names"):
                # Force to list to avoid OmegaConf serialization issues
                class_names = list(datamodule.class_names)
                labels_mapping = {str(i): name for i, name in enumerate(class_names)}

                labels_path = save_dir / "labels_mapping.json"
                with open(labels_path, "w") as f:
                    json.dump(labels_mapping, f, indent=2)
                logger.info(f"Labels mapping saved to {labels_path}")

        except Exception as e:
            logger.warning(f"Failed to export labels mapping: {e}")
