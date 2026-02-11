"""
Task abstractions using Pydantic for configuration validation.

This module provides the base task interface and concrete task implementations
for the object detection training framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import lightning as L
import omegaconf
import torch
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from object_detection_training.utils.hydra import register


class BaseTask(BaseModel, ABC):
    """
    Abstract base task with Pydantic validation.

    All tasks inherit from this class to define their configuration schema
    and execution logic.
    """

    name: str = Field(description="Name of the task")
    output_dir: Path | None = Field(
        default=None, description="Output directory for task artifacts"
    )

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    @abstractmethod
    def run(self) -> dict[str, str | None]:
        """Execute the task."""
        pass

    def __call__(self) -> dict[str, str | None]:
        """Allow tasks to be called directly."""
        logger.info(f"Running task: {self.name}")
        return self.run()


@register(group="task")
class TrainTask(BaseTask):
    """
    Training task for object detection models.

    Configures and runs PyTorch Lightning training with the specified
    model, data, and training parameters.
    """

    name: str = Field(default="train", description="Task name")

    # Model configuration
    model: L.LightningModule = Field(
        description="Model configuration (instantiated via Hydra)"
    )

    # Data configuration
    data: L.LightningDataModule = Field(
        description="DataModule configuration (instantiated via Hydra)"
    )

    # Trainer configuration
    trainer: L.Trainer | dict[str, object] = Field(
        description="PyTorch Lightning Trainer configuration (instantiated via Hydra)"
    )

    # Callbacks
    callbacks: list[L.Callback] | dict[str, L.Callback] | None = Field(
        default=None, description="List of callbacks (instantiated via Hydra)"
    )

    # Loggers
    loggers: list[L.pytorch.loggers.Logger] | None = Field(
        default=None, description="List of loggers (instantiated via Hydra)"
    )

    # Additional training options
    ckpt_path: Path | None = Field(
        default=None, description="Path to checkpoint to resume from"
    )
    seed: int | None = Field(default=42, description="Random seed for training")

    @model_validator(mode="after")
    def validate_task(self) -> TrainTask:
        """Validate task configuration."""
        # Ensure output_dir is set
        if self.output_dir is None:
            self.output_dir = Path("outputs")
        return self

    def run(self) -> dict[str, str | None]:
        """
        Execute the training task.

        Returns:
            Training results including metrics and checkpoint paths.
        """
        import lightning as L

        from object_detection_training.utils.seed import seed_everything

        # Set seed for reproducibility
        if self.seed is not None:
            seed_everything(self.seed)
            logger.info(f"Set random seed to {self.seed}")

        if (
            getattr(self.model, "download_pretrained", False)
            and getattr(self.model, "pretrain_weights", None) is None
        ):
            logger.info("Pretrained weights will be downloaded by the model wrapper.")

        # Ensure output directory exists
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Setup trainer with callbacks and loggers
        trainer_kwargs: dict[str, object] = {}
        if self.callbacks:
            logger.info(f"Callbacks type: {type(self.callbacks)}")
            if isinstance(self.callbacks, dict):
                logger.info("Callbacks is a dict, converting to list values.")
                self.callbacks = list(self.callbacks.values())
            logger.info(f"Callbacks list: {self.callbacks}")
            trainer_kwargs["callbacks"] = self.callbacks
        if self.loggers:
            trainer_kwargs["logger"] = self.loggers
        trainer_kwargs["default_root_dir"] = str(self.output_dir)

        # Merge trainer config with additional kwargs
        if isinstance(self.trainer, L.Trainer):
            trainer = self.trainer
        else:
            # Assume it's a config dict that needs instantiation
            trainer = L.Trainer(**{**self.trainer, **trainer_kwargs})  # type: ignore[arg-type]

        # Log model info
        logger.info(f"Model: {type(self.model).__name__}")
        logger.info(f"DataModule: {type(self.data).__name__}")

        # Run training
        logger.info("Starting training...")
        trainer.fit(self.model, datamodule=self.data, ckpt_path=self.ckpt_path)

        # Run test if test dataloader is available
        # We need to manually setup 'test' stage first to ensure test_dataset
        # is initialized. If the datamodule supports checking for test data
        # availability without setup, that's better, but BaseDataModule pattern
        # requires setup.
        try:
            self.data.setup("test")
            if self.data.test_dataloader() is not None:
                logger.info("Running test evaluation...")
                trainer.test(self.model, datamodule=self.data)
            else:
                logger.info("No test dataset provided, skipping test evaluation")
        except Exception as e:
            logger.warning(f"Could not run test evaluation: {e}")

        logger.info("Training completed successfully!")

        return {
            "best_model_path": (
                getattr(trainer.checkpoint_callback, "best_model_path", None)
                if trainer.checkpoint_callback
                else None
            ),
            "output_dir": str(self.output_dir),
        }


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
    input_height: int = Field(default=640, description="Input image height")
    input_width: int = Field(default=640, description="Input image width")

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

        # Export via BaseDetectionModel.export_onnx
        if not hasattr(self.model, "export_onnx"):
            msg = (
                f"Model {type(self.model).__name__} does not implement export_onnx. "
                "Only BaseDetectionModel subclasses are supported."
            )
            raise AttributeError(msg)

        onnx_path = self.model.export_onnx(  # type: ignore[operator]
            output_path=str(onnx_out),
            input_height=self.input_height,
            input_width=self.input_width,
            opset_version=self.opset_version,
            simplify=self.simplify,
        )
        logger.info(f"ONNX model exported to {onnx_path}")

        return {"onnx_path": onnx_path}
