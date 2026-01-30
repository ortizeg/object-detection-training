"""
Task abstractions using Pydantic for configuration validation.

This module provides the base task interface and concrete task implementations
for the object detection training framework.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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

    model_config = {"extra": "forbid"}

    @abstractmethod
    def run(self) -> Any:
        """Execute the task."""
        pass

    def __call__(self) -> Any:
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
    model: Any = Field(description="Model configuration (instantiated via Hydra)")

    # Data configuration
    data: Any = Field(description="DataModule configuration (instantiated via Hydra)")

    # Trainer configuration
    trainer: Any = Field(
        description="PyTorch Lightning Trainer configuration (instantiated via Hydra)"
    )

    # Callbacks
    callbacks: list | None = Field(
        default=None, description="List of callbacks (instantiated via Hydra)"
    )

    # Loggers
    loggers: list | None = Field(
        default=None, description="List of loggers (instantiated via Hydra)"
    )

    # Additional training options
    ckpt_path: Path | None = Field(
        default=None, description="Path to checkpoint to resume from"
    )
    seed: int | None = Field(default=42, description="Random seed for training")

    @model_validator(mode="after")
    def validate_task(self) -> "TrainTask":
        """Validate task configuration."""
        # Ensure output_dir is set
        if self.output_dir is None:
            self.output_dir = Path("outputs")
        return self

    def run(self) -> Any:
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Setup trainer with callbacks and loggers
        trainer_kwargs = {}
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
            trainer = L.Trainer(**{**self.trainer, **trainer_kwargs})

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
                trainer.checkpoint_callback.best_model_path
                if trainer.checkpoint_callback
                else None
            ),
            "output_dir": str(self.output_dir),
        }
