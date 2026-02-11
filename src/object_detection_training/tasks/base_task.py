"""
Abstract Base Task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field


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
