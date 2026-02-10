"""Shared test fixtures for the object detection training framework."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from object_detection_training.types import DetectionTarget


@pytest.fixture()
def sample_image() -> Image.Image:
    """Create a 640x480 RGB PIL image with random pixels."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def sample_image_tensor() -> torch.Tensor:
    """Create a [3, 480, 640] float32 image tensor in [0, 1]."""
    gen = torch.Generator().manual_seed(42)
    return torch.rand(3, 480, 640, generator=gen)


@pytest.fixture()
def sample_boxes() -> torch.Tensor:
    """Create sample XYXY bounding boxes (absolute pixel coords)."""
    return torch.tensor(
        [
            [10.0, 20.0, 100.0, 200.0],
            [50.0, 60.0, 150.0, 250.0],
            [200.0, 100.0, 400.0, 300.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture()
def sample_target(sample_boxes: torch.Tensor) -> DetectionTarget:
    """Create a DetectionTarget TypedDict."""
    return {
        "boxes": sample_boxes,
        "labels": torch.tensor([0, 1, 0], dtype=torch.int64),
        "image_id": torch.tensor([1]),
        "area": torch.tensor([16200.0, 19000.0, 40000.0]),
        "iscrowd": torch.zeros(3, dtype=torch.int64),
        "orig_size": torch.tensor([480, 640]),
        "size": torch.tensor([480, 640]),
    }


@pytest.fixture()
def mock_trainer() -> MagicMock:
    """Create a MagicMock Lightning Trainer."""
    trainer = MagicMock()
    trainer.log_dir = str(Path(__file__).parent / "test_logs")
    trainer.current_epoch = 0
    trainer.loggers = []
    trainer.callback_metrics = {}
    trainer.datamodule = None
    trainer.checkpoint_callback = None
    return trainer


@pytest.fixture()
def mock_pl_module() -> MagicMock:
    """Create a MagicMock LightningModule with basic model attributes."""
    module = MagicMock()

    # Simulate a simple linear layer for state_dict / parameters
    linear = torch.nn.Linear(10, 5)
    module.state_dict.return_value = linear.state_dict()
    module.parameters.return_value = list(linear.parameters())
    module.buffers.return_value = list(linear.buffers())
    module.load_state_dict = MagicMock()

    return module
