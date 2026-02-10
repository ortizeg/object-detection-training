"""Tests for random seed utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from object_detection_training.utils.seed import seed_everything


class TestSeedEverything:
    """Tests for seed_everything function."""

    def test_sets_python_random(self) -> None:
        """Python random module produces deterministic values after seeding."""
        seed_everything(42)
        a = random.random()  # noqa: S311
        seed_everything(42)
        b = random.random()  # noqa: S311
        assert a == b

    def test_sets_numpy_random(self) -> None:
        """Numpy random produces deterministic values after seeding."""
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_sets_torch_random(self) -> None:
        """PyTorch random produces deterministic values after seeding."""
        seed_everything(42)
        a = torch.rand(5)
        seed_everything(42)
        b = torch.rand(5)
        torch.testing.assert_close(a, b)

    def test_sets_python_hash_seed(self) -> None:
        """PYTHONHASHSEED environment variable is set."""
        seed_everything(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_sets_cudnn_flags(self) -> None:
        """cuDNN deterministic flags are set."""
        seed_everything(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_different_seeds_produce_different_values(self) -> None:
        """Different seeds produce different random values."""
        seed_everything(42)
        a = torch.rand(5)
        seed_everything(99)
        b = torch.rand(5)
        assert not torch.equal(a, b)
