"""Tests for the VisualizationCallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from object_detection_training.callbacks.visualization import VisualizationCallback


class TestVisualizationCallbackInit:
    """Tests for initialization."""

    def test_default_parameters(self) -> None:
        cb = VisualizationCallback()
        assert cb.num_samples == 10
        assert cb.confidence_threshold == 0.3
        assert cb.mean is None
        assert cb.std is None

    def test_custom_parameters(self) -> None:
        cb = VisualizationCallback(
            num_samples=5, confidence_threshold=0.5, output_dir="viz_out"
        )
        assert cb.num_samples == 5
        assert cb.confidence_threshold == 0.5


class TestVisualizationCallbackNormalization:
    """Tests for normalization resolution and image conversion."""

    def test_resolve_normalization_from_transforms(self) -> None:
        """Extracts mean/std from Normalize transform."""
        # Create a mock datamodule with transforms
        normalize_transform = MagicMock()
        normalize_transform.__class__ = type("Normalize", (), {"__name__": "Normalize"})
        normalize_transform.mean = [0.485, 0.456, 0.406]
        normalize_transform.std = [0.229, 0.224, 0.225]

        transforms = MagicMock()
        transforms.transforms = [normalize_transform]

        datamodule = MagicMock()
        datamodule.val_transforms = transforms

        cb = VisualizationCallback()
        cb._resolve_normalization(datamodule)

        assert cb.mean is not None
        assert cb.std is not None
        assert cb.mean.shape == (3, 1, 1)
        assert cb.std.shape == (3, 1, 1)

    def test_resolve_normalization_no_transforms(self) -> None:
        """Gracefully handles missing transforms."""
        datamodule = MagicMock()
        datamodule.val_transforms = None
        datamodule.train_transforms = None

        cb = VisualizationCallback()
        cb._resolve_normalization(datamodule)

        assert cb.mean is None
        assert cb.std is None
        assert cb._normalization_resolved is True

    def test_resolve_normalization_called_once(self) -> None:
        """Resolution is only done once (cached)."""
        datamodule = MagicMock()
        datamodule.val_transforms = None
        datamodule.train_transforms = None

        cb = VisualizationCallback()
        cb._resolve_normalization(datamodule)
        cb._resolve_normalization(datamodule)  # second call is a no-op

        assert cb._normalization_resolved is True


class TestVisualizationCallbackToDisplayImage:
    """Tests for _to_display_image conversion."""

    def test_unnormalized_image(self) -> None:
        """Tensor in [0, 255] range is converted to uint8."""
        cb = VisualizationCallback()
        tensor = torch.ones(3, 4, 4) * 128.0

        result = cb._to_display_image(tensor)

        assert result.dtype == np.uint8
        assert result.shape == (4, 4, 3)
        assert np.all(result == 128)

    def test_normalized_image(self) -> None:
        """Normalized image is denormalized then scaled to [0, 255]."""
        cb = VisualizationCallback()
        cb.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        cb.std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

        # A tensor of 0 after denormalization: 0 * 0.5 + 0.5 = 0.5 -> * 255 = 127.5
        tensor = torch.zeros(3, 4, 4)
        result = cb._to_display_image(tensor)

        assert result.dtype == np.uint8
        assert result.shape == (4, 4, 3)
        # Should be approximately 127 or 128
        assert np.all(result >= 127)
        assert np.all(result <= 128)

    def test_output_clipped_to_0_255(self) -> None:
        """Output values are clipped to [0, 255]."""
        cb = VisualizationCallback()
        # Very large and very small values
        tensor = torch.tensor([[[500.0]], [[-100.0]], [[200.0]]])  # [3, 1, 1]
        result = cb._to_display_image(tensor)

        assert result.min() >= 0
        assert result.max() <= 255


class TestVisualizationCallbackBoxNormalization:
    """Tests for _has_box_normalization static method."""

    def test_detects_normalize_box_coords(self) -> None:
        """Returns True when NormalizeBoxCoords is in transforms."""
        normalize_box_transform = MagicMock()
        normalize_box_transform.__class__ = type(
            "NormalizeBoxCoords", (), {"__name__": "NormalizeBoxCoords"}
        )

        transforms = MagicMock()
        transforms.transforms = [normalize_box_transform]

        datamodule = MagicMock()
        datamodule.val_transforms = transforms

        assert VisualizationCallback._has_box_normalization(datamodule) is True

    def test_no_normalize_box_coords(self) -> None:
        """Returns False when no NormalizeBoxCoords."""
        other_transform = MagicMock()
        other_transform.__class__ = type("Resize", (), {"__name__": "Resize"})

        transforms = MagicMock()
        transforms.transforms = [other_transform]

        datamodule = MagicMock()
        datamodule.val_transforms = transforms

        assert VisualizationCallback._has_box_normalization(datamodule) is False

    def test_no_transforms(self) -> None:
        """Returns False when no transforms exist."""
        datamodule = MagicMock()
        datamodule.val_transforms = None
        datamodule.train_transforms = None

        assert VisualizationCallback._has_box_normalization(datamodule) is False
