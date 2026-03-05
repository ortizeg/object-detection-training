"""Tests for online class-balanced sampler."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError
from torch.utils.data import WeightedRandomSampler

from object_detection_training.data.sampler import (
    SamplerConfig,
    TrackingWeightedRandomSampler,
    build_weighted_sampler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def class_names() -> list[str]:
    return ["ball", "player", "referee"]


@pytest.fixture()
def image_ids() -> list[int]:
    return [1, 2, 3, 4]


@pytest.fixture()
def annotations_df() -> pd.DataFrame:
    """Dataset where 'ball' is rare (1 ann), 'player' common (5 anns)."""
    return pd.DataFrame(
        {
            "image_id": [1, 2, 2, 3, 3, 3, 4],
            "category_name": [
                "ball",
                "player",
                "player",
                "player",
                "player",
                "player",
                "referee",
            ],
        }
    )


# ---------------------------------------------------------------------------
# TestSamplerConfig
# ---------------------------------------------------------------------------


class TestSamplerConfig:
    def test_default_disabled(self) -> None:
        config = SamplerConfig()
        assert config.mode == "disabled"
        assert config.class_weights is None
        assert config.num_samples is None
        assert config.replacement is True

    def test_auto_valid(self) -> None:
        config = SamplerConfig(mode="auto")
        assert config.mode == "auto"

    def test_manual_requires_weights(self) -> None:
        with pytest.raises(ValidationError, match="class_weights"):
            SamplerConfig(mode="manual")

    def test_manual_with_weights(self) -> None:
        config = SamplerConfig(
            mode="manual", class_weights={"ball": 5.0, "player": 1.0}
        )
        assert config.class_weights == {"ball": 5.0, "player": 1.0}

    def test_frozen(self) -> None:
        config = SamplerConfig()
        with pytest.raises(ValidationError):
            config.mode = "auto"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestBuildWeightedSampler
# ---------------------------------------------------------------------------


class TestBuildWeightedSampler:
    def test_disabled_returns_none(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(mode="disabled")
        result = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert result is None

    def test_auto_returns_tracking_sampler(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(mode="auto")
        result = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert isinstance(result, TrackingWeightedRandomSampler)

    def test_auto_rare_class_higher_weight(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(mode="auto")
        sampler = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert sampler is not None
        weights = list(sampler.weights)  # type: ignore[arg-type]
        # Image 1 has 'ball' (rare) — should have higher weight than
        # images 2,3 which have 'player' (common)
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_manual_uses_provided_weights(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(
            mode="manual",
            class_weights={"ball": 10.0, "player": 1.0, "referee": 5.0},
        )
        sampler = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert sampler is not None
        weights = list(sampler.weights)  # type: ignore[arg-type]
        # Image 1 has ball (10.0), image 4 has referee (5.0)
        assert weights[0] == 10.0
        assert weights[3] == 5.0

    def test_manual_missing_class_raises(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(
            mode="manual",
            class_weights={"ball": 5.0, "player": 1.0},
            # Missing 'referee'
        )
        with pytest.raises(ValueError, match="missing classes"):
            build_weighted_sampler(config, annotations_df, image_ids, class_names)

    def test_num_samples_default(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(mode="auto")
        sampler = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert sampler is not None
        assert sampler.num_samples == len(image_ids)

    def test_num_samples_override(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(mode="auto", num_samples=100)
        sampler = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert sampler is not None
        assert sampler.num_samples == 100

    def test_background_images_get_median_weight(
        self,
        class_names: list[str],
    ) -> None:
        """Images with no annotations should get the median weight."""
        annotations_df = pd.DataFrame(
            {
                "image_id": [1, 2],
                "category_name": ["ball", "player"],
            }
        )
        # Image 3 has no annotations (background)
        image_ids = [1, 2, 3]
        config = SamplerConfig(mode="auto")
        sampler = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert sampler is not None
        weights = list(sampler.weights)  # type: ignore[arg-type]
        # Background image weight should be median of the two annotated weights
        annotated_weights = sorted([weights[0], weights[1]])
        expected_median = (annotated_weights[0] + annotated_weights[1]) / 2
        assert weights[2] == pytest.approx(expected_median)


# ---------------------------------------------------------------------------
# TestTrackingWeightedRandomSampler
# ---------------------------------------------------------------------------


class TestTrackingWeightedRandomSampler:
    def test_records_indices(self) -> None:
        sampler = TrackingWeightedRandomSampler(
            weights=[1.0, 2.0, 3.0], num_samples=5, replacement=True
        )
        assert sampler._last_indices == []

        indices = list(sampler)
        assert len(indices) == 5
        assert sampler._last_indices == indices
        # All indices should be valid
        assert all(0 <= i < 3 for i in indices)

    def test_updates_each_iteration(self) -> None:
        sampler = TrackingWeightedRandomSampler(
            weights=[1.0, 1.0], num_samples=3, replacement=True
        )
        first = list(sampler)
        assert sampler._last_indices == first

        second = list(sampler)
        assert sampler._last_indices == second

    def test_is_subclass_of_weighted_random_sampler(self) -> None:
        sampler = TrackingWeightedRandomSampler(
            weights=[1.0], num_samples=1, replacement=True
        )
        assert isinstance(sampler, WeightedRandomSampler)

    def test_build_returns_tracking_sampler(
        self,
        annotations_df: pd.DataFrame,
        image_ids: list[int],
        class_names: list[str],
    ) -> None:
        config = SamplerConfig(mode="auto")
        sampler = build_weighted_sampler(config, annotations_df, image_ids, class_names)
        assert isinstance(sampler, TrackingWeightedRandomSampler)
        # Iterate to populate _last_indices
        indices = list(sampler)  # type: ignore[arg-type]
        assert len(indices) == len(image_ids)
        assert sampler._last_indices == indices
