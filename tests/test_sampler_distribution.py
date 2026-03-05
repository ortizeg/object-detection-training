"""Tests for SamplerDistributionCallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from object_detection_training.callbacks.sampler_distribution import (
    SamplerDistributionCallback,
)
from object_detection_training.data.sampler import TrackingWeightedRandomSampler


@pytest.fixture()
def fake_dataset() -> MagicMock:
    """Minimal mock DetectionDataset for callback tests."""
    ds = MagicMock()
    ds.image_ids = [1, 2, 3]
    ds.class_names = ["ball", "player"]
    ds.split = "train"
    ds.annotations_df = pd.DataFrame(
        {
            "image_id": [1, 2, 2, 3],
            "category_name": ["ball", "player", "player", "player"],
        }
    )
    return ds


@pytest.fixture()
def callback() -> SamplerDistributionCallback:
    return SamplerDistributionCallback()


class TestSamplerDistributionCallbackOnFitStart:
    def test_caches_baseline(
        self, callback: SamplerDistributionCallback, fake_dataset: MagicMock
    ) -> None:
        trainer = MagicMock()
        trainer.datamodule.train_detection_dataset = fake_dataset
        pl_module = MagicMock()

        callback.on_fit_start(trainer, pl_module)

        assert callback._baseline is not None
        assert callback._stats is not None
        assert "category_name" in callback._baseline.columns

    def test_no_datamodule_is_noop(self, callback: SamplerDistributionCallback) -> None:
        trainer = MagicMock(spec=[])  # no datamodule attr
        trainer.datamodule = None
        pl_module = MagicMock()

        callback.on_fit_start(trainer, pl_module)

        assert callback._baseline is None

    def test_no_dataset_is_noop(self, callback: SamplerDistributionCallback) -> None:
        trainer = MagicMock()
        trainer.datamodule = MagicMock(spec=[])  # no train_detection_dataset
        pl_module = MagicMock()

        callback.on_fit_start(trainer, pl_module)

        assert callback._baseline is None


class TestSamplerDistributionCallbackOnTrainEpochEnd:
    def test_logs_metrics(
        self, callback: SamplerDistributionCallback, fake_dataset: MagicMock
    ) -> None:
        # Setup: run on_fit_start first
        trainer = MagicMock()
        trainer.datamodule.train_detection_dataset = fake_dataset
        pl_module = MagicMock()

        callback.on_fit_start(trainer, pl_module)

        # Create a tracking sampler with indices pointing to images
        sampler = TrackingWeightedRandomSampler(
            weights=[1.0, 1.0, 1.0], num_samples=3, replacement=True
        )
        # Simulate an epoch iteration that drew indices [0, 0, 1]
        # image_ids[0]=1 (ball), image_ids[1]=2 (player, player)
        sampler._last_indices = [0, 0, 1]

        trainer.train_dataloader = MagicMock()
        trainer.train_dataloader.sampler = sampler
        trainer.current_epoch = 0

        callback.on_train_epoch_end(trainer, pl_module)

        # Verify pl_module.log was called with sampler/ metrics
        log_calls = {call.args[0] for call in pl_module.log.call_args_list}
        assert "sampler/ball_pct" in log_calls
        assert "sampler/player_pct" in log_calls
        assert "sampler/ball_ratio" in log_calls
        assert "sampler/player_ratio" in log_calls

    def test_no_sampler_is_noop(
        self, callback: SamplerDistributionCallback, fake_dataset: MagicMock
    ) -> None:
        trainer = MagicMock()
        trainer.datamodule.train_detection_dataset = fake_dataset
        pl_module = MagicMock()

        callback.on_fit_start(trainer, pl_module)

        # No TrackingWeightedRandomSampler on the dataloader
        trainer.train_dataloader = MagicMock()
        trainer.train_dataloader.sampler = MagicMock()  # not a tracking sampler

        callback.on_train_epoch_end(trainer, pl_module)

        pl_module.log.assert_not_called()

    def test_skips_without_fit_start(
        self, callback: SamplerDistributionCallback
    ) -> None:
        trainer = MagicMock()
        pl_module = MagicMock()

        # on_fit_start never called — _stats and _baseline are None
        callback.on_train_epoch_end(trainer, pl_module)

        pl_module.log.assert_not_called()
