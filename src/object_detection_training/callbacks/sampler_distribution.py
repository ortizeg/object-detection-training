"""Callback to log sampler class distribution each epoch.

Compares the original dataset class distribution against the effective
distribution produced by weighted random sampling, making it easy to
verify that class balancing is working as expected.
"""

from __future__ import annotations

import lightning as L
import pandas as pd
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from object_detection_training.data.dataset_stats import DatasetStatistics
from object_detection_training.data.detection_dataset import DetectionDataset
from object_detection_training.data.sampler import TrackingWeightedRandomSampler


class SamplerDistributionCallback(L.Callback):
    """Log baseline vs effective class distribution each epoch.

    On fit start, computes the original class distribution from the full
    training set. At the end of each training epoch, reads the sampled
    indices from ``TrackingWeightedRandomSampler`` and computes the
    effective distribution. Both are printed as a comparison table and
    logged as per-class metrics.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stats: DatasetStatistics | None = None
        self._baseline: pd.DataFrame | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute and cache the baseline class distribution."""
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        dataset: DetectionDataset | None = getattr(
            datamodule, "train_detection_dataset", None
        )
        if dataset is None:
            return

        self._stats = DatasetStatistics(dataset)
        self._baseline = self._stats.class_distribution()
        logger.info("SamplerDistributionCallback: baseline distribution cached.")

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Compare effective vs baseline distribution and log metrics."""
        if self._stats is None or self._baseline is None:
            return

        # Access the sampler from the train dataloader
        train_dl = trainer.train_dataloader
        if train_dl is None:
            return

        sampler = getattr(train_dl, "sampler", None)
        if not isinstance(sampler, TrackingWeightedRandomSampler):
            return

        if not sampler._last_indices:
            return

        # Map dataset indices back to image IDs
        image_ids = self._stats.dataset.image_ids
        sampled_image_ids = [image_ids[i] for i in sampler._last_indices]

        effective = self._stats.class_distribution_for_images(sampled_image_ids)

        self._print_comparison_table(self._baseline, effective, trainer.current_epoch)
        self._log_metrics(self._baseline, effective, pl_module)

    def _print_comparison_table(
        self,
        baseline: pd.DataFrame,
        effective: pd.DataFrame,
        epoch: int,
    ) -> None:
        """Print a rich table comparing baseline vs effective distribution."""
        console = Console()
        table = Table(
            title=f"Sampler Distribution (Epoch {epoch})",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        table.add_column("Class", style="cyan")
        table.add_column("Baseline Count", justify="right")
        table.add_column("Baseline %", justify="right")
        table.add_column("Effective Count", justify="right")
        table.add_column("Effective %", justify="right")
        table.add_column("Ratio", justify="right", style="yellow")

        baseline_map = dict(
            zip(baseline["category_name"], baseline["percentage"], strict=True)
        )
        baseline_count_map = dict(
            zip(baseline["category_name"], baseline["count"], strict=True)
        )
        effective_map = dict(
            zip(effective["category_name"], effective["percentage"], strict=True)
        )
        effective_count_map = dict(
            zip(effective["category_name"], effective["count"], strict=True)
        )

        all_classes = sorted(
            set(baseline["category_name"]) | set(effective["category_name"])
        )

        for cls in all_classes:
            b_count = baseline_count_map.get(cls, 0)
            b_pct = baseline_map.get(cls, 0.0)
            e_count = effective_count_map.get(cls, 0)
            e_pct = effective_map.get(cls, 0.0)
            ratio = e_pct / b_pct if b_pct > 0 else 0.0

            table.add_row(
                cls,
                str(b_count),
                f"{b_pct:.1f}%",
                str(e_count),
                f"{e_pct:.1f}%",
                f"{ratio:.2f}x",
            )

        console.print(table)

    def _log_metrics(
        self,
        baseline: pd.DataFrame,
        effective: pd.DataFrame,
        pl_module: L.LightningModule,
    ) -> None:
        """Log per-class sampler metrics via pl_module.log()."""
        baseline_map = dict(
            zip(baseline["category_name"], baseline["percentage"], strict=True)
        )
        effective_map = dict(
            zip(effective["category_name"], effective["percentage"], strict=True)
        )

        for cls in effective["category_name"]:
            e_pct = effective_map.get(cls, 0.0)
            b_pct = baseline_map.get(cls, 0.0)
            ratio = e_pct / b_pct if b_pct > 0 else 0.0

            pl_module.log(f"sampler/{cls}_pct", e_pct, on_epoch=True, on_step=False)
            pl_module.log(f"sampler/{cls}_ratio", ratio, on_epoch=True, on_step=False)
