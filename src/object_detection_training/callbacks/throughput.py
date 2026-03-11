"""Throughput logging callback for benchmarking data loading and training speed.

Logs per-epoch wall time, samples/sec, and batches/sec to both the console
(via loguru) and the active Lightning logger (W&B / TensorBoard).
"""

from __future__ import annotations

import time
from typing import Any

import lightning as L
from loguru import logger


class ThroughputCallback(L.Callback):
    """Measures and logs training throughput per epoch.

    Tracks:
    - Epoch wall time (seconds)
    - Samples per second (batch_size * batches / wall_time)
    - Batches per second
    - Data loading time vs compute time (via on_train_batch_start/end)

    Parameters
    ----------
    batch_size:
        Per-GPU batch size (used to compute samples/sec).
    world_size:
        Number of GPUs / processes. Defaults to 1.
    """

    def __init__(self, batch_size: int = 8, world_size: int = 1) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.world_size = world_size

        # Per-epoch accumulators
        self._epoch_start: float = 0.0
        self._batch_count: int = 0
        self._data_time_total: float = 0.0
        self._compute_time_total: float = 0.0
        self._batch_end_time: float = 0.0

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._epoch_start = time.perf_counter()
        self._batch_count = 0
        self._data_time_total = 0.0
        self._compute_time_total = 0.0
        self._batch_end_time = self._epoch_start

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Time between previous batch_end and this batch_start = data loading
        now = time.perf_counter()
        self._data_time_total += now - self._batch_end_time

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        now = time.perf_counter()
        self._batch_count += 1
        self._batch_end_time = now

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        epoch_time = time.perf_counter() - self._epoch_start
        if self._batch_count == 0:
            return

        total_samples = self._batch_count * self.batch_size * self.world_size
        samples_per_sec = total_samples / epoch_time
        batches_per_sec = self._batch_count / epoch_time
        data_pct = (self._data_time_total / epoch_time) * 100 if epoch_time > 0 else 0

        epoch = trainer.current_epoch

        logger.info(
            f"Epoch {epoch} throughput: "
            f"{samples_per_sec:.1f} samples/s, "
            f"{batches_per_sec:.1f} batches/s, "
            f"{epoch_time:.1f}s wall, "
            f"{self._batch_count} batches, "
            f"data_loading={data_pct:.1f}%"
        )

        # Log to Lightning logger (W&B / TensorBoard)
        if trainer.logger is not None:
            metrics = {
                "throughput/samples_per_sec": samples_per_sec,
                "throughput/batches_per_sec": batches_per_sec,
                "throughput/epoch_wall_sec": epoch_time,
                "throughput/data_loading_pct": data_pct,
                "throughput/total_batches": self._batch_count,
            }
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
