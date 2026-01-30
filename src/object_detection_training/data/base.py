"""
Base data module for object detection training.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import lightning as L
from torch.utils.data import DataLoader


class BaseDataModule(L.LightningDataModule):
    """
    Abstract base class for object detection data modules.

    All data modules should inherit from this class and implement
    the abstract methods for creating datasets.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initialize the data module.

        Args:
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            persistent_workers: Whether to keep workers alive between epochs.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        self.train_dataset: Any | None = None
        self.val_dataset: Any | None = None
        self.test_dataset: Any | None = None

    @abstractmethod
    def setup_train_dataset(self) -> Any:
        """Create and return the training dataset."""
        pass

    @abstractmethod
    def setup_val_dataset(self) -> Any:
        """Create and return the validation dataset."""
        pass

    def setup_test_dataset(self) -> Any | None:
        """Create and return the test dataset. Optional, returns None by default."""
        return None

    @abstractmethod
    def collate_fn(self, batch):
        """Custom collate function for batching."""
        pass

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for the given stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = self.setup_train_dataset()
            self.val_dataset = self.setup_val_dataset()

        if stage == "test" or stage is None:
            self.test_dataset = self.setup_test_dataset()

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader | None:
        """Return test data loader if test dataset exists."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )
