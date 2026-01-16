import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from loguru import logger

from object_detection_training.rfdetr.coco import (
    collate_fn,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from object_detection_training.utils.hydra import register


@register(group="data", name="coco")
class COCODataModule(L.LightningDataModule):
    """
    Lightning DataModule for COCO dataset.

    Expected directory structure:
    - train_path/images/ and train_path/_annotations.coco.json
    - val_path/images/ and val_path/_annotations.coco.json
    - test_path/images/ and test_path/_annotations.coco.json (optional)
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        input_height: int = 640,
        input_width: int = 640,
        multi_scale: bool = False,
        expanded_scales: bool = False,
        skip_random_resize: bool = True,
        patch_size: int = 16,
        num_windows: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        square_resize_div_64: bool = True,
        image_mean: List[float] = [123.675, 116.28, 103.53],
        image_std: List[float] = [58.395, 57.12, 57.375],
    ):
        """
        Initialize COCO data module.

        Args:
            train_path: Path to training data directory.
            val_path: Path to validation data directory.
            test_path: Optional path to test data directory.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.
            input_height: Base input height (must be divisible by 64 for RFDETR).
            input_width: Base input width (must be divisible by 64 for RFDETR).
            multi_scale: Enable multi-scale augmentation.
            expanded_scales: Use expanded scale range.
            skip_random_resize: Skip random resize augmentation.
            patch_size: Patch size for multi-scale computation.
            num_windows: Number of windows for multi-scale computation.
            pin_memory: Whether to pin memory for faster GPU transfer.
            persistent_workers: Whether to keep workers alive between epochs.
            square_resize_div_64: Use square resize divisible by 64 for RFDETR.
            image_mean: Mean for image normalization (0-255 scale).
            image_std: Std for image normalization (0-255 scale).
        """
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)

        # Force num_workers=0 on MPS to avoid multiprocessing crash
        if torch.backends.mps.is_available() and num_workers > 0:
            logger.warning(
                f"MPS detected, setting num_workers=0 (was {num_workers}) "
                "to avoid 'share_filename_cpu' error."
            )
            num_workers = 0

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        self.test_path = Path(test_path) if test_path else None

        self.input_height = input_height
        self.input_width = input_width
        self.multi_scale = multi_scale
        self.expanded_scales = expanded_scales
        self.skip_random_resize = skip_random_resize
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.square_resize_div_64 = square_resize_div_64
        self.image_mean = image_mean
        self.image_std = image_std

        # Cache for num_classes and mapping
        self._num_classes: Optional[int] = None
        self._class_names: Optional[list] = None
        self._label_map: Optional[Dict[int, int]] = None

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the training set."""
        if self._num_classes is None:
            self._load_metadata()
        return self._num_classes

    @property
    def class_names(self) -> list:
        """Returns the human-readable class names."""
        if self._class_names is None:
            self._load_metadata()
        return self._class_names

    def _load_metadata(self):
        """Loads metadata from training annotation file."""
        ann_file = self.train_path / "_annotations.coco.json"
        if not ann_file.exists():
            # Fallback to images dir if shared structure
            ann_file = self.train_path / "annotations" / "instances_train2017.json"

        if not ann_file.exists():
            logger.error(f"Annotation file not found in {self.train_path}")
            self._num_classes = 1  # Default fallback
            self._class_names = ["object"]
            return

        with open(ann_file, "r") as f:
            data = json.load(f)

        categories = data["categories"]
        # Ensure classes are sorted by ID for consistency
        categories.sort(key=lambda x: x["id"])

        self._class_names = [cat["name"] for cat in categories]
        self._num_classes = len(categories)

        # Map original IDs to 0-indexed contiguous IDs
        self._label_map = {cat["id"]: i for i, cat in enumerate(categories)}

        logger.info(
            f"Loaded {self._num_classes} classes from {ann_file.name}: "
            f"{self._class_names}"
        )

    def _get_transforms(self, image_set: str):
        """Get transforms based on configuration and normalization parameters."""
        if self.square_resize_div_64:
            return make_coco_transforms_square_div_64(
                image_set,
                self.input_height,
                self.input_width,
                multi_scale=self.multi_scale,
                expanded_scales=self.expanded_scales,
                skip_random_resize=self.skip_random_resize,
                patch_size=self.patch_size,
                num_windows=self.num_windows,
                mean=self.image_mean,
                std=self.image_std,
            )
        else:
            return make_coco_transforms(
                image_set,
                self.input_height,
                self.input_width,
                multi_scale=self.multi_scale,
                expanded_scales=self.expanded_scales,
                skip_random_resize=self.skip_random_resize,
                patch_size=self.patch_size,
                num_windows=self.num_windows,
                mean=self.image_mean,
                std=self.image_std,
            )

    def _get_img_folder(self, path: Path) -> Path:
        """Helper to find image folder (either path itself or path/images)."""
        images_dir = path / "images"
        if images_dir.exists() and images_dir.is_dir():
            return images_dir
        return path

    def setup_train_dataset(self) -> Any:
        """Helper to create training dataset for visualization/stats."""
        from object_detection_training.rfdetr.coco import CocoDetection

        return CocoDetection(
            img_folder=self._get_img_folder(self.train_path),
            ann_file=self.train_path / "_annotations.coco.json",
            transforms=self._get_transforms("train"),
            include_masks=False,
            label_map=self._label_map,
        )

    def train_dataloader(self):
        from object_detection_training.rfdetr.coco import CocoDetection

        dataset = CocoDetection(
            img_folder=self._get_img_folder(self.train_path),
            ann_file=self.train_path / "_annotations.coco.json",
            transforms=self._get_transforms("train"),
            include_masks=False,
            label_map=self._label_map,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        from object_detection_training.rfdetr.coco import CocoDetection

        dataset = CocoDetection(
            img_folder=self._get_img_folder(self.val_path),
            ann_file=self.val_path / "_annotations.coco.json",
            transforms=self._get_transforms("val"),
            include_masks=False,
            label_map=self._label_map,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        if not self.test_path:
            return None

        from object_detection_training.rfdetr.coco import CocoDetection

        dataset = CocoDetection(
            img_folder=self._get_img_folder(self.test_path),
            ann_file=self.test_path / "_annotations.coco.json",
            transforms=self._get_transforms("test"),
            include_masks=False,
            label_map=self._label_map,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
