"""
COCO format data module for object detection training.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from object_detection_training.data.base import BaseDataModule
from object_detection_training.rfdetr.coco import (
    CocoDetection,
    collate_fn,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from object_detection_training.utils.hydra import register


@register(group="data")
class COCODataModule(BaseDataModule):
    """
    COCO format data module with RFDETR augmentations.

    Expects data in COCO format with the following structure:
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
            square_resize_div_64: Use square resize divisible by 64 (recommended for
                RFDETR).
        """

        self.train_path = Path(train_path)
        self.val_path = Path(val_path)

        # Force num_workers=0 on MPS to avoid multiprocessing crash
        import torch

        if torch.backends.mps.is_available() and num_workers > 0:
            logger.warning(
                f"MPS detected, setting num_workers=0 (was {num_workers}) "
                "to avoid 'share_filename_cpu' error."
            )
            num_workers = 0

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        self.test_path = Path(test_path) if test_path else None

        self.input_height = input_height
        self.input_width = input_width
        self.multi_scale = multi_scale
        self.expanded_scales = expanded_scales
        self.skip_random_resize = skip_random_resize
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.square_resize_div_64 = square_resize_div_64

        # Cache for num_classes and mapping
        self._num_classes: Optional[int] = None
        self._class_names: Optional[list] = None
        self._label_map: Optional[Dict[int, int]] = None

        self.save_hyperparameters()

    @property
    def num_classes(self) -> int:
        """Get number of classes from COCO annotations."""
        if self._num_classes is None:
            self._load_class_info()
        return self._num_classes

    @property
    def class_names(self) -> list:
        """Get class names from COCO annotations."""
        if self._class_names is None:
            self._load_class_info()
        return self._class_names

    @property
    def label_map(self) -> Dict[int, int]:
        """Get mapping from COCO category_id to 0-indexed labels."""
        if self._label_map is None:
            self._load_class_info()
        return self._label_map

    def _load_class_info(self):
        """Load class information from COCO annotations."""
        _, ann_file = self._get_img_folder_and_ann_file(self.train_path)
        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        categories = sorted(coco_data.get("categories", []), key=lambda x: x["id"])
        self._num_classes = len(categories)
        self._class_names = [
            cat.get("name", f"class_{cat['id']}") for cat in categories
        ]
        # Map original category_id to 0..N-1
        self._label_map = {cat["id"]: i for i, cat in enumerate(categories)}

        logger.info(f"Detected {self._num_classes} classes: {self._class_names}")
        logger.debug(f"Label map: {self._label_map}")

    def _get_img_folder_and_ann_file(self, data_path: Path):
        """Get image folder and annotation file from data path."""
        # Try Roboflow format first
        ann_file = data_path / "_annotations.coco.json"
        if ann_file.exists():
            return data_path, ann_file

        # Try standard COCO format
        ann_file = data_path / "annotations.json"
        if ann_file.exists():
            return data_path, ann_file

        # Try images subdirectory
        img_folder = data_path / "images"
        if img_folder.exists():
            ann_file = data_path / "_annotations.coco.json"
            if ann_file.exists():
                return img_folder, ann_file
            ann_file = data_path / "annotations.json"
            if ann_file.exists():
                return img_folder, ann_file

        raise FileNotFoundError(
            f"Could not find annotation file in {data_path}. "
            "Expected '_annotations.coco.json' or 'annotations.json'"
        )

    def _get_transforms(self, image_set: str):
        """Get transforms based on configuration."""
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
            )

    def setup_train_dataset(self) -> Any:
        """Create training dataset with augmentations."""
        logger.info(f"Setting up training dataset from {self.train_path}")

        img_folder, ann_file = self._get_img_folder_and_ann_file(self.train_path)
        transforms = self._get_transforms("train")
        dataset = CocoDetection(
            img_folder, ann_file, transforms=transforms, label_map=self.label_map
        )
        logger.info(f"Training dataset: {len(dataset)} images")
        return dataset

    def setup_val_dataset(self) -> Any:
        """Create validation dataset."""
        logger.info(f"Setting up validation dataset from {self.val_path}")

        img_folder, ann_file = self._get_img_folder_and_ann_file(self.val_path)
        # Use simple transforms for validation (no multi-scale)
        transforms = self._get_transforms("val")
        dataset = CocoDetection(
            img_folder, ann_file, transforms=transforms, label_map=self.label_map
        )
        logger.info(f"Validation dataset: {len(dataset)} images")
        return dataset

    def setup_test_dataset(self) -> Optional[Any]:
        """Create test dataset if path is provided."""
        if self.test_path is None:
            logger.info("No test path provided, skipping test dataset setup")
            return None

        logger.info(f"Setting up test dataset from {self.test_path}")

        img_folder, ann_file = self._get_img_folder_and_ann_file(self.test_path)
        transforms = self._get_transforms("test")
        dataset = CocoDetection(
            img_folder, ann_file, transforms=transforms, label_map=self.label_map
        )
        logger.info(f"Test dataset: {len(dataset)} images")
        return dataset

    def collate_fn(self, batch):
        """Use RFDETR collate function."""
        return collate_fn(batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer batch to device, handling NestedTensor."""
        batch_images, batch_targets = batch

        # Handle NestedTensor or Tensor
        if hasattr(batch_images, "to"):
            batch_images = batch_images.to(device)

        # Handle targets (tuple of dicts)
        new_targets = []
        for t in batch_targets:
            new_t = {}
            for k, v in t.items():
                if hasattr(v, "to"):
                    new_t[k] = v.to(device)
                else:
                    new_t[k] = v
            new_targets.append(new_t)

        return batch_images, tuple(new_targets)
