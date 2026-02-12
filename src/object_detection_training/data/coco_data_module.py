from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from loguru import logger
from torchvision.transforms import v2

from object_detection_training.data.cache_dataset import CacheDataset
from object_detection_training.data.coco_detection_dataset import COCODetectionDataset
from object_detection_training.models.rfdetr.collate import collate_fn
from object_detection_training.types import DetectionTarget
from object_detection_training.utils.hydra import register


@register
class COCODataModule(L.LightningDataModule):
    """Lightning DataModule for COCO dataset.

    Expected directory structure:
    - train_path/images/ and train_path/_annotations.coco.json
    - val_path/images/ and val_path/_annotations.coco.json
    - test_path/images/ and test_path/_annotations.coco.json (optional)

    Transforms are provided as pre-built ``v2.Compose`` pipelines via
    explicit parameters (``train_transforms``, ``val_transforms``,
    ``test_transforms``, ``post_mosaic_transforms``).  These are
    instantiated by Hydra from the YAML configs in ``conf/transforms/``
    (co-located in the package at ``object_detection_training/conf/``).
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        input_height: int = 640,
        input_width: int = 640,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        selected_categories: list[str] | None = None,
        size_thresholds: dict[str, float] | None = None,
        # -- v2 transform pipelines (from Hydra conf/transforms/*.yaml) --
        train_transforms: v2.Compose | None = None,
        val_transforms: v2.Compose | None = None,
        test_transforms: v2.Compose | None = None,
        post_mosaic_transforms: v2.Compose | None = None,
        # -- Mosaic / MixUp config (DataModule-level) --
        mosaic: dict[str, bool | float] | None = None,
        # -- Multi-scale params kept for Hydra ${data.*} interpolation --
        multi_scale: bool = False,
        expanded_scales: bool = False,
        skip_random_resize: bool = True,
        patch_size: int = 16,
        num_windows: int = 4,
        use_cache: bool = False,
        cache_type: Literal["ram", "disk", "auto"] = "disk",
    ):
        """Initialize COCO data module.

        Args:
            train_path: Path to training data directory.
            val_path: Path to validation data directory.
            test_path: Optional path to test data directory.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.
            input_height: Base input height (must be divisible by 64 for RFDETR).
            input_width: Base input width (must be divisible by 64 for RFDETR).
            pin_memory: Whether to pin memory for faster GPU transfer.
            persistent_workers: Whether to keep workers alive between epochs.
            selected_categories: Optional category names to keep.
            size_thresholds: Box size classification thresholds.
            train_transforms: v2.Compose pipeline for training augmentation.
            val_transforms: v2.Compose pipeline for validation preprocessing.
            test_transforms: v2.Compose pipeline for test preprocessing.
            post_mosaic_transforms: v2.Compose pipeline applied after mosaic/mixup.
            mosaic: Mosaic/MixUp config dict with ``enabled`` and ``mixup_prob``.
            multi_scale: Enable multi-scale augmentation (used by transform YAML refs).
            expanded_scales: Use expanded scale range (used by transform YAML refs).
            skip_random_resize: Skip random resize (used by transform YAML refs).
            patch_size: Patch size for multi-scale (used by transform YAML refs).
            num_windows: Windows for multi-scale (transform YAML refs).
            use_cache: Cache decoded images for faster loading.
            cache_type: Cache backend - 'ram', 'disk', or 'auto'.
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

        # v2 transform pipelines
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.post_mosaic_transforms = post_mosaic_transforms

        # Mosaic config
        mosaic = mosaic or {}
        self._mosaic_enabled: bool = bool(mosaic.get("enabled", False))
        self._mixup_prob: float = float(mosaic.get("mixup_prob", 0.3))

        # Multi-scale params (stored for Hydra ${data.*} interpolation only)
        self.multi_scale = multi_scale
        self.expanded_scales = expanded_scales
        self.skip_random_resize = skip_random_resize
        self.patch_size = patch_size
        self.num_windows = num_windows

        # Cache for num_classes and mapping
        self._num_classes: int | None = None
        self._class_names: list[str] | None = None
        self._label_map: dict[int, int] | None = None

        # Category filtering and size thresholds
        self.selected_categories = selected_categories
        self.size_thresholds = size_thresholds or {"small": 32, "medium": 96}

        # Caching
        self.use_cache = use_cache
        self.cache_type = cache_type

        # Lazy-loaded detection dataset for DataFrame access
        self._train_detection_dataset: COCODetectionDataset | None = None

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the training set."""
        if self._num_classes is None:
            self._load_metadata()
        if self._num_classes is None:
            raise RuntimeError("Failed to load num_classes from metadata")
        return self._num_classes

    @property
    def class_names(self) -> list[str]:
        """Returns the human-readable class names."""
        if self._class_names is None:
            self._load_metadata()
        if self._class_names is None:
            raise RuntimeError("Failed to load class_names from metadata")
        return self._class_names

    def _load_metadata(self) -> None:
        """Loads metadata from training annotation file."""
        ann_file = self.train_path / "_annotations.coco.json"
        if not ann_file.exists():
            # Fallback to images dir if shared structure
            ann_file = self.train_path / "annotations" / "instances_train2017.json"

        if not ann_file.exists():
            logger.error(f"Annotation file not found in {self.train_path}")
            self._num_classes = 1  # Default fallback
            self._class_names = ["object"]
            self._label_map = {}
            return

        with open(ann_file) as f:
            data = json.load(f)

        all_categories = data.get("categories", [])
        # Ensure classes are sorted by ID for consistency if not using
        # selected_categories
        all_categories.sort(key=lambda x: x["id"])

        # Apply category filtering if specified
        if self.selected_categories is not None:
            # Definitively use the order in selected_categories
            name_to_id = {cat["name"]: cat["id"] for cat in all_categories}
            self._class_names = self.selected_categories
            self._num_classes = len(self.selected_categories)
            self._label_map = {}
            for i, name in enumerate(self.selected_categories):
                if name in name_to_id:
                    self._label_map[name_to_id[name]] = i
                else:
                    logger.warning(f"Category '{name}' not found in training JSON.")
            logger.info(
                f"Using {self._num_classes} selected categories: {self._class_names}"
            )
        else:
            # Filter out supercategories with no annotations (like "basketball")
            annotations = data.get("annotations", [])
            used_cat_ids = {ann["category_id"] for ann in annotations}
            categories = [cat for cat in all_categories if cat["id"] in used_cat_ids]

            self._class_names = [cat["name"] for cat in categories]
            self._num_classes = len(categories)
            # Map original IDs to 0-indexed contiguous IDs
            self._label_map = {cat["id"]: i for i, cat in enumerate(categories)}
            logger.info(
                f"Discovered {self._num_classes} categories in training set: "
                f"{self._class_names}"
            )

    def _get_img_folder(self, path: Path) -> Path:
        """Helper to find image folder (either path itself or path/images)."""
        images_dir = path / "images"
        if images_dir.exists() and images_dir.is_dir():
            return images_dir
        return path

    def _create_detection_dataset(self, path: Path, split: str) -> COCODetectionDataset:
        """Create a COCODetectionDataset for the given path and split."""
        # Ensure metadata is loaded to provide consistent mapping across all splits
        if self._label_map is None:
            self._load_metadata()

        return COCODetectionDataset(
            root_path=str(path),
            split=split,
            selected_categories=self.selected_categories,
            small_threshold=self.size_thresholds.get("small", 32.0),
            medium_threshold=self.size_thresholds.get("medium", 96.0),
            label_map=self._label_map,
            class_names=self._class_names,
        )

    def setup_train_dataset(self) -> COCODetectionDataset:
        """Helper to create training dataset for visualization/stats."""
        if self._train_detection_dataset is None:
            self._train_detection_dataset = self._create_detection_dataset(
                self.train_path, "train"
            )
        # Apply transforms directly to the same dataset object
        self._train_detection_dataset.transforms = self.train_transforms
        return self._train_detection_dataset

    def train_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[tuple[torch.Tensor, DetectionTarget]]:
        """Return training data loader using new COCODetectionDataset."""
        if self._train_detection_dataset is None:
            self._train_detection_dataset = self._create_detection_dataset(
                self.train_path, "train"
            )
            # Update class info from the dataset
            self._num_classes = self._train_detection_dataset.num_classes
            self._class_names = list(self._train_detection_dataset.class_names)
            self._label_map = self._train_detection_dataset.label_map

        train_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, DetectionTarget]]
        if self._mosaic_enabled:
            # Mosaic operates on raw PIL images — base dataset has no transforms
            self._train_detection_dataset.transforms = None
            from object_detection_training.data.mosaic import MosaicMixupDataset

            train_dataset = MosaicMixupDataset(  # type: ignore[assignment]
                self._train_detection_dataset,
                input_height=self.input_height,
                input_width=self.input_width,
                mixup_prob=self._mixup_prob,
                post_transforms=self.post_mosaic_transforms,
            )
        elif self.use_cache:
            self._train_detection_dataset.transforms = None
            train_dataset = CacheDataset(  # type: ignore[assignment]
                self._train_detection_dataset,
                cache_type=self.cache_type,
                transforms=self.train_transforms,
            )
        else:
            self._train_detection_dataset.transforms = self.train_transforms
            train_dataset = self._train_detection_dataset  # type: ignore[assignment]

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[tuple[torch.Tensor, DetectionTarget]]:
        """Return validation data loader using new COCODetectionDataset."""
        val_dataset_raw = self._create_detection_dataset(self.val_path, "val")
        val_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, DetectionTarget]]
        if self.use_cache:
            val_dataset_raw.transforms = None
            val_dataset = CacheDataset(  # type: ignore[assignment]
                val_dataset_raw,
                cache_type=self.cache_type,
                transforms=self.val_transforms,
            )
        else:
            val_dataset_raw.transforms = self.val_transforms
            val_dataset = val_dataset_raw  # type: ignore[assignment]
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[tuple[torch.Tensor, DetectionTarget]] | None:
        """Return test data loader if test dataset exists."""
        if not self.test_path:
            return None

        test_dataset = self._create_detection_dataset(self.test_path, "test")
        test_dataset.transforms = self.test_transforms
        return torch.utils.data.DataLoader(
            test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @property
    def train_detection_dataset(self) -> COCODetectionDataset | None:
        """Expose training COCODetectionDataset for stats/sampling."""
        if self._train_detection_dataset is None:
            self._train_detection_dataset = self._create_detection_dataset(
                self.train_path, "train"
            )
        return self._train_detection_dataset

    def export_labels_mapping(self, save_path: Path) -> None:
        """Export labels mapping JSON for model outputs."""
        dataset = self.train_detection_dataset
        if dataset is None:
            raise RuntimeError("Training detection dataset not initialized")
        dataset.export_labels_mapping(save_path)
