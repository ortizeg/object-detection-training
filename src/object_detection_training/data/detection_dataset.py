"""
Base detection dataset abstraction with Pydantic configuration.

Provides abstract base class for detection datasets with:
- Efficient pandas DataFrame storage for annotations
- Category filtering via selected_categories
- Computed metadata (area, max_dim, size_class)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel, Field, field_validator


class SizeThresholds(BaseModel):
    """Box size classification thresholds in original image pixels.

    Boxes are classified by max(width, height):
    - small: max_dim <= small_threshold
    - medium: small_threshold < max_dim <= medium_threshold
    - large: max_dim > medium_threshold
    """

    small: float = Field(
        default=32.0, description="Max dimension threshold for small boxes"
    )
    medium: float = Field(
        default=96.0, description="Max dimension threshold for medium boxes"
    )

    @field_validator("medium")
    @classmethod
    def medium_gt_small(cls, v: float, info) -> float:
        if "small" in info.data and v <= info.data["small"]:
            raise ValueError("medium threshold must be greater than small threshold")
        return v


class DetectionDataset(torch.utils.data.Dataset, ABC):
    """Base class for detection datasets with pandas DataFrame storage.

    Inherits from torch.utils.data.Dataset to provide standard indexing.
    The base class handles:
    - Category filtering based on selected_categories
    - Computing metadata columns (area, max_dim, size_class)
    - Label remapping to contiguous 0-indexed IDs
    - Standard __getitem__ output format for training
    """

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        selected_categories: list[str] | None = None,
        small_threshold: float = 32.0,
        medium_threshold: float = 96.0,
        label_map: dict[int, int] | None = None,
        class_names: list[str] | None = None,
        transforms: Any | None = None,
    ):
        """Initialize detection dataset.

        Args:
            root_path: Root path to dataset split directory.
            split: Dataset split ('train', 'val', 'test').
            selected_categories: Categories to keep.
            small_threshold: Threshold for small boxes.
            medium_threshold: Threshold for medium boxes.
            label_map: Optional externally provided label mapping (original_id ->
                contiguous_id).
            class_names: Optional externally provided class names.
            transforms: Optional transforms to apply to (image, target) pairs.
        """
        self.root_path = Path(root_path)
        self.split = split
        self.selected_categories = selected_categories
        self.size_thresholds = SizeThresholds(
            small=small_threshold, medium=medium_threshold
        )
        self.transforms = transforms

        # Lazy-loaded DataFrames
        self._images_df: pd.DataFrame | None = None
        self._annotations_df: pd.DataFrame | None = None
        self._categories: dict[int, str] | None = None

        # Mapping: original_dataset_id -> model_contiguous_id
        # Use provided mapping if available, otherwise compute during load
        self._label_map = label_map
        self._class_names = class_names

        # Cache for image IDs for indexing
        self._image_ids: list[int] | None = None

    @property
    def images_df(self) -> pd.DataFrame:
        """Image metadata DataFrame."""
        if self._images_df is None:
            self._load_and_process()
        return self._images_df

    @property
    def annotations_df(self) -> pd.DataFrame:
        """Detection annotations DataFrame with computed metadata."""
        if self._annotations_df is None:
            self._load_and_process()
        return self._annotations_df

    @property
    def categories(self) -> dict[int, str]:
        """Category ID to name mapping (after filtering)."""
        if self._categories is None:
            self._load_and_process()
        return self._categories

    @property
    def label_map(self) -> dict[int, int]:
        """Mapping from original category IDs to contiguous 0-indexed IDs."""
        if self._label_map is None:
            self._load_and_process()
        return self._label_map

    @property
    def num_classes(self) -> int:
        """Number of classes defined in configuration or found in dataset."""
        if self.selected_categories is not None:
            return len(self.selected_categories)
        if self._class_names is not None:
            return len(self._class_names)
        return len(self.categories)

    @property
    def class_names(self) -> list[str]:
        """Ordered list of class names (by contiguous ID)."""
        if self._class_names is not None:
            return self._class_names

        # Sort by contiguous ID and return names
        sorted_cats = sorted(
            [
                (self.label_map[orig_id], name)
                for orig_id, name in self.categories.items()
            ],
            key=lambda x: x[0],
        )
        return [name for _, name in sorted_cats]

    @property
    def image_ids(self) -> list[int]:
        """Ordered list of image IDs for indexing."""
        if self._image_ids is None:
            self._image_ids = self.images_df["image_id"].tolist()
        return self._image_ids

    @abstractmethod
    def load_annotations(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
        """Load raw annotations from the dataset format.

        Returns:
            Tuple of (images_df, annotations_df, categories_dict).
            - images_df: DataFrame with columns matching ImageSchema
            - annotations_df: DataFrame with raw bbox columns (before computed metadata)
            - categories_dict: Mapping of category_id -> category_name
        """
        pass

    @abstractmethod
    def _load_image(self, image_id: int) -> Image.Image:
        """Load an image by its ID.

        Args:
            image_id: Unique image ID.

        Returns:
            PIL Image in RGB format.
        """
        pass

    def _load_and_process(self) -> None:
        """Load annotations and apply filtering + metadata computation."""
        images_df, annotations_df, all_categories = self.load_annotations()

        # If label_map was NOT provided externally, compute it now
        if self._label_map is None:
            if self.selected_categories is not None:
                # Filter to only selected category names
                selected_set = set(self.selected_categories)
                self._categories = {
                    cid: name
                    for cid, name in all_categories.items()
                    if name in selected_set
                }

                # Map based on ORDER in selected_categories (Remapping)
                name_to_orig_id = {name: cid for cid, name in all_categories.items()}
                self._label_map = {}
                for idx, name in enumerate(self.selected_categories):
                    if name in name_to_orig_id:
                        orig_id = name_to_orig_id[name]
                        self._label_map[orig_id] = idx
            else:
                # Filter out categories with no annotations across ANY images in this
                # split
                # Note: This could cause divergence between splits if not externally
                # synced!
                used_cat_ids = set(annotations_df["category_id"].unique())
                self._categories = {
                    cid: name
                    for cid, name in all_categories.items()
                    if cid in used_cat_ids
                }
                # Create contiguous label mapping (0-indexed) sorted by ID
                sorted_cat_ids = sorted(self._categories.keys())
                self._label_map = {
                    orig_id: idx for idx, orig_id in enumerate(sorted_cat_ids)
                }
        else:
            # Use externally provided label_map to filter categories/annotations
            valid_orig_ids = set(self._label_map.keys())
            self._categories = {
                cid: name
                for cid, name in all_categories.items()
                if cid in valid_orig_ids
            }

        # Filter annotations to only include valid categories
        valid_cat_ids = set(self._categories.keys())
        annotations_df = annotations_df[
            annotations_df["category_id"].isin(valid_cat_ids)
        ].copy()

        # Add category name column
        annotations_df["category_name"] = annotations_df["category_id"].map(
            self._categories
        )

        # Compute metadata columns
        annotations_df = self._compute_metadata(annotations_df)

        # Do NOT filter images to only those with annotations (keep background images)
        self._images_df = images_df.copy()
        self._annotations_df = annotations_df

    def _compute_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns to annotations DataFrame.

        Computes:
        - area: bbox_w * bbox_h
        - max_dim: max(bbox_w, bbox_h)
        - size_class: 'small', 'medium', or 'large' based on thresholds
        """
        # Area (use existing if present, else compute)
        if "area" not in df.columns:
            df["area"] = df["bbox_w"] * df["bbox_h"]

        # Max dimension for size classification
        df["max_dim"] = df[["bbox_w", "bbox_h"]].max(axis=1)

        # Size class based on max dimension thresholds
        small_thresh = self.size_thresholds.small
        medium_thresh = self.size_thresholds.medium

        conditions = [
            df["max_dim"] <= small_thresh,
            df["max_dim"] <= medium_thresh,
        ]
        choices = ["small", "medium"]
        df["size_class"] = np.select(conditions, choices, default="large")

        return df

    def get_annotations_for_image(self, image_id: int) -> pd.DataFrame:
        """Get all annotations for a specific image."""
        return self.annotations_df[self.annotations_df["image_id"] == image_id]

    def get_image_info(self, image_id: int) -> pd.Series | None:
        """Get image metadata for a specific image."""
        matches = self.images_df[self.images_df["image_id"] == image_id]
        if len(matches) == 0:
            return None
        return matches.iloc[0]

    def add_computed_column(
        self, column_name: str, compute_fn: callable, **kwargs
    ) -> None:
        """Add a custom computed column to annotations_df.

        Args:
            column_name: Name for the new column.
            compute_fn: Function that takes the DataFrame and returns a Series.
            **kwargs: Additional arguments passed to compute_fn.

        Example:
            dataset.add_computed_column(
                "aspect_ratio",
                lambda df: df["bbox_w"] / df["bbox_h"]
            )
        """
        if self._annotations_df is None:
            self._load_and_process()
        self._annotations_df[column_name] = compute_fn(self._annotations_df, **kwargs)

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_ids)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', "
            f"images={len(self.images_df)}, "
            f"annotations={len(self.annotations_df)}, "
            f"classes={self.num_classes})"
        )

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        """Get image and target for a given index.

        Format matches DETR/RF-DETR expectations:
        - image: PIL Image or Tensor
        - target: dict with boxes, labels, image_id, area, iscrowd, orig_size, size

        Args:
            idx: Index into the images DataFrame.

        Returns:
            Tuple of (image, target_dict).
        """
        image_id = self.image_ids[idx]

        # Load image (delegated to subclass)
        img = self._load_image(image_id)
        w, h = img.size

        # Get annotations for this image
        anns_df = self.get_annotations_for_image(image_id)

        # Build target dict in COCO/DETR format
        # boxes in [x, y, w, h] format -> convert to [x1, y1, x2, y2]
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        label_map = self.label_map

        for _, ann in anns_df.iterrows():
            x, y, bw, bh = ann["bbox_x"], ann["bbox_y"], ann["bbox_w"], ann["bbox_h"]
            # Convert to x1, y1, x2, y2
            x2 = x + bw
            y2 = y + bh

            # Clamp to image boundaries
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Skip degenerate boxes
            if x2 <= x or y2 <= y:
                continue

            boxes.append([x, y, x2, y2])
            # Map to contiguous labels
            labels.append(label_map[ann["category_id"]])
            areas.append(ann["area"] if "area" in ann else bw * bh)
            iscrowd.append(1 if ann.get("is_crowd", False) else 0)

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([h, w]),
            "size": torch.as_tensor([h, w]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @property
    def labels_mapping(self) -> dict[int, str]:
        """Get mapping from contiguous label IDs to class names.

        This is the inverse of label_map: contiguous_id -> class_name.
        Useful for interpreting model outputs.
        """
        return dict(enumerate(self.class_names))

    def export_labels_mapping(
        self,
        save_path: Path,
        include_original_ids: bool = True,
    ) -> None:
        """Export labels mapping to JSON file.

        Creates a labels_mapping.json that maps model output indices to class names.
        Should be saved alongside model outputs (e.g., with model_info.json).

        Args:
            save_path: Path to save the JSON file.
            include_original_ids: If True, include original dataset category IDs.
        """
        mapping = {
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "id_to_name": {str(k): v for k, v in self.labels_mapping.items()},
        }

        if include_original_ids:
            # Add reverse mapping for reference
            mapping["name_to_original_id"] = {
                name: orig_id for orig_id, name in self.categories.items()
            }
            mapping["original_id_to_contiguous_id"] = {
                str(k): v for k, v in self.label_map.items()
            }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(mapping, f, indent=2)
