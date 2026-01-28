"""
COCO format detection dataset implementation.

Efficiently loads COCO JSON annotations into pandas DataFrames without pycocotools.
Uses json.load() + pd.json_normalize() for fast parsing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from loguru import logger
from PIL import Image

from object_detection_training.data.detection_dataset import DetectionDataset
from object_detection_training.models.rfdetr.collate import collate_fn
from object_detection_training.utils.json_utils import load_json


class COCODetectionDataset(DetectionDataset):
    """COCO format dataset with efficient JSON to pandas loading.

    Expected directory structure:
    - root_path/_annotations.coco.json
    - root_path/images/ (or root_path/ if images are directly there)

    Does NOT require pycocotools.
    """

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        selected_categories: Optional[List[str]] = None,
        small_threshold: float = 32.0,
        medium_threshold: float = 96.0,
        label_map: Optional[Dict[int, int]] = None,
        class_names: Optional[List[str]] = None,
        transforms: Optional[Any] = None,
    ):
        """Initialize COCO detection dataset.

        Args:
            root_path: Root path to dataset split.
            split: Dataset split ('train', 'val', 'test').
            selected_categories: Categories to keep.
            small_threshold: Threshold for small boxes.
            medium_threshold: Threshold for medium boxes.
            label_map: Optional externally provided label mapping.
            class_names: Optional externally provided class names.
            transforms: Optional transforms.
        """
        super().__init__(
            root_path=root_path,
            split=split,
            selected_categories=selected_categories,
            small_threshold=small_threshold,
            medium_threshold=medium_threshold,
            label_map=label_map,
            class_names=class_names,
            transforms=transforms,
        )

        # Find annotation file
        self.ann_file = self._find_annotation_file()
        self.img_folder = self._find_img_folder()

    def _find_annotation_file(self) -> Path:
        """Locate the COCO annotation JSON file."""
        candidates = [
            self.root_path / "_annotations.coco.json",
            self.root_path / "annotations.json",
            self.root_path / f"instances_{self.split}2017.json",
            self.root_path / "annotations" / f"instances_{self.split}2017.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"No COCO annotation file found in {self.root_path}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def _find_img_folder(self) -> Path:
        """Locate the images folder."""
        images_dir = self.root_path / "images"
        if images_dir.exists() and images_dir.is_dir():
            return images_dir
        return self.root_path

    def load_annotations(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
        """Load COCO JSON into pandas DataFrames.

        Returns:
            Tuple of (images_df, annotations_df, categories_dict).
        """
        logger.info(f"Loading COCO annotations from {self.ann_file}")

        # Load JSON (fast with orjson if available)
        data = load_json(self.ann_file)

        # Parse categories
        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        logger.info(f"Found {len(categories)} categories")

        # Parse images into DataFrame
        images_list = data.get("images", [])
        if images_list:
            images_df = pd.DataFrame(images_list)
            # Rename columns to match schema
            images_df = images_df.rename(columns={"id": "image_id"})
            # Keep only required columns
            images_df = images_df[["image_id", "file_name", "width", "height"]].copy()
        else:
            images_df = pd.DataFrame(
                columns=["image_id", "file_name", "width", "height"]
            )

        logger.info(f"Loaded {len(images_df)} images")

        # Parse annotations into DataFrame
        annotations_list = data.get("annotations", [])
        if annotations_list:
            annotations_df = pd.DataFrame(annotations_list)

            # Rename columns to match schema
            annotations_df = annotations_df.rename(columns={"id": "annotation_id"})

            # Extract bbox components (COCO format: [x, y, width, height])
            if "bbox" in annotations_df.columns:
                bbox_df = pd.DataFrame(
                    annotations_df["bbox"].tolist(),
                    columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h"],
                )
                annotations_df = pd.concat(
                    [annotations_df.drop(columns=["bbox"]), bbox_df], axis=1
                )

            # Handle iscrowd
            if "iscrowd" in annotations_df.columns:
                annotations_df["is_crowd"] = annotations_df["iscrowd"].astype(bool)
                annotations_df = annotations_df.drop(columns=["iscrowd"])
            else:
                annotations_df["is_crowd"] = False

            # Keep only required columns
            required_cols = [
                "annotation_id",
                "image_id",
                "category_id",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "is_crowd",
            ]
            # Keep area if present
            if "area" in annotations_df.columns:
                required_cols.append("area")

            annotations_df = annotations_df[
                [c for c in required_cols if c in annotations_df.columns]
            ].copy()
        else:
            annotations_df = pd.DataFrame(
                columns=[
                    "annotation_id",
                    "image_id",
                    "category_id",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                    "is_crowd",
                ]
            )

        logger.info(f"Loaded {len(annotations_df)} annotations")

        return images_df, annotations_df, categories

    def get_image_path(self, image_id: int) -> Optional[Path]:
        """Get the full path to an image file."""
        info = self.get_image_info(image_id)
        if info is None:
            return None
        return self.img_folder / info["file_name"]

    def _load_image(self, image_id: int) -> Image.Image:
        """Load an image by its ID.

        Args:
            image_id: Unique image ID.

        Returns:
            PIL Image in RGB format.
        """
        img_path = self.get_image_path(image_id)
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")


def collate_fn_with_image_ids(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[Any, List[Dict[str, Any]], List[int]]:
    """Collate function that also returns image IDs.

    Useful for evaluation where image IDs are needed.
    Uses rfdetr's collate_fn internally for NestedTensor support.

    Args:
        batch: List of (image_tensor, target_dict) tuples.

    Returns:
        Tuple of (nested_tensor_images, list_of_targets, list_of_image_ids).
    """
    # Use rfdetr collate_fn for the images and targets
    samples, targets = collate_fn(batch)

    # Extract image IDs
    image_ids = []
    for target in targets:
        if "image_id" in target:
            image_ids.append(
                target["image_id"].item()
                if torch.is_tensor(target["image_id"])
                else target["image_id"]
            )
        else:
            image_ids.append(-1)

    return samples, targets, image_ids
