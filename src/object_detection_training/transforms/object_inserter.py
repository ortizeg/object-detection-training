"""ObjectInserter: paste masked object crops into training images.

Loads pre-extracted crops (from ``scripts/extract_object_crops.py``) and pastes
them into training images with random affine transforms, avoiding occlusion of
existing detections.  Designed to address class imbalance by inserting
underrepresented categories.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from object_detection_training.schemas.crop_manifest import (
    CropMetadata,
    decode_rle,
)
from object_detection_training.utils.boxes import box_iou_1_to_n


class ObjectInserter(v2.Transform):
    """Paste masked object crops into training images for data augmentation.

    Reads pre-extracted crop directories (one subfolder per category, each
    containing ``{annotation_id}.png`` + ``{annotation_id}.json`` pairs) and
    randomly inserts them into training images while respecting existing
    detections.

    Args:
        crops_dir: Path to the root directory of extracted crops.
        category_to_label: Mapping from category name to contiguous label ID.
        categories: Which categories to insert (``None`` = all available).
        p: Probability of attempting insertion per image.
        max_objects_per_image: Maximum number of objects to insert per image.
        iou_threshold: Maximum allowed IoU with existing boxes for placement.
        max_placement_retries: Number of random placement attempts per object.
        max_crop_ratio: Maximum crop size relative to image dimensions.
        default_rotation_range: Default (min, max) rotation in degrees.
        default_scale_range: Default (min, max) scale factor.
        category_configs: Per-category overrides for ``rotation_range`` and
            ``scale_range``.
    """

    def __init__(
        self,
        crops_dir: str,
        category_to_label: dict[str, int],
        categories: list[str] | None = None,
        p: float = 0.5,
        max_objects_per_image: int = 2,
        iou_threshold: float = 0.3,
        max_placement_retries: int = 50,
        max_crop_ratio: float = 0.4,
        default_rotation_range: tuple[float, float] = (-15.0, 15.0),
        default_scale_range: tuple[float, float] = (0.8, 1.2),
        category_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.max_objects_per_image = max_objects_per_image
        self.iou_threshold = iou_threshold
        self.max_placement_retries = max_placement_retries
        self.max_crop_ratio = max_crop_ratio
        self.category_to_label = category_to_label
        self.default_rotation_range = default_rotation_range
        self.default_scale_range = default_scale_range
        self.category_configs = category_configs or {}

        # Load all crops into memory
        self._crop_pool: dict[
            str, list[tuple[Image.Image, np.ndarray[Any, Any], CropMetadata]]
        ] = {}
        self._load_crops(Path(crops_dir), categories)

    def _load_crops(self, crops_dir: Path, categories: list[str] | None) -> None:
        """Scan category directories and load all crop data."""
        if not crops_dir.is_dir():
            logger.warning(f"Crops directory not found: {crops_dir}")
            return

        for cat_dir in sorted(crops_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_name = cat_dir.name
            if categories is not None and cat_name not in categories:
                continue
            if cat_name not in self.category_to_label:
                continue

            entries: list[tuple[Image.Image, np.ndarray[Any, Any], CropMetadata]] = []
            for json_path in sorted(cat_dir.glob("*.json")):
                png_path = json_path.with_suffix(".png")
                if not png_path.exists():
                    continue
                meta = CropMetadata.from_json(json_path)
                crop_img = Image.open(png_path).convert("RGB")
                mask = decode_rle(meta.mask)
                entries.append((crop_img, mask, meta))

            if entries:
                self._crop_pool[cat_name] = entries
                logger.debug(f"Loaded {len(entries)} crops for '{cat_name}'")

        total = sum(len(v) for v in self._crop_pool.values())
        logger.info(
            f"ObjectInserter: loaded {total} crops across "
            f"{len(self._crop_pool)} categories"
        )

    def forward(self, *inputs: Any) -> Any:
        """Apply object insertion augmentation.

        Expects ``(image, target)`` where *image* is a PIL Image and *target*
        is a dict with ``boxes`` (``tv_tensors.BoundingBoxes`` in XYXY),
        ``labels``, ``area``, and ``iscrowd`` tensors.
        """
        if len(inputs) < 2:
            return inputs if len(inputs) != 1 else inputs[0]

        image, target = inputs[0], inputs[1]
        rest = inputs[2:]

        # Only operate on PIL images
        if not isinstance(image, Image.Image):
            return inputs if rest else (image, target)

        # No crops loaded — passthrough
        if not self._crop_pool:
            return inputs if rest else (image, target)

        # Probabilistic gate
        if random.random() > self.p:  # noqa: S311
            return inputs if rest else (image, target)

        img_w, img_h = image.size
        image = image.copy()

        # Work with numpy for blending
        img_array = np.array(image)

        # Extract current boxes as list of [x1,y1,x2,y2] lists
        existing_boxes = target.get("boxes")
        if (
            existing_boxes is not None
            and isinstance(existing_boxes, torch.Tensor)
            and existing_boxes.numel() > 0
        ):
            # Ensure 2D shape (N, 4) before converting
            if existing_boxes.dim() == 1:
                existing_boxes = existing_boxes.unsqueeze(0)
            boxes_list: list[list[float]] = existing_boxes.tolist()
        else:
            boxes_list = []

        new_boxes: list[list[float]] = []
        new_labels: list[int] = []
        new_areas: list[float] = []

        available_cats = list(self._crop_pool.keys())
        n_insert = random.randint(1, self.max_objects_per_image)  # noqa: S311

        for _ in range(n_insert):
            cat_name = random.choice(available_cats)  # noqa: S311
            crop_img, crop_mask, _meta = random.choice(  # noqa: S311
                self._crop_pool[cat_name]
            )

            # Get per-category config
            cat_cfg = self.category_configs.get(cat_name, {})
            rot_range = cat_cfg.get("rotation_range", self.default_rotation_range)
            scale_range = cat_cfg.get("scale_range", self.default_scale_range)

            # Random affine
            scale = random.uniform(scale_range[0], scale_range[1])  # noqa: S311
            angle = random.uniform(rot_range[0], rot_range[1])  # noqa: S311

            t_crop, t_mask = self._apply_affine(crop_img, crop_mask, scale, angle)
            t_h, t_w = t_mask.shape[:2]

            # Cap crop size relative to image
            max_w = int(img_w * self.max_crop_ratio)
            max_h = int(img_h * self.max_crop_ratio)
            if t_w > max_w or t_h > max_h:
                downscale = min(max_w / t_w, max_h / t_h)
                t_crop, t_mask = self._apply_affine(
                    crop_img, crop_mask, scale * downscale, angle
                )
                t_h, t_w = t_mask.shape[:2]

            # All existing + already-inserted boxes
            all_boxes = boxes_list + new_boxes
            position = self._find_valid_position(t_w, t_h, img_w, img_h, all_boxes)
            if position is None:
                continue

            x, y = position
            img_array = self._paste_with_mask(img_array, t_crop, t_mask, x, y)

            # Record new box in XYXY format
            new_box = [float(x), float(y), float(x + t_w), float(y + t_h)]
            new_boxes.append(new_box)
            new_labels.append(self.category_to_label[cat_name])
            new_areas.append(float(t_w * t_h))

        # Rebuild image
        result_image = Image.fromarray(img_array)

        # Update target
        if new_boxes:
            target = dict(target)  # shallow copy

            all_box_list = boxes_list + new_boxes
            canvas_size = (img_h, img_w)
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.tensor(all_box_list, dtype=torch.float32),
                format="XYXY",
                canvas_size=canvas_size,
            )

            old_labels = target.get("labels", torch.tensor([], dtype=torch.int64))
            target["labels"] = torch.cat(
                [old_labels, torch.tensor(new_labels, dtype=torch.int64)]
            )

            old_area = target.get("area", torch.tensor([], dtype=torch.float32))
            target["area"] = torch.cat(
                [old_area, torch.tensor(new_areas, dtype=torch.float32)]
            )

            old_iscrowd = target.get(
                "iscrowd", torch.zeros(len(boxes_list), dtype=torch.int64)
            )
            target["iscrowd"] = torch.cat(
                [old_iscrowd, torch.zeros(len(new_boxes), dtype=torch.int64)]
            )

        if rest:
            return (result_image, target, *rest)
        return result_image, target

    @staticmethod
    def _apply_affine(
        crop: Image.Image,
        mask: np.ndarray[Any, Any],
        scale: float,
        angle: float,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Resize and rotate a crop and its mask.

        Returns:
            Tuple of (crop_array, mask_array) as numpy arrays.
        """
        w, h = crop.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized_crop = crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
        resized_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (new_w, new_h), Image.Resampling.NEAREST
        )

        if abs(angle) > 0.5:
            resized_crop = resized_crop.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                expand=True,
                fillcolor=(0, 0, 0),
            )
            resized_mask = resized_mask.rotate(
                angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0
            )

        crop_array = np.array(resized_crop)
        mask_array = np.array(resized_mask) > 127

        return crop_array, mask_array

    def _find_valid_position(
        self,
        crop_w: int,
        crop_h: int,
        img_w: int,
        img_h: int,
        existing_boxes: list[list[float]],
    ) -> tuple[int, int] | None:
        """Find a random position that doesn't overlap existing boxes too much."""
        if crop_w >= img_w or crop_h >= img_h:
            return None

        existing_tensor = (
            torch.tensor(existing_boxes, dtype=torch.float32)
            if existing_boxes
            else torch.zeros(0, 4)
        )

        for _ in range(self.max_placement_retries):
            x = random.randint(0, img_w - crop_w)  # noqa: S311
            y = random.randint(0, img_h - crop_h)  # noqa: S311

            if existing_tensor.numel() == 0:
                return (x, y)

            new_box = torch.tensor(
                [[x, y, x + crop_w, y + crop_h]], dtype=torch.float32
            )
            ious = box_iou_1_to_n(new_box, existing_tensor)
            if ious.max().item() < self.iou_threshold:
                return (x, y)

        return None

    @staticmethod
    def _paste_with_mask(
        img: np.ndarray[Any, Any],
        crop: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any],
        x: int,
        y: int,
    ) -> np.ndarray[Any, Any]:
        """Alpha-blend crop onto image using mask."""
        ch, cw = crop.shape[:2]
        ih, iw = img.shape[:2]

        # Clamp to image bounds
        x2 = min(x + cw, iw)
        y2 = min(y + ch, ih)
        cw_actual = x2 - x
        ch_actual = y2 - y

        if cw_actual <= 0 or ch_actual <= 0:
            return img

        region_mask = mask[:ch_actual, :cw_actual]
        if region_mask.ndim == 2:
            region_mask = region_mask[:, :, np.newaxis]

        img[y:y2, x:x2] = np.where(
            region_mask,
            crop[:ch_actual, :cw_actual],
            img[y:y2, x:x2],
        )
        return img
