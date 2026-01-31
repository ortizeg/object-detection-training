"""Mosaic and MixUp augmentation for YOLOX training.

Implements the standard mosaic augmentation (combining 4 images into one) and
optional MixUp blending to improve model generalization and detection at
various scales.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image

from object_detection_training.data.detection_dataset import DetectionDataset


class MosaicMixupDataset(torch.utils.data.Dataset[tuple[Any, dict[str, Any]]]):
    """Dataset wrapper that applies Mosaic and optional MixUp augmentation.

    Mosaic combines 4 random images into a single training image by placing
    them in quadrants around a random center point. This forces the model to
    learn objects at different scales and in different contexts.

    MixUp blends the mosaic result with another random image using alpha
    blending, adding further regularization.

    The base dataset must have ``transforms=None`` so this wrapper operates
    on raw PIL images with pixel xyxy bounding boxes.
    """

    def __init__(
        self,
        dataset: DetectionDataset,
        input_height: int = 640,
        input_width: int = 640,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.3,
        post_transforms: Any | None = None,
    ):
        """Initialize Mosaic + MixUp dataset wrapper.

        Args:
            dataset: Base detection dataset (transforms should be None).
            input_height: Target canvas height.
            input_width: Target canvas width.
            mosaic_prob: Probability of applying mosaic (vs single image).
            mixup_prob: Probability of applying MixUp after mosaic.
            post_transforms: Transforms to apply after mosaic/mixup
                (e.g. HFlip, ColorJitter, PILToTensor, RandomErasing).
        """
        self.dataset = dataset
        self.input_height = input_height
        self.input_width = input_width
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.post_transforms = post_transforms
        self.enabled = True

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        if not self.enabled or random.random() > self.mosaic_prob:  # noqa: S311
            return self._get_single(idx)
        return self._get_mosaic(idx)

    def _get_single(self, idx: int) -> tuple[Any, dict[str, Any]]:
        """Get a single image, resized to input size."""
        img, target = self.dataset[idx]
        orig_w, orig_h = img.size

        img_resized = img.resize((self.input_width, self.input_height), Image.BILINEAR)

        target = target.copy()
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= self.input_width / orig_w
            boxes[:, [1, 3]] *= self.input_height / orig_h
            target["boxes"] = boxes

        target["size"] = torch.tensor([self.input_height, self.input_width])

        if self.post_transforms is not None:
            img_resized, target = self.post_transforms(img_resized, target)

        return img_resized, target

    def _get_mosaic(self, idx: int) -> tuple[Any, dict[str, Any]]:
        """Create a 4-image mosaic with optional MixUp."""
        n = len(self.dataset)
        indices = [idx] + [random.randint(0, n - 1) for _ in range(3)]  # noqa: S311

        # Random center point with margin so each quadrant is meaningful
        cx = int(
            random.uniform(  # noqa: S311
                self.input_width * 0.25, self.input_width * 0.75
            )
        )
        cy = int(
            random.uniform(  # noqa: S311
                self.input_height * 0.25, self.input_height * 0.75
            )
        )

        # Gray fill (YOLOX default)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)

        all_boxes: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        # Quadrant definitions: (x_offset, y_offset, quad_width, quad_height)
        quadrants = [
            (0, 0, cx, cy),
            (cx, 0, self.input_width - cx, cy),
            (0, cy, cx, self.input_height - cy),
            (cx, cy, self.input_width - cx, self.input_height - cy),
        ]

        first_target: dict[str, Any] | None = None
        for i, (q_idx, (x_off, y_off, qw, qh)) in enumerate(
            zip(indices, quadrants, strict=True)
        ):
            if qw <= 0 or qh <= 0:
                continue

            img, target = self.dataset[q_idx]
            if i == 0:
                first_target = target

            orig_w, orig_h = img.size

            # Resize image to fill its quadrant
            resized = img.resize((qw, qh), Image.BILINEAR)
            canvas[y_off : y_off + qh, x_off : x_off + qw] = np.array(resized)

            # Scale and offset boxes to canvas coordinates
            boxes = target["boxes"]
            if boxes.numel() > 0:
                scale_x = qw / orig_w
                scale_y = qh / orig_h
                boxes = boxes.clone()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + x_off
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + y_off
                all_boxes.append(boxes)
                all_labels.append(target["labels"])

        # Combine all boxes from the 4 quadrants
        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            labels = torch.cat(all_labels, dim=0)

            # Clip to canvas boundaries
            boxes[:, 0].clamp_(0, self.input_width)
            boxes[:, 1].clamp_(0, self.input_height)
            boxes[:, 2].clamp_(0, self.input_width)
            boxes[:, 3].clamp_(0, self.input_height)

            # Filter degenerate boxes (< 2px)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]
            labels = labels[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Optional MixUp blending
        if self.mixup_prob > 0 and random.random() < self.mixup_prob:  # noqa: S311
            canvas, boxes, labels = self._apply_mixup(canvas, boxes, labels)

        # Build result
        result_img = Image.fromarray(canvas)

        result_target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": (
                first_target["image_id"]
                if first_target is not None
                else torch.tensor([0])
            ),
            "area": (
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                if boxes.numel() > 0
                else torch.zeros(0, dtype=torch.float32)
            ),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
            "orig_size": torch.tensor([self.input_height, self.input_width]),
            "size": torch.tensor([self.input_height, self.input_width]),
        }

        # Apply post-mosaic transforms (HFlip, ColorJitter, ToTensor, etc.)
        if self.post_transforms is not None:
            result_img, result_target = self.post_transforms(result_img, result_target)

        return result_img, result_target

    def _apply_mixup(
        self,
        canvas: np.ndarray,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Apply MixUp: alpha-blend canvas with a random image."""
        mix_idx = random.randint(0, len(self.dataset) - 1)  # noqa: S311
        mix_img, mix_target = self.dataset[mix_idx]
        orig_w, orig_h = mix_img.size

        mix_resized = mix_img.resize(
            (self.input_width, self.input_height), Image.BILINEAR
        )
        mix_arr = np.array(mix_resized)

        # Beta distribution — keep mosaic dominant (alpha >= 0.5)
        alpha = float(np.random.beta(1.5, 1.5))
        alpha = max(alpha, 1.0 - alpha)

        canvas = canvas.astype(np.float32) * alpha + mix_arr.astype(np.float32) * (
            1.0 - alpha
        )
        canvas = canvas.clip(0, 255).astype(np.uint8)

        # Add MixUp image's boxes (scaled to canvas size)
        mix_boxes = mix_target["boxes"]
        if mix_boxes.numel() > 0:
            mix_boxes = mix_boxes.clone()
            mix_boxes[:, [0, 2]] *= self.input_width / orig_w
            mix_boxes[:, [1, 3]] *= self.input_height / orig_h
            boxes = torch.cat([boxes, mix_boxes], dim=0)
            labels = torch.cat([labels, mix_target["labels"]], dim=0)

        return canvas, boxes, labels
