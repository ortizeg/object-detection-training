"""Mosaic and MixUp augmentation for YOLOX training.

Implements the standard mosaic augmentation (combining 4 images into one) and
optional MixUp blending to improve model generalization and detection at
various scales.

Supports **cached mode** (inspired by RTMDet) where companion images for
mosaic/mixup are sampled from an in-memory cache of recently loaded samples
instead of hitting the dataset (disk).  This reduces I/O from 6 reads per
sample to 1, giving a ~1.5-1.8x end-to-end training speedup with no
accuracy impact when the cache is large enough (default 40).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision import tv_tensors

from object_detection_training.data.detection_dataset import DetectionDataset
from object_detection_training.types import DetectionTarget

# Type alias for a cached sample (raw PIL image + target dict).
_CachedSample = tuple[Image.Image, DetectionTarget]

# Internal cache stores images as uint8 numpy arrays to avoid expensive
# copy.deepcopy on PIL Images.  On retrieval we wrap back into PIL (cheap)
# and shallow-copy the target dict — downstream code already clones tensors
# before mutation, so deep-copy is unnecessary.
_CachedArraySample = tuple[np.ndarray, DetectionTarget]


class MosaicMixupDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor | Image.Image, DetectionTarget]]
):
    """Dataset wrapper that applies Mosaic and optional MixUp augmentation.

    Mosaic combines 4 random images into a single training image by placing
    them in quadrants around a random center point. This forces the model to
    learn objects at different scales and in different contexts.

    MixUp blends the mosaic result with another random image using alpha
    blending, adding further regularization.

    When ``use_cache=True`` (default), companion images are drawn from a
    fixed-size in-memory queue instead of the underlying dataset, cutting
    disk reads from ~6 per sample to 1.  With ``max_cached_images=40`` and
    ``random_pop=True`` the sampling distribution is statistically
    equivalent to standard mosaic (RTMDet, Table 7a).

    The base dataset must have ``transforms=None`` so this wrapper operates
    on raw PIL images with pixel xyxy bounding boxes.
    """

    def __init__(
        self,
        dataset: DetectionDataset | torch.utils.data.Dataset[Any],
        input_height: int = 640,
        input_width: int = 640,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.3,
        post_transforms: Any | None = None,
        *,
        use_cache: bool = True,
        max_cached_images: int = 40,
        random_pop: bool = True,
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
            use_cache: When True, companion images for mosaic/mixup are
                sampled from an in-memory cache instead of the dataset.
            max_cached_images: Maximum number of samples kept in the cache.
                RTMDet uses 40 for mosaic.  Must be >= 4 for mosaic to work.
            random_pop: If True, evict a random entry when the cache is
                full (approximates uniform sampling).  If False, use FIFO
                eviction (acts like repeated augmentation for small caches).
        """
        self.dataset = dataset
        self.input_height = input_height
        self.input_width = input_width
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.post_transforms = post_transforms
        self.enabled = True

        # Cache config
        self.use_cache = use_cache
        self.max_cached_images = max(max_cached_images, 4)
        self.random_pop = random_pop
        self._cache: deque[_CachedArraySample] = deque(maxlen=self.max_cached_images)
        self._cache_logged = False

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        # Always load the current sample from the dataset (1 disk read).
        img, target = self.dataset[idx]

        # Update cache with the freshly loaded sample.
        if self.use_cache and self.enabled:
            self._push_cache(img, target)

        if not self.enabled or random.random() > self.mosaic_prob:  # noqa: S311
            return self._get_single_from_loaded(img, target)
        return self._get_mosaic(img, target)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _push_cache(self, img: Image.Image, target: DetectionTarget) -> None:
        """Add a sample to the cache, evicting if full.

        Converts PIL Image to a uint8 numpy array for storage — numpy arrays
        are contiguous memory and trivially cheap to copy compared to the
        Python object-graph traversal that ``copy.deepcopy`` performs on PIL
        Images.
        """
        if len(self._cache) >= self.max_cached_images and self.random_pop:
            # Random eviction — approximates uniform sampling over dataset
            pop_idx = random.randint(0, len(self._cache) - 1)  # noqa: S311
            del self._cache[pop_idx]
        # When random_pop=False, deque(maxlen=...) auto-evicts oldest (FIFO)
        # Deep-clone all tensors so cached entries are fully detached from the
        # dataset's memory.  Without this, forked dataloader workers that read
        # from the cache trigger copy-on-write page faults on shared tensor
        # pages, causing cumulative RAM growth that eventually OOMs.
        target_cloned: DetectionTarget = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in target.items()
        }
        self._cache.append((np.asarray(img, dtype=np.uint8).copy(), target_cloned))

    def _sample_from_cache(self) -> _CachedSample:
        """Return a sample from the cache, converting back to PIL.

        Instead of ``copy.deepcopy`` (which traverses the full Python object
        graph of a PIL Image — slow), we store uint8 numpy arrays and wrap
        them back into PIL on retrieval.  The target dict gets a shallow copy;
        downstream code already ``.clone()``s tensors before in-place mutation.
        """
        arr, target = random.choice(self._cache)  # noqa: S311
        # Clone tensors on retrieval so each consumer gets independent memory,
        # preventing COW page faults in forked dataloader workers.
        target_out: DetectionTarget = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in target.items()
        }
        return Image.fromarray(arr), target_out

    def _cache_ready(self) -> bool:
        """Cache needs at least 4 entries for mosaic."""
        return len(self._cache) >= 4

    # ------------------------------------------------------------------
    # Companion fetching (cache or dataset)
    # ------------------------------------------------------------------

    def _get_companion(self) -> _CachedSample:
        """Get a companion sample — from cache if ready, else dataset."""
        if self.use_cache and self._cache_ready():
            if not self._cache_logged:
                logger.info(
                    f"Mosaic cache active: {len(self._cache)}/{self.max_cached_images}"
                    " samples cached, serving companions from memory"
                )
                self._cache_logged = True
            return self._sample_from_cache()
        idx = random.randint(0, len(self.dataset) - 1)  # noqa: S311
        return self.dataset[idx]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Single image path
    # ------------------------------------------------------------------

    def _get_single_from_loaded(
        self, img: Image.Image, target: DetectionTarget
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        """Resize a pre-loaded image to input size."""
        orig_w, orig_h = img.size
        img_resized = img.resize((self.input_width, self.input_height), Image.BILINEAR)

        target = target.copy()
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= self.input_width / orig_w
            boxes[:, [1, 3]] *= self.input_height / orig_h
            target["boxes"] = tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=(self.input_height, self.input_width),
            )
        else:
            target["boxes"] = tv_tensors.BoundingBoxes(
                target["boxes"],
                format="XYXY",
                canvas_size=(self.input_height, self.input_width),
            )

        target["size"] = torch.tensor([self.input_height, self.input_width])

        if self.post_transforms is not None:
            img_resized, target = self.post_transforms(img_resized, target)

        return img_resized, target

    # ------------------------------------------------------------------
    # Mosaic path
    # ------------------------------------------------------------------

    def _get_mosaic(
        self, current_img: Image.Image, current_target: DetectionTarget
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        """Create a 4-image mosaic with optional MixUp.

        The first image is the already-loaded current sample.  The remaining
        3 (and the optional mixup companion) come from the cache when
        available, otherwise from the dataset.
        """
        # 3 companion images from cache or dataset
        companions = [self._get_companion() for _ in range(3)]
        images_and_targets: list[_CachedSample] = [
            (current_img, current_target),
            *companions,
        ]

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

        first_target: DetectionTarget | None = None
        for i, ((img, target), (x_off, y_off, qw, qh)) in enumerate(
            zip(images_and_targets, quadrants, strict=True)
        ):
            if qw <= 0 or qh <= 0:
                continue

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

        result_target: DetectionTarget = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=(self.input_height, self.input_width),
            ),
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
        """Apply MixUp: alpha-blend canvas with a companion image."""
        mix_img, mix_target = self._get_companion()
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
