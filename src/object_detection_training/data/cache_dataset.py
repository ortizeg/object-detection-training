"""
Cached dataset for faster training data loading.

Wraps any DetectionDataset and caches decoded images + pre-built target
tensors.  Supports two cache backends:

- **ram**: Stores samples in a Python list for zero-overhead reads.
- **disk**: Lazy write-through cache using individual .npy files.
  On first access (epoch 1), samples are read from the underlying dataset
  and cached to disk. Subsequent reads bypass PIL I/O entirely.
  With DDP, each rank caches different indices via DistributedSampler,
  so all ranks collaboratively build the full cache in one epoch.

Cache stores **pre-transform** data so stochastic augmentations still
produce different results each epoch.
"""

from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from loguru import logger
from PIL import Image

from object_detection_training.data.detection_dataset import DetectionDataset
from object_detection_training.types import DetectionTarget

__all__ = ["CacheDataset"]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_target(target: DetectionTarget) -> bytes:
    """Serialize a target dict of tensors to bytes using torch.save."""
    buf = io.BytesIO()
    torch.save(target, buf)
    return buf.getvalue()


def _deserialize_target(data: bytes) -> DetectionTarget:
    """Deserialize a target dict from bytes."""
    buf = io.BytesIO(data)
    result: DetectionTarget = torch.load(buf, weights_only=False)
    return result


def _coerce_to_pil(img: Image.Image | torch.Tensor) -> Image.Image:
    """Coerce an image to PIL format for caching."""
    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(img_np)
    return img


def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (H, W, 3) uint8."""
    return np.asarray(img, dtype=np.uint8)


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array (H, W, 3) to PIL Image."""
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# CacheDataset
# ---------------------------------------------------------------------------
class CacheDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor | Image.Image, DetectionTarget]]
):
    """Wraps a DetectionDataset with an in-memory or on-disk cache.

    Parameters
    ----------
    dataset:
        The underlying DetectionDataset
    cache_type:
        Type of cache storage backend:
        - "ram": Store decoded images in memory (fastest, high RAM usage).
          Built upfront before training starts.
        - "disk": Lazy write-through cache using .npy files. Samples are
          cached on first access during epoch 1. With DDP, each rank
          caches different indices so the full cache builds collaboratively.
        - "auto": Automatically select based on available system RAM.
    cache_dir:
        Directory for .npy files (disk mode only). Defaults to
        ``{dataset.root_path}/.cache/{split}/``.
    transforms:
        Optional transforms applied **after** cache retrieval so that
        stochastic augmentations produce different results each epoch.
    rebuild:
        If ``True``, delete any existing cache and rebuild from scratch.
    """

    def __init__(
        self,
        dataset: DetectionDataset,
        cache_type: Literal["ram", "disk", "auto"] = "disk",
        cache_dir: str | Path | None = None,
        transforms: Any | None = None,
        rebuild: bool = False,
        **kwargs: Any,
    ) -> None:
        self._dataset = dataset
        self.transforms = transforms

        if cache_type == "auto":
            self._cache_type = self._resolve_cache_type()
        else:
            self._cache_type = cache_type

        # --- RAM cache state ---
        self._ram_cache: list[tuple[np.ndarray, DetectionTarget] | None] | None = None

        # --- Disk cache state ---
        self._cache_dir: Path | None = None

        if self._cache_type == "ram":
            self._build_ram_cache()
        else:
            if cache_dir is None:
                cache_dir = dataset.root_path / ".cache" / dataset.split
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            if rebuild:
                self._clear_disk_cache()

            cached_count = self._count_cached()
            total = len(self._dataset)
            logger.info(
                f"Disk cache: {self._cache_dir} "
                f"({cached_count}/{total} samples already cached, "
                f"remaining will be cached lazily during training)"
            )

    # ------------------------------------------------------------------
    # RAM cache (upfront, like YOLOX official)
    # ------------------------------------------------------------------
    def _build_ram_cache(self) -> None:
        """Load all samples into a Python list for zero-overhead reads.

        Uses ThreadPool for parallel I/O (like YOLOX official).
        """
        from multiprocessing.pool import ThreadPool

        from tqdm import tqdm

        saved_transforms = self._dataset.transforms
        self._dataset.transforms = None

        total = len(self._dataset)
        num_threads = min(8, max(1, (os.cpu_count() or 1) - 1))
        logger.info(f"Building RAM cache: {total} samples (threads={num_threads})")

        def _load_sample(idx: int) -> tuple[np.ndarray, DetectionTarget]:
            img, target = self._dataset[idx]
            img = _coerce_to_pil(img)
            return _pil_to_numpy(img), target

        results: list[tuple[np.ndarray, DetectionTarget] | None] = [None] * total
        pool = ThreadPool(num_threads)
        loaded = pool.imap(_load_sample, range(total))

        mem_bytes = 0
        gb = 1 << 30
        for i, sample in enumerate(
            tqdm(loaded, total=total, desc="RAM Cache", unit="img")
        ):
            results[i] = sample
            mem_bytes += sample[0].nbytes

        pool.close()
        pool.join()

        self._ram_cache = results
        self._dataset.transforms = saved_transforms
        logger.info(
            f"RAM cache built: {total} samples, {mem_bytes / gb:.1f}GB in memory"
        )

    # ------------------------------------------------------------------
    # Disk cache helpers
    # ------------------------------------------------------------------
    def _img_cache_path(self, idx: int) -> Path:
        """Return the .npy file path for a given index."""
        if self._cache_dir is None:
            raise RuntimeError("No cache_dir configured for disk cache")
        return self._cache_dir / f"{idx:06d}_img.npy"

    def _target_cache_path(self, idx: int) -> Path:
        """Return the target cache file path for a given index."""
        if self._cache_dir is None:
            raise RuntimeError("No cache_dir configured for disk cache")
        return self._cache_dir / f"{idx:06d}_target.bin"

    def _is_cached(self, idx: int) -> bool:
        """Check if a sample is already cached on disk."""
        return self._img_cache_path(idx).exists()

    def _count_cached(self) -> int:
        """Count how many samples are already cached."""
        if self._cache_dir is None:
            return 0
        return len(list(self._cache_dir.glob("*_img.npy")))

    def _clear_disk_cache(self) -> None:
        """Remove all cached files."""
        if self._cache_dir is None:
            return
        import shutil

        if self._cache_dir.exists():
            logger.info(f"Removing existing cache: {self._cache_dir}")
            shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _write_to_disk(
        self, idx: int, img: np.ndarray, target: DetectionTarget
    ) -> None:
        """Write a single sample to disk cache (atomic via rename)."""
        img_path = self._img_cache_path(idx)
        target_path = self._target_cache_path(idx)

        # Atomic write: write to temp then rename (safe for concurrent DDP)
        # np.save appends .npy if missing, so use .tmp extension without .npy
        tmp_img = img_path.parent / f"{img_path.stem}.tmp"
        tmp_target = target_path.parent / f"{target_path.stem}.tmp"

        np.save(str(tmp_img), img)
        # np.save auto-appends .npy, so the actual file is .tmp.npy
        tmp_img_actual = tmp_img.with_suffix(".tmp.npy")
        with open(tmp_target, "wb") as f:
            f.write(_serialize_target(target))

        tmp_img_actual.rename(img_path)
        tmp_target.rename(target_path)

    def _read_from_disk(self, idx: int) -> tuple[np.ndarray, DetectionTarget]:
        """Read a single sample from disk cache."""
        img = np.load(str(self._img_cache_path(idx)))
        with open(self._target_cache_path(idx), "rb") as f:
            target = _deserialize_target(f.read())
        return img, target

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        """Retrieve a sample from cache and apply transforms."""
        if self._cache_type == "ram":
            return self._getitem_ram(idx)
        return self._getitem_disk(idx)

    def _getitem_ram(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        """Read from in-memory list (deepcopy to prevent mutation)."""
        if self._ram_cache is None:
            raise RuntimeError("RAM cache not initialized")
        cached = self._ram_cache[idx]
        if cached is None:
            raise RuntimeError(f"RAM cache miss at index {idx}")
        img_arr, target = cached
        img = _numpy_to_pil(img_arr.copy())
        target = copy.deepcopy(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def _getitem_disk(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        """Lazy write-through disk cache.

        On cache hit: read from .npy file (fast local I/O).
        On cache miss: read from underlying dataset, write to cache, return.
        """
        if self._is_cached(idx):
            img_arr, target = self._read_from_disk(idx)
        else:
            # Cache miss — read from underlying dataset (GCS FUSE)
            saved_transforms = self._dataset.transforms
            self._dataset.transforms = None
            try:
                img, target = self._dataset[idx]
                img = _coerce_to_pil(img)
                img_arr = _pil_to_numpy(img)
                # Write to disk for future epochs
                self._write_to_disk(idx, img_arr, target)
            finally:
                self._dataset.transforms = saved_transforms

        img = _numpy_to_pil(img_arr)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # ------------------------------------------------------------------
    # Proxy properties for compatibility
    # ------------------------------------------------------------------
    @property
    def dataset(self) -> DetectionDataset:
        """Access the underlying wrapped dataset."""
        return self._dataset

    @property
    def num_classes(self) -> int:
        return self._dataset.num_classes

    @property
    def class_names(self) -> list[str]:
        return self._dataset.class_names

    @property
    def label_map(self) -> dict[int, int]:
        return self._dataset.label_map

    @property
    def image_ids(self) -> list[int]:
        return self._dataset.image_ids

    def __repr__(self) -> str:
        if self._cache_type == "ram":
            return (
                f"CacheDataset(wrapped={self._dataset!r}, "
                f"cache_type='ram', "
                f"cached={len(self)} samples)"
            )
        cached = self._count_cached()
        return (
            f"CacheDataset(wrapped={self._dataset!r}, "
            f"cache_type='disk', cache={self._cache_dir}, "
            f"cached={cached}/{len(self)} samples)"
        )

    def _resolve_cache_type(self) -> Literal["ram", "disk"]:
        """Determine whether to use RAM or disk cache based on available memory.

        Estimates RAM usage from image metadata. Accounts for DDP world size
        since multiple processes share the same physical RAM.
        """
        try:
            total_pixels = self._estimate_dataset_pixels()
            estimated_bytes = int(total_pixels * 3 * 1.2)  # 1.2x overhead

            available_bytes = psutil.virtual_memory().available
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            per_process_available = available_bytes // max(world_size, 1)
            threshold = per_process_available * 0.5

            logger.info(
                f"Auto-cache: Est. dataset size={estimated_bytes / 1e9:.2f}GB, "
                f"Available RAM={available_bytes / 1e9:.2f}GB, "
                f"World size={world_size}, "
                f"Per-process budget={per_process_available / 1e9:.2f}GB, "
                f"Threshold={threshold / 1e9:.2f}GB"
            )

            if estimated_bytes < threshold:
                logger.info("Auto-cache: Selected 'ram' mode")
                return "ram"
            else:
                logger.info("Auto-cache: Selected 'disk' mode")
                return "disk"

        except Exception as e:
            logger.warning(f"Auto-cache dispatch failed: {e}. Defaulting to 'disk'.")
            return "disk"

    def _estimate_dataset_pixels(self) -> int:
        """Estimate total pixels in dataset using metadata (no image load)."""
        try:
            if self._dataset.images_df is not None:
                widths = self._dataset.images_df["width"]
                heights = self._dataset.images_df["height"]
                return int((widths * heights).sum())
        except Exception as e:
            logger.debug(f"Could not access image metadata: {e}")

        logger.warning("Could not access image metadata. Assuming 1920x1080 per image.")
        return len(self._dataset) * 1920 * 1080
