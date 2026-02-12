"""
Cached dataset for faster training data loading.

Wraps any DetectionDataset and caches decoded images + pre-built target
tensors.  Supports two cache backends:

- **ram**: Stores samples in a Python list for zero-overhead reads.
- **disk**: Persists samples in a SQLite database for cross-run reuse.

Cache stores **pre-transform** data so stochastic augmentations still
produce different results each epoch.
"""

from __future__ import annotations

import concurrent.futures
import copy
import io
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from object_detection_training.data.detection_dataset import DetectionDataset
from object_detection_training.types import DetectionTarget

__all__ = ["CacheDataset"]

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cache (
    idx       INTEGER PRIMARY KEY,
    img_blob  BLOB    NOT NULL,
    target    BLOB    NOT NULL
);
"""

_INSERT = """
INSERT OR REPLACE INTO cache (idx, img_blob, target)
VALUES (?, ?, ?);
"""

_SELECT = "SELECT img_blob, target FROM cache WHERE idx = ?;"

_COUNT = "SELECT COUNT(*) FROM cache;"

# JPEG quality for compressed storage (95 is visually lossless)
_JPEG_QUALITY = 95


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_image(img: Image.Image) -> bytes:
    """Serialize a PIL Image to compressed JPEG bytes.

    Stores JPEG-compressed data (~20-30x smaller than raw pixels for
    typical photos) while remaining visually lossless at quality=95.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def _deserialize_image(img_bytes: bytes) -> Image.Image:
    """Reconstruct a PIL Image from JPEG bytes."""
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


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


# ---------------------------------------------------------------------------
# CacheDataset
# ---------------------------------------------------------------------------
class CacheDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor | Image.Image, DetectionTarget]]
):
    """Wraps a DetectionDataset with an in-memory or on-disk cache.

    On first access (or explicit ``build_cache()``), every sample is read
    from the underlying dataset and cached.  Subsequent reads bypass PIL
    I/O and annotation parsing entirely.

    Parameters
    ----------
    dataset:
        The underlying DetectionDataset
    cache_type:
        Type of cache storage backend:
        - "ram": Store decoded images in memory (fastest, high RAM usage).
        - "disk": Store compressed images in SQLite (slower, low RAM usage).
        - "auto": Automatically select based on available system RAM.
    cache_dir:
        Directory for the SQLite DB (disk mode only).  Defaults to
        ``{dataset.root_path}/.cache/``.
    transforms:
        Optional transforms applied **after** cache retrieval so that
        stochastic augmentations produce different results each epoch.
    rebuild:
        If ``True``, delete any existing cache and rebuild from scratch.
    num_threads:
        Number of worker threads for parallel cache building. Defaults to
        min(32, os.cpu_count() + 4).
    """

    def __init__(
        self,
        dataset: DetectionDataset,
        cache_type: Literal["ram", "disk", "auto"] = "disk",
        cache_dir: str | Path | None = None,
        transforms: Any | None = None,
        rebuild: bool = False,
        num_threads: int | None = None,
    ) -> None:
        self._dataset = dataset
        self.transforms = transforms

        if cache_type == "auto":
            self._cache_type = self._resolve_cache_type()
        else:
            self._cache_type = cache_type

        if num_threads is None:
            # Default to reasonable number of threads for I/O bound work
            cpu_count = os.cpu_count() or 1
            self.num_threads = min(32, cpu_count + 4)
        else:
            self.num_threads = num_threads

        # --- RAM cache state ---
        self._ram_cache: list[tuple[Image.Image, DetectionTarget] | None] | None = None

        # --- Disk cache state ---
        self._db_path: Path | None = None
        self._conn: sqlite3.Connection | None = None

        if self._cache_type == "ram":
            self._build_ram_cache()
        else:
            # Determine cache location
            if cache_dir is None:
                cache_dir = dataset.root_path / ".cache"
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self._cache_dir / f"{dataset.split}.db"

            if rebuild and self._db_path.exists():
                logger.info(f"Removing existing cache: {self._db_path}")
                self._db_path.unlink()

            self._ensure_disk_cache()

    # ------------------------------------------------------------------
    # RAM cache
    # ------------------------------------------------------------------
    def _build_ram_cache(self) -> None:
        """Load all samples into a Python list for zero-overhead reads."""
        saved_transforms = self._dataset.transforms
        self._dataset.transforms = None

        total = len(self._dataset)
        logger.info(f"Building RAM cache: {total} samples (threads={self.num_threads})")

        def _load_sample(idx: int) -> tuple[Image.Image, DetectionTarget]:
            img, target = self._dataset[idx]
            img = _coerce_to_pil(img)
            return img, target

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
        ) as executor:
            # Use map to preserve order corresponding to indices
            results = list(
                tqdm(
                    executor.map(_load_sample, range(total)),
                    total=total,
                    desc="RAM Cache",
                    unit="img",
                )
            )

        self._ram_cache = results
        self._dataset.transforms = saved_transforms
        logger.info(f"RAM cache built: {total} samples in memory")

    # ------------------------------------------------------------------
    # SQLite connection management
    # ------------------------------------------------------------------
    @property
    def _connection(self) -> sqlite3.Connection:
        """Return a per-process SQLite connection (WAL mode for readers)."""
        if self._conn is None:
            if self._db_path is None:
                raise RuntimeError("No db_path configured for disk cache")
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    # ------------------------------------------------------------------
    # Disk cache building
    # ------------------------------------------------------------------
    def _ensure_disk_cache(self) -> None:
        """Build the disk cache if it is missing or incomplete."""
        conn = self._connection
        conn.execute(_CREATE_TABLE)
        conn.commit()

        count = conn.execute(_COUNT).fetchone()[0]
        expected = len(self._dataset)

        if count >= expected:
            logger.info(f"Cache hit: {self._db_path} ({count} samples already cached)")
            return
        logger.info(
            f"Building cache: {self._db_path} "
            f"({count}/{expected} samples present, caching remaining)"
        )
        self.build_cache()

    def build_cache(self) -> None:
        """Populate the SQLite cache from the underlying dataset.

        Temporarily strips transforms from the wrapped dataset so that
        raw (pre-transform) data is cached. Uses ThreadPoolExecutor for
        parallel I/O and encoding.
        """
        saved_transforms = self._dataset.transforms
        self._dataset.transforms = None

        conn = self._connection
        conn.execute(_CREATE_TABLE)

        total = len(self._dataset)
        batch_size = 1000

        # Optimization: Fetch all existing indices at once to avoid SELECT inside loop
        existing_cursor = conn.execute("SELECT idx FROM cache")
        existing_indices = {row[0] for row in existing_cursor.fetchall()}

        indices_to_process = [i for i in range(total) if i not in existing_indices]

        if not indices_to_process:
            self._dataset.transforms = saved_transforms
            logger.info("All samples already cached.")
            return

        logger.info(
            f"Building disk cache: {len(indices_to_process)} samples "
            f"(threads={self.num_threads})"
        )

        def _process_sample(idx: int) -> tuple[int, bytes, bytes] | None:
            try:
                img, target = self._dataset[idx]
                img = _coerce_to_pil(img)
                img_bytes = _serialize_image(img)
                target_bytes = _serialize_target(target)
                return idx, img_bytes, target_bytes
            except Exception as e:
                logger.error(f"Failed to process sample {idx}: {e}")
                return None

        # Execute parallel processing
        pending_inserts = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
        ) as executor:
            futures = [
                executor.submit(_process_sample, idx) for idx in indices_to_process
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(indices_to_process),
                desc="Disk Cache",
                unit="img",
            ):
                result = future.result()
                if result is None:
                    continue

                pending_inserts.append(result)

                if len(pending_inserts) >= batch_size:
                    conn.executemany(_INSERT, pending_inserts)
                    conn.commit()
                    pending_inserts.clear()

        # Flush remaining
        if pending_inserts:
            conn.executemany(_INSERT, pending_inserts)
            conn.commit()

        self._dataset.transforms = saved_transforms
        logger.info(f"Cache built: {total} samples in {self._db_path}")

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
        img, target = copy.deepcopy(cached)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def _getitem_disk(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        """Read from SQLite disk cache."""
        row = self._connection.execute(_SELECT, (idx,)).fetchone()

        if row is None:
            raise RuntimeError(
                f"Cache miss at index {idx} — cache may be corrupted. "
                f"Delete {self._db_path} and re-run to rebuild."
            )

        img_bytes, target_bytes = row
        img = _deserialize_image(img_bytes)
        target: DetectionTarget = _deserialize_target(target_bytes)

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
        return (
            f"CacheDataset(wrapped={self._dataset!r}, "
            f"cache_type='disk', cache={self._db_path}, "
            f"cached={len(self)} samples)"
        )

    def _close(self) -> None:
        """Close SQLite connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _resolve_cache_type(self) -> Literal["ram", "disk"]:
        """Determine whether to use RAM or disk cache based on available memory.

        Uses dataset image dimensions (metadata) to estimate uncompressed RAM usage.
        If estimated usage is < 50% of currently available system RAM, selects 'ram'.
        Otherwise selects 'disk'.
        """
        try:
            # Estimate dataset size in RAM (uncompressed uint8 pixels)
            # stored as list of tuples (PIL Image, dict)
            # Image: W * H * 3 bytes
            # Target: negligible compared to image
            total_pixels = self._estimate_dataset_pixels()
            estimated_bytes = int(total_pixels * 3 * 1.2)  # 1.2x overhead

            # Check system RAM
            available_bytes = psutil.virtual_memory().available
            threshold = available_bytes * 0.5

            logger.info(
                f"Auto-cache: Est. dataset size={estimated_bytes / 1e9:.2f}GB, "
                f"Available RAM={available_bytes / 1e9:.2f}GB, "
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
            # DetectionDataset guarantees images_df with width/height columns
            if self._dataset.images_df is not None:
                widths = self._dataset.images_df["width"]
                heights = self._dataset.images_df["height"]
                return int((widths * heights).sum())
        except Exception as e:
            logger.debug(f"Could not access image metadata: {e}")

        # Fallback if metadata unavailable
        logger.warning("Could not access image metadata. Assuming 1920x1080 per image.")
        return len(self._dataset) * 1920 * 1080
