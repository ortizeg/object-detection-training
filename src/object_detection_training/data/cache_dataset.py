"""
SQLite-backed cache dataset for faster training data loading.

Wraps any DetectionDataset and caches decoded images + pre-built target
tensors in a local SQLite database.  Eliminates repeated PIL disk reads
and DataFrame queries after the first full pass.

Cache stores **pre-transform** data so stochastic augmentations still
produce different results each epoch.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image

from object_detection_training.data.detection_dataset import DetectionDataset
from object_detection_training.types import DetectionTarget

__all__ = ["CacheDataset"]

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cache (
    idx       INTEGER PRIMARY KEY,
    img_bytes BLOB    NOT NULL,
    img_mode  TEXT    NOT NULL,
    img_w     INTEGER NOT NULL,
    img_h     INTEGER NOT NULL,
    target    BLOB    NOT NULL
);
"""

_INSERT = """
INSERT OR REPLACE INTO cache (idx, img_bytes, img_mode, img_w, img_h, target)
VALUES (?, ?, ?, ?, ?, ?);
"""

_SELECT = "SELECT img_bytes, img_mode, img_w, img_h, target FROM cache WHERE idx = ?;"

_COUNT = "SELECT COUNT(*) FROM cache;"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_image(img: Image.Image) -> tuple[bytes, str, int, int]:
    """Serialize a PIL Image to raw bytes + metadata.

    Stores raw pixel data (no re-encoding) for fast deserialization.
    """
    img_array = np.array(img)
    return img_array.tobytes(), img.mode, img.size[0], img.size[1]


def _deserialize_image(
    img_bytes: bytes, mode: str, width: int, height: int
) -> Image.Image:
    """Reconstruct a PIL Image from raw bytes + metadata."""
    channels = len(mode)  # "RGB" -> 3, "L" -> 1
    img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
        (height, width, channels)
    )
    return Image.fromarray(img_array, mode=mode)


def _serialize_target(target: DetectionTarget) -> bytes:
    """Serialize a target dict of tensors to bytes using torch.save."""
    buf = io.BytesIO()
    torch.save(target, buf)  # type: ignore[no-untyped-call]
    return buf.getvalue()


def _deserialize_target(data: bytes) -> DetectionTarget:
    """Deserialize a target dict from bytes."""
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=False)  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# CacheDataset
# ---------------------------------------------------------------------------
class CacheDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor | Image.Image, DetectionTarget]]
):
    """Wraps a DetectionDataset with a SQLite blob cache.

    On first access (or explicit ``build_cache()``), every sample is read
    from the underlying dataset, serialized, and written to a SQLite DB.
    Subsequent reads bypass PIL I/O and annotation parsing entirely.

    Parameters
    ----------
    dataset:
        The underlying DetectionDataset to cache.
    cache_dir:
        Directory for the cache DB file.  Defaults to
        ``{dataset.root_path}/.cache/``.
    transforms:
        Optional transforms applied **after** cache retrieval so that
        stochastic augmentations produce different results each epoch.
    rebuild:
        If ``True``, delete any existing cache and rebuild from scratch.
    """

    def __init__(
        self,
        dataset: DetectionDataset,
        cache_dir: str | Path | None = None,
        transforms: Any | None = None,
        rebuild: bool = False,
    ) -> None:
        self._dataset = dataset
        self.transforms = transforms

        # Determine cache location
        if cache_dir is None:
            cache_dir = dataset.root_path / ".cache"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._cache_dir / f"{dataset.split}.db"

        # Optionally wipe existing cache
        if rebuild and self._db_path.exists():
            logger.info(f"Removing existing cache: {self._db_path}")
            self._db_path.unlink()

        # Connection is created lazily (per-process for DataLoader workers)
        self._conn: sqlite3.Connection | None = None

        # Build cache if it doesn't exist or is incomplete
        self._ensure_cache()

    # ------------------------------------------------------------------
    # SQLite connection management
    # ------------------------------------------------------------------
    @property
    def _connection(self) -> sqlite3.Connection:
        """Return a per-process SQLite connection (WAL mode for readers)."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    # ------------------------------------------------------------------
    # Cache building
    # ------------------------------------------------------------------
    def _ensure_cache(self) -> None:
        """Build the cache if it is missing or incomplete."""
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
        raw (pre-transform) data is cached.
        """
        saved_transforms = self._dataset.transforms
        self._dataset.transforms = None

        conn = self._connection
        conn.execute(_CREATE_TABLE)

        total = len(self._dataset)
        batch_size = 500  # commit every N inserts for performance

        for idx in range(total):
            # Check if already cached
            row = conn.execute(_SELECT, (idx,)).fetchone()
            if row is not None:
                continue

            img, target = self._dataset[idx]

            # img may be a PIL Image or a Tensor; coerce to PIL for storage
            if isinstance(img, torch.Tensor):
                # CHW -> HWC uint8
                img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
                img = Image.fromarray(img_np)

            img_bytes, mode, w, h = _serialize_image(img)
            target_bytes = _serialize_target(target)

            conn.execute(_INSERT, (idx, img_bytes, mode, w, h, target_bytes))

            if (idx + 1) % batch_size == 0:
                conn.commit()
                logger.debug(f"  cached {idx + 1}/{total} samples")

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
        row = self._connection.execute(_SELECT, (idx,)).fetchone()

        if row is None:
            raise RuntimeError(
                f"Cache miss at index {idx} — cache may be corrupted. "
                f"Delete {self._db_path} and re-run to rebuild."
            )

        img_bytes, mode, w, h, target_bytes = row
        img = _deserialize_image(img_bytes, mode, w, h)
        target: DetectionTarget = _deserialize_target(target_bytes)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target  # type: ignore[return-value]

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
        return (
            f"CacheDataset(wrapped={self._dataset!r}, "
            f"cache={self._db_path}, "
            f"cached={len(self)} samples)"
        )
