"""
Cached dataset for faster training data loading.

Wraps any DetectionDataset and caches decoded images + pre-built target
tensors.  Supports two cache backends:

- **ram**: Stores samples in a Python list for zero-overhead reads.
- **disk**: Lazy write-through sharded binary cache. Each sample is
  serialized and appended to a binary shard file with an index for
  fast random access. With DDP, each rank writes to its own shard
  (no contention), and all ranks can read from all shards.

Inspired by:
- YOLOX (Megvii): ThreadPool RAM cache, .npy disk cache
- mlproject (viebboy): Sharded BinaryBlob with index files
- mmengine: serialize_data for shared memory across workers

Cache stores **pre-transform** data so stochastic augmentations still
produce different results each epoch.
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import psutil  # type: ignore[import-untyped]
import torch
from loguru import logger
from PIL import Image

from object_detection_training.data.detection_dataset import DetectionDataset
from object_detection_training.types import DetectionTarget

__all__ = ["CacheDataset"]

# ---------------------------------------------------------------------------
# Binary format constants
# ---------------------------------------------------------------------------
# Each record: [img_len (4 bytes)] [target_len (4 bytes)] [img_bytes] [target_bytes]
_HEADER_SIZE = 8  # two uint32 lengths


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_sample(
    img: npt.NDArray[np.uint8], target: DetectionTarget
) -> tuple[bytes, bytes]:
    """Serialize image as JPEG bytes and target via torch.save.

    JPEG compression reduces disk write volume ~5-10x vs raw numpy,
    drastically cutting IO contention during cache warmup.
    """
    pil_img = Image.fromarray(img)
    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="JPEG", quality=95)
    img_bytes = img_buf.getvalue()

    target_buf = io.BytesIO()
    torch.save(target, target_buf)
    target_bytes = target_buf.getvalue()

    return img_bytes, target_bytes


def _deserialize_sample(
    img_bytes: bytes, target_bytes: bytes
) -> tuple[npt.NDArray[np.uint8], DetectionTarget]:
    """Deserialize JPEG image and target from bytes."""
    img = np.asarray(Image.open(io.BytesIO(img_bytes)), dtype=np.uint8)
    target: DetectionTarget = torch.load(io.BytesIO(target_bytes), weights_only=False)
    return img, target


def _coerce_to_pil(img: Image.Image | torch.Tensor) -> Image.Image:
    """Coerce an image to PIL format for caching."""
    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(img_np)
    return img


def _pil_to_numpy(img: Image.Image) -> npt.NDArray[np.uint8]:
    """Convert PIL Image to numpy array (H, W, 3) uint8."""
    return np.asarray(img, dtype=np.uint8)


def _numpy_to_pil(arr: npt.NDArray[np.uint8]) -> Image.Image:
    """Convert numpy array (H, W, 3) to PIL Image."""
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# BinaryShard — single append-only binary file + index
# ---------------------------------------------------------------------------
class BinaryShard:
    """A single shard of the binary cache.

    Stores serialized samples in an append-only binary file with a
    separate index mapping sample_idx -> (byte_offset, img_len, target_len).

    File format per record:
        [img_len: uint32][target_len: uint32][img_bytes][target_bytes]

    The index file is a CSV: sample_idx,byte_offset,img_len,target_len
    """

    def __init__(self, bin_path: Path, idx_path: Path) -> None:
        self._bin_path = bin_path
        self._idx_path = idx_path
        # In-memory index: sample_idx -> (byte_offset, img_len, target_len)
        self._index: dict[int, tuple[int, int, int]] = {}
        # File handles (lazily opened, per-process via PID tracking)
        self._read_fh: io.BufferedReader | None = None
        self._write_fh: io.BufferedWriter | None = None
        self._idx_fh: io.TextIOWrapper | None = None
        self._pid: int | None = None
        # Load existing index if present
        self._load_index()

    def _load_index(self) -> None:
        """Load the index file into memory."""
        if not self._idx_path.exists():
            return
        with open(self._idx_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                sample_idx = int(parts[0])
                byte_offset = int(parts[1])
                img_len = int(parts[2])
                target_len = int(parts[3])
                self._index[sample_idx] = (byte_offset, img_len, target_len)

    def _ensure_read_fh(self) -> io.BufferedReader:
        """Get or create a read file handle (process-safe)."""
        pid = os.getpid()
        if self._read_fh is None or self._pid != pid:
            if self._read_fh is not None:
                with contextlib.suppress(Exception):
                    self._read_fh.close()
            self._read_fh = open(self._bin_path, "rb")  # noqa: SIM115
            self._pid = pid
        return self._read_fh

    def _ensure_write_fhs(self) -> tuple[io.BufferedWriter, io.TextIOWrapper]:
        """Get or create write file handles (append mode)."""
        if self._write_fh is None:
            self._write_fh = open(self._bin_path, "ab")  # noqa: SIM115
            self._idx_fh = open(self._idx_path, "a")  # noqa: SIM115
        if self._idx_fh is None:
            raise RuntimeError("Index file handle not initialized")
        return self._write_fh, self._idx_fh

    def __contains__(self, sample_idx: int) -> bool:
        return sample_idx in self._index

    def __len__(self) -> int:
        return len(self._index)

    def write(self, sample_idx: int, img_bytes: bytes, target_bytes: bytes) -> None:
        """Append a sample to the shard."""
        if sample_idx in self._index:
            return  # Already cached

        bin_fh, idx_fh = self._ensure_write_fhs()

        byte_offset = bin_fh.tell()
        # Write header + data
        header = struct.pack("<II", len(img_bytes), len(target_bytes))
        bin_fh.write(header)
        bin_fh.write(img_bytes)
        bin_fh.write(target_bytes)
        bin_fh.flush()

        # Write index entry
        idx_fh.write(
            f"{sample_idx},{byte_offset},{len(img_bytes)},{len(target_bytes)}\n"
        )
        idx_fh.flush()

        # Update in-memory index
        self._index[sample_idx] = (
            byte_offset,
            len(img_bytes),
            len(target_bytes),
        )

    def read(self, sample_idx: int) -> tuple[bytes, bytes]:
        """Read a sample from the shard by index."""
        if sample_idx not in self._index:
            raise KeyError(f"Sample {sample_idx} not in shard {self._bin_path}")

        byte_offset, img_len, target_len = self._index[sample_idx]
        fh = self._ensure_read_fh()
        fh.seek(byte_offset + _HEADER_SIZE)
        img_bytes = fh.read(img_len)
        target_bytes = fh.read(target_len)
        return img_bytes, target_bytes

    def close(self) -> None:
        """Close all file handles."""
        for fh in (self._read_fh, self._write_fh, self._idx_fh):
            if fh is not None:
                with contextlib.suppress(Exception):
                    fh.close()
        self._read_fh = None
        self._write_fh = None
        self._idx_fh = None


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
        - "ram": In-memory cache (fastest, high RAM usage). Built upfront.
        - "disk": Lazy write-through sharded binary cache. Each DDP rank
          writes to its own shard file; all ranks read from all shards.
          Training starts immediately — cache builds during epoch 1.
        - "auto": Selects ram/disk based on available memory and world size.
    cache_dir:
        Directory for shard files (disk mode). Defaults to
        ``{dataset.root_path}/.cache/{split}/``.
    transforms:
        Optional transforms applied **after** cache retrieval.
    rebuild:
        If True, delete existing cache and start fresh.
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
        self._ram_cache: (
            list[tuple[npt.NDArray[np.uint8], DetectionTarget] | None] | None
        ) = None

        # --- Disk cache state ---
        self._cache_dir: Path | None = None
        self._shards: list[BinaryShard] = []
        self._write_shard: BinaryShard | None = None
        # Per-worker write shards (lazily created after fork)
        self._worker_shards: dict[int, BinaryShard] = {}
        self._rank: int = 0

        if self._cache_type == "ram":
            self._build_ram_cache()
        else:
            if cache_dir is None:
                # Use local storage for cache (not GCS FUSE mount).
                # /tmp is local SSD on cloud VMs — fast and no cross-job
                # contention since /tmp is per-VM.
                cache_dir = Path("/tmp") / "dataset_cache" / dataset.split  # noqa: S108
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            if rebuild:
                self._clear_disk_cache()

            self._init_shards()

    # ------------------------------------------------------------------
    # Shard management
    # ------------------------------------------------------------------
    def _init_shards(self) -> None:
        """Discover existing shards and set up read access.

        Write shards are created lazily per DataLoader worker in
        ``_get_worker_write_shard`` to avoid cross-worker file contention.
        Each worker gets its own shard: ``shard_{rank}_w{worker_id}.bin``.
        """
        if self._cache_dir is None:
            return

        self._rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Discover all existing shards (from this and previous runs)
        self._shards = []
        for idx_file in sorted(self._cache_dir.glob("shard_*.idx")):
            shard_bin = idx_file.with_suffix(".bin")
            if shard_bin.exists():
                shard = BinaryShard(shard_bin, idx_file)
                self._shards.append(shard)

        total_cached = sum(len(s) for s in self._shards)
        total = len(self._dataset)
        logger.info(
            f"Disk cache: {self._cache_dir} "
            f"({total_cached}/{total} samples across {len(self._shards)} shards, "
            f"rank={self._rank})"
        )

    def _get_worker_write_shard(self) -> BinaryShard:
        """Get or create a write shard for the current DataLoader worker.

        Each worker gets its own shard file so all workers can write in
        parallel with zero contention. Full cache in 1 epoch.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if worker_id not in self._worker_shards:
            if self._cache_dir is None:
                raise RuntimeError("Cache dir not set")
            write_bin = self._cache_dir / f"shard_{self._rank:03d}_w{worker_id:02d}.bin"
            write_idx = self._cache_dir / f"shard_{self._rank:03d}_w{worker_id:02d}.idx"
            shard = BinaryShard(write_bin, write_idx)
            self._worker_shards[worker_id] = shard
            # Also add to read shards list
            self._shards.append(shard)

        return self._worker_shards[worker_id]

    def _find_in_shards(self, sample_idx: int) -> BinaryShard | None:
        """Find which shard contains a given sample index."""
        for shard in self._shards:
            if sample_idx in shard:
                return shard
        return None

    def _clear_disk_cache(self) -> None:
        """Remove all cached files."""
        if self._cache_dir is None:
            return
        import shutil

        if self._cache_dir.exists():
            logger.info(f"Removing existing cache: {self._cache_dir}")
            shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # RAM cache (upfront, like YOLOX official)
    # ------------------------------------------------------------------
    def _build_ram_cache(self) -> None:
        """Load all samples into a Python list for zero-overhead reads."""
        from multiprocessing.pool import ThreadPool

        from tqdm import tqdm

        saved_transforms = self._dataset.transforms
        self._dataset.transforms = None

        total = len(self._dataset)
        num_threads = min(8, max(1, (os.cpu_count() or 1) - 1))
        logger.info(f"Building RAM cache: {total} samples (threads={num_threads})")

        def _load_sample(idx: int) -> tuple[npt.NDArray[np.uint8], DetectionTarget]:
            img, target = self._dataset[idx]
            img = _coerce_to_pil(img)
            return _pil_to_numpy(img), target

        results: list[tuple[npt.NDArray[np.uint8], DetectionTarget] | None] = [
            None
        ] * total
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
        """Read from in-memory list.

        Images are stored as uint8 numpy arrays — wrapping back to PIL
        and shallow-copying the target dict is ~10x faster than
        ``copy.deepcopy`` on a PIL Image + nested dict.  Downstream
        transforms always clone tensors before in-place mutation.
        """
        if self._ram_cache is None:
            raise RuntimeError("RAM cache not initialized")
        cached = self._ram_cache[idx]
        if cached is None:
            raise RuntimeError(f"RAM cache miss at index {idx}")
        img_arr, target = cached
        img: torch.Tensor | Image.Image = _numpy_to_pil(img_arr)
        target = target.copy()

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def _getitem_disk(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, DetectionTarget]:
        """Lazy write-through sharded disk cache.

        On cache hit: seek + read from any shard that has this sample.
        On cache miss: read from underlying dataset, write to this worker's
        private shard. Each DataLoader worker writes to its own shard file
        (``shard_{rank}_w{worker_id}.bin``), so all workers write in parallel
        with zero contention. Full cache in 1 epoch.
        """
        shard = self._find_in_shards(idx)

        if shard is not None:
            # Cache hit — read from shard
            img_bytes, target_bytes = shard.read(idx)
            img_arr, target = _deserialize_sample(img_bytes, target_bytes)
        else:
            # Cache miss — read from underlying dataset
            saved_transforms = self._dataset.transforms
            self._dataset.transforms = None
            try:
                img, target = self._dataset[idx]
                img = _coerce_to_pil(img)
                img_arr = _pil_to_numpy(img)
            finally:
                self._dataset.transforms = saved_transforms

            # Write to this worker's private shard (no contention)
            write_shard = self._get_worker_write_shard()
            img_bytes, target_bytes = _serialize_sample(img_arr, target)
            write_shard.write(idx, img_bytes, target_bytes)

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
        total_cached = sum(len(s) for s in self._shards)
        return (
            f"CacheDataset(wrapped={self._dataset!r}, "
            f"cache_type='disk', shards={len(self._shards)}, "
            f"cached={total_cached}/{len(self)} samples)"
        )

    def _resolve_cache_type(self) -> Literal["ram", "disk"]:
        """Determine whether to use RAM or disk cache based on available memory.

        Accounts for DDP world size since multiple processes share RAM.
        """
        try:
            total_pixels = self._estimate_dataset_pixels()
            estimated_bytes = int(total_pixels * 3 * 1.2)  # 1.2x overhead

            available_bytes = psutil.virtual_memory().available
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            per_process_available = available_bytes // max(world_size, 1)
            # RAM cache loads ALL samples per rank (DistributedSampler
            # shuffles, so every rank eventually sees every sample).
            # Use 50% of per-process budget as threshold.
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
