"""Tests for CacheDataset — correctness and benchmark.

Creates a temporary synthetic COCO dataset (in-memory), wraps it with
CacheDataset, and verifies:
1. Cached output is identical to uncached output (both RAM and disk modes).
2. Cached iteration is measurably faster than uncached.
3. Sharded disk cache builds lazily and supports multi-shard reads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from object_detection_training.data.cache_dataset import CacheDataset
from object_detection_training.data.coco_detection_dataset import COCODetectionDataset

# ---------------------------------------------------------------------------
# Fixtures — synthetic COCO dataset
# ---------------------------------------------------------------------------
NUM_IMAGES = 20
IMG_WIDTH = 640
IMG_HEIGHT = 480
NUM_CATEGORIES = 3
ANNOTATIONS_PER_IMAGE = 5
NUM_BENCHMARK_ITERS = 5


def _create_synthetic_coco(root: Path) -> Path:
    """Create a minimal COCO-format dataset with real JPEG files."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [
        {"id": i + 1, "name": f"class_{i}", "supercategory": "none"}
        for i in range(NUM_CATEGORIES)
    ]

    ann_id = 1
    for img_id in range(1, NUM_IMAGES + 1):
        filename = f"img_{img_id:04d}.jpg"

        rng = np.random.default_rng(seed=img_id)
        arr = rng.integers(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img.save(images_dir / filename, format="JPEG")

        images.append(
            {
                "id": img_id,
                "file_name": filename,
                "width": IMG_WIDTH,
                "height": IMG_HEIGHT,
            }
        )

        for _ in range(ANNOTATIONS_PER_IMAGE):
            x = int(rng.integers(0, IMG_WIDTH - 50))
            y = int(rng.integers(0, IMG_HEIGHT - 50))
            w = int(rng.integers(10, 50))
            h = int(rng.integers(10, 50))
            cat_id = int(rng.integers(1, NUM_CATEGORIES + 1))

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    ann_file = root / "_annotations.coco.json"
    with open(ann_file, "w") as f:
        json.dump(coco_data, f)

    return root


@pytest.fixture()
def coco_root(tmp_path: Path) -> Path:
    """Create a temporary synthetic COCO dataset."""
    return _create_synthetic_coco(tmp_path / "coco_data")


@pytest.fixture()
def base_dataset(coco_root: Path) -> COCODetectionDataset:
    """Create an uncached COCODetectionDataset."""
    return COCODetectionDataset(
        root_path=str(coco_root),
        split="train",
    )


@pytest.fixture()
def cached_dataset_disk(base_dataset: COCODetectionDataset) -> CacheDataset:
    """Create a disk-cached version of the dataset (fully populated)."""
    cache = CacheDataset(
        dataset=base_dataset,
        cache_type="disk",
        rebuild=True,
    )
    # Access all samples to populate the lazy cache
    for idx in range(len(cache)):
        _ = cache[idx]
    return cache


@pytest.fixture()
def cached_dataset_ram(base_dataset: COCODetectionDataset) -> CacheDataset:
    """Create a RAM-cached version of the dataset."""
    return CacheDataset(
        dataset=base_dataset,
        cache_type="ram",
    )


# ---------------------------------------------------------------------------
# Correctness tests (parametrized over cache backends)
# ---------------------------------------------------------------------------
class TestCacheDatasetCorrectness:
    """Verify cached output matches uncached output exactly."""

    @pytest.fixture(
        params=["disk", "ram"],
    )
    def cached_dataset(
        self,
        request: pytest.FixtureRequest,
        base_dataset: COCODetectionDataset,
    ) -> CacheDataset:
        """Parametrized fixture yielding both cache backends."""
        cache_type: Literal["ram", "disk"] = request.param
        cache = CacheDataset(
            dataset=base_dataset,
            cache_type=cache_type,
            rebuild=True,
        )
        # For disk mode, access all samples to populate the lazy cache
        if cache_type == "disk":
            for idx in range(len(cache)):
                _ = cache[idx]
        return cache

    def test_same_length(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset: CacheDataset,
    ) -> None:
        assert len(cached_dataset) == len(base_dataset)

    def test_same_output_all_samples(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset: CacheDataset,
    ) -> None:
        """Every sample from the cache matches the original dataset."""
        for idx in range(len(base_dataset)):
            img_orig, target_orig = base_dataset[idx]
            img_cached, target_cached = cached_dataset[idx]

            orig_arr = np.array(img_orig)
            cached_arr = np.array(img_cached)

            assert orig_arr.shape == cached_arr.shape, (
                f"Image shape mismatch at idx {idx}: "
                f"{orig_arr.shape} vs {cached_arr.shape}"
            )
            assert np.array_equal(orig_arr, cached_arr), f"Pixel mismatch at idx {idx}"

            for key in target_orig:
                assert key in target_cached, f"Missing key {key} at idx {idx}"
                if torch.is_tensor(target_orig[key]):
                    assert torch.equal(target_orig[key], target_cached[key]), (
                        f"Tensor mismatch for key={key} at idx {idx}"
                    )

    def test_num_classes_proxy(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset: CacheDataset,
    ) -> None:
        assert cached_dataset.num_classes == base_dataset.num_classes

    def test_class_names_proxy(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset: CacheDataset,
    ) -> None:
        assert cached_dataset.class_names == base_dataset.class_names

    def test_label_map_proxy(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset: CacheDataset,
    ) -> None:
        assert cached_dataset.label_map == base_dataset.label_map


# ---------------------------------------------------------------------------
# Disk-specific tests
# ---------------------------------------------------------------------------
class TestCacheDatasetDisk:
    """Tests specific to the sharded disk backend."""

    def test_rebuild_flag(
        self,
        base_dataset: COCODetectionDataset,
    ) -> None:
        """Rebuild=True recreates the cache from scratch."""
        cache1 = CacheDataset(dataset=base_dataset, cache_type="disk", rebuild=True)
        cache2 = CacheDataset(dataset=base_dataset, cache_type="disk", rebuild=True)
        assert len(cache1) == len(cache2)
        img1, _ = cache1[0]
        img2, _ = cache2[0]
        assert np.array_equal(np.array(img1), np.array(img2))

    def test_cache_reuse(
        self,
        base_dataset: COCODetectionDataset,
        tmp_path: Path,
    ) -> None:
        """Second CacheDataset reuses existing shard files."""
        cache_dir = tmp_path / "shared_cache"
        cache1 = CacheDataset(
            dataset=base_dataset,
            cache_type="disk",
            cache_dir=cache_dir,
            rebuild=True,
        )
        # Populate the cache
        for idx in range(len(cache1)):
            _ = cache1[idx]

        # Verify shard files exist
        bin_files = list(cache_dir.glob("shard_*.bin"))
        assert len(bin_files) >= 1

        first_bin = sorted(bin_files)[0]
        mtime_after_build = first_bin.stat().st_mtime

        time.sleep(0.1)

        # Second cache should reuse existing shard
        cache2 = CacheDataset(
            dataset=base_dataset, cache_type="disk", cache_dir=cache_dir
        )
        _ = cache2[0]
        mtime_reuse = first_bin.stat().st_mtime

        assert mtime_reuse == mtime_after_build

    def test_lazy_cache_builds_incrementally(
        self,
        base_dataset: COCODetectionDataset,
        tmp_path: Path,
    ) -> None:
        """Lazy cache only writes samples as they are accessed."""
        cache_dir = tmp_path / "lazy_cache"
        cache = CacheDataset(
            dataset=base_dataset,
            cache_type="disk",
            cache_dir=cache_dir,
            rebuild=True,
        )

        # Shard files may exist but should be empty
        total_cached = sum(len(s) for s in cache._shards)
        assert total_cached == 0

        # Access first 5 samples
        for idx in range(5):
            _ = cache[idx]

        total_cached = sum(len(s) for s in cache._shards)
        assert total_cached == 5

        # Access remaining
        for idx in range(5, len(base_dataset)):
            _ = cache[idx]

        total_cached = sum(len(s) for s in cache._shards)
        assert total_cached == len(base_dataset)

    def test_multi_shard_read(
        self,
        base_dataset: COCODetectionDataset,
        tmp_path: Path,
    ) -> None:
        """Simulate multi-rank: write to different shards, read from all."""
        cache_dir = tmp_path / "multi_shard"
        cache_dir.mkdir()

        # Simulate rank 0 writing even indices
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            cache0 = CacheDataset(
                dataset=base_dataset,
                cache_type="disk",
                cache_dir=cache_dir,
            )
            for idx in range(0, len(base_dataset), 2):
                _ = cache0[idx]

        # Simulate rank 1 writing odd indices
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}):
            cache1 = CacheDataset(
                dataset=base_dataset,
                cache_type="disk",
                cache_dir=cache_dir,
            )
            for idx in range(1, len(base_dataset), 2):
                _ = cache1[idx]

        # Verify 2 shard files
        assert len(list(cache_dir.glob("shard_*.bin"))) == 2

        # New cache instance should read from both shards (all samples cached)
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            cache_reader = CacheDataset(
                dataset=base_dataset,
                cache_type="disk",
                cache_dir=cache_dir,
            )

        total_cached = sum(len(s) for s in cache_reader._shards)
        assert total_cached == len(base_dataset)

        # Verify all samples readable and correct
        for idx in range(len(base_dataset)):
            img_orig, _target_orig = base_dataset[idx]
            img_cached, _target_cached = cache_reader[idx]
            assert np.array_equal(np.array(img_orig), np.array(img_cached))


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------
class TestCacheDatasetBenchmark:
    """Measure iteration speed: cached vs uncached."""

    def test_disk_cached_faster_than_uncached(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset_disk: CacheDataset,
    ) -> None:
        """Disk-cached dataset should be at least usable (speedup > 0.5x)."""
        speedup = self._measure_speedup(base_dataset, cached_dataset_disk)
        assert speedup >= 0.5, (
            f"Expected >=0.5x speedup for disk cache, got {speedup:.2f}x"
        )

    def test_ram_cached_faster_than_uncached(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset_ram: CacheDataset,
    ) -> None:
        """RAM-cached dataset should be at least 2x faster."""
        speedup = self._measure_speedup(base_dataset, cached_dataset_ram)
        assert speedup >= 2.0, (
            f"Expected >=2x speedup for RAM cache, got {speedup:.2f}x"
        )

    @staticmethod
    def _measure_speedup(base: COCODetectionDataset, cached: CacheDataset) -> float:
        n_samples = len(base)
        _ = cached[0]  # warm up

        t0 = time.perf_counter()
        for _ in range(NUM_BENCHMARK_ITERS):
            for idx in range(n_samples):
                _img, _target = base[idx]
        uncached_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(NUM_BENCHMARK_ITERS):
            for idx in range(n_samples):
                _img, _target = cached[idx]
        cached_time = time.perf_counter() - t0

        speedup = uncached_time / cached_time

        print(f"\n{'=' * 60}")
        print(
            f"  CacheDataset Benchmark ({n_samples} images x "
            f"{NUM_BENCHMARK_ITERS} iters, type={cached._cache_type})"
        )
        print(f"{'=' * 60}")
        print(f"  Uncached: {uncached_time:.3f}s")
        print(f"  Cached:   {cached_time:.3f}s")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"{'=' * 60}")

        return speedup


# ---------------------------------------------------------------------------
# Auto Mode tests
# ---------------------------------------------------------------------------
class TestCacheDatasetAutoMode:
    """Tests for cache_type='auto' logic."""

    def test_auto_selects_ram(
        self,
        base_dataset: COCODetectionDataset,
    ) -> None:
        """Should select 'ram' if estimated size < 50% available RAM."""
        with mock.patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 1024**3  # 1 GB

            cache = CacheDataset(
                dataset=base_dataset,
                cache_type="auto",
            )
            assert cache._cache_type == "ram"
            assert cache._ram_cache is not None

    def test_auto_selects_disk(
        self,
        base_dataset: COCODetectionDataset,
    ) -> None:
        """Should select 'disk' if estimated size >= 50% available RAM."""
        with mock.patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 20 * 1024**2  # 20 MB

            cache = CacheDataset(
                dataset=base_dataset,
                cache_type="auto",
            )
            assert cache._cache_type == "disk"
            assert cache._ram_cache is None

    def test_auto_fallback_on_error(
        self,
        base_dataset: COCODetectionDataset,
    ) -> None:
        """Should default to 'disk' if psutil fails."""
        with mock.patch("psutil.virtual_memory", side_effect=ImportError):
            cache = CacheDataset(
                dataset=base_dataset,
                cache_type="auto",
            )
            assert cache._cache_type == "disk"
