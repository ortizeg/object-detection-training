"""Tests for CacheDataset — correctness and benchmark.

Creates a temporary synthetic COCO dataset (in-memory), wraps it with
CacheDataset, and verifies:
1. Cached output is identical to uncached output.
2. Cached iteration is measurably faster than uncached.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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

        # Create a random RGB image and save as JPEG
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
def cached_dataset(base_dataset: COCODetectionDataset) -> CacheDataset:
    """Create a cached version of the dataset."""
    return CacheDataset(
        dataset=base_dataset,
        rebuild=True,
    )


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------
class TestCacheDatasetCorrectness:
    """Verify cached output matches uncached output exactly."""

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

            # Compare images (both should be PIL Images pre-transforms)
            orig_arr = np.array(img_orig)
            cached_arr = np.array(img_cached)

            # JPEG re-encoding in the synthetic dataset means raw pixels
            # should still match since we cache the decoded PIL image
            assert orig_arr.shape == cached_arr.shape, (
                f"Image shape mismatch at idx {idx}: "
                f"{orig_arr.shape} vs {cached_arr.shape}"
            )
            # Allow small tolerance for JPEG decode variations
            assert np.allclose(orig_arr, cached_arr, atol=1), (
                f"Image pixel mismatch at idx {idx}"
            )

            # Compare target tensors
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

    def test_rebuild_flag(
        self,
        base_dataset: COCODetectionDataset,
    ) -> None:
        """Rebuild=True recreates the cache from scratch."""
        cache1 = CacheDataset(dataset=base_dataset, rebuild=True)
        cache2 = CacheDataset(dataset=base_dataset, rebuild=True)
        assert len(cache1) == len(cache2)
        # Both should produce identical results
        img1, _ = cache1[0]
        img2, _ = cache2[0]
        assert np.array_equal(np.array(img1), np.array(img2))

    def test_cache_reuse(
        self,
        base_dataset: COCODetectionDataset,
        tmp_path: Path,
    ) -> None:
        """Second CacheDataset reuses existing DB without rebuilding."""
        cache_dir = tmp_path / "shared_cache"
        cache1 = CacheDataset(dataset=base_dataset, cache_dir=cache_dir, rebuild=True)
        db_path = cache_dir / f"{base_dataset.split}.db"
        assert db_path.exists()
        mtime_after_build = db_path.stat().st_mtime

        # Small sleep to detect any file modification
        time.sleep(0.1)

        # Creating another CacheDataset should NOT rebuild
        _cache2 = CacheDataset(dataset=base_dataset, cache_dir=cache_dir)
        mtime_reuse = db_path.stat().st_mtime

        # File should not have been modified (no rebuild)
        assert mtime_reuse == mtime_after_build

        # Should still work
        assert len(cache1) == len(base_dataset)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------
class TestCacheDatasetBenchmark:
    """Measure iteration speed: cached vs uncached."""

    def test_cached_faster_than_uncached(
        self,
        base_dataset: COCODetectionDataset,
        cached_dataset: CacheDataset,
    ) -> None:
        """Cached dataset iteration should be at least 1.5x faster."""
        n_samples = len(base_dataset)

        # Warm up cached dataset (already built by fixture)
        _ = cached_dataset[0]

        # Benchmark uncached
        t0 = time.perf_counter()
        for _ in range(NUM_BENCHMARK_ITERS):
            for idx in range(n_samples):
                _img, _target = base_dataset[idx]
        uncached_time = time.perf_counter() - t0

        # Benchmark cached
        t0 = time.perf_counter()
        for _ in range(NUM_BENCHMARK_ITERS):
            for idx in range(n_samples):
                _img, _target = cached_dataset[idx]
        cached_time = time.perf_counter() - t0

        speedup = uncached_time / cached_time

        # Print results for visibility in test output
        print(f"\n{'=' * 60}")
        print(
            f"  CacheDataset Benchmark ({n_samples} images x "
            f"{NUM_BENCHMARK_ITERS} iters)"
        )
        print(f"{'=' * 60}")
        print(f"  Uncached: {uncached_time:.3f}s")
        print(f"  Cached:   {cached_time:.3f}s")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"{'=' * 60}")

        assert speedup >= 1.5, (
            f"Expected ≥1.5x speedup, got {speedup:.2f}x "
            f"(uncached={uncached_time:.3f}s, cached={cached_time:.3f}s)"
        )
