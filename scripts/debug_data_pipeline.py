#!/usr/bin/env python3
"""Debug script to verify data pipeline produces correct box coordinates.

Run with: pixi run python scripts/debug_data_pipeline.py

Checks:
1. Raw dataset output (XYXY pixel coords)
2. After mosaic + post_transforms (CXCYWH pixel coords)
3. After collation (NestedTensor + targets)
4. SimOTA in-box check (are any anchors matched?)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision import tv_tensors

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))


def check_box_sanity(boxes: torch.Tensor, label: str, expected_range: float = 640.0):
    """Check if boxes are in expected pixel coordinate range."""
    if boxes.numel() == 0:
        print(f"  [{label}] WARNING: No boxes!")
        return False

    print(f"  [{label}] shape={boxes.shape}, dtype={boxes.dtype}")
    if isinstance(boxes, tv_tensors.BoundingBoxes):
        print(f"  [{label}] format={boxes.format}, canvas_size={boxes.canvas_size}")

    print(f"  [{label}] min={boxes.min().item():.4f}, max={boxes.max().item():.4f}")
    print(f"  [{label}] mean={boxes.mean().item():.4f}, std={boxes.std().item():.4f}")
    print(f"  [{label}] first 3 boxes:\n{boxes[:3]}")

    # Check for obvious issues
    if boxes.max().item() <= 1.0:
        print(f"  [{label}] *** LIKELY NORMALIZED [0,1] — expected pixel coords! ***")
        return False
    if boxes.max().item() > expected_range * 2:
        print(f"  [{label}] *** VALUES TOO LARGE — max={boxes.max().item():.1f} ***")
        return False
    if torch.isnan(boxes).any():
        print(f"  [{label}] *** CONTAINS NaN! ***")
        return False
    if torch.isinf(boxes).any():
        print(f"  [{label}] *** CONTAINS Inf! ***")
        return False

    # For CXCYWH, check that w,h are positive
    if boxes.shape[1] == 4:
        col2 = boxes[:, 2]
        col3 = boxes[:, 3]
        if (col2 <= 0).any() or (col3 <= 0).any():
            print(f"  [{label}] *** NEGATIVE w/h values (if CXCYWH)! ***")
            # Could be XYXY, so also check x2>x1, y2>y1
            if (col2 < boxes[:, 0]).any():
                print(f"  [{label}] *** x2 < x1 (invalid XYXY)! ***")
                return False

    print(f"  [{label}] OK")
    return True


def main():
    from object_detection_training.data.coco_detection_dataset import (
        COCODetectionDataset,
    )

    # Try to find COCO data
    coco_paths = [
        Path("/workspace/data/coco/train2017"),
        Path("/data/coco/train2017"),
        Path("data/coco/train2017"),
    ]
    data_path = None
    for p in coco_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        print("No COCO data found, using synthetic data for pipeline test")
        _test_synthetic()
        _test_mosaic_synthetic()
        return

    print(f"Using COCO data at: {data_path}")
    _test_real(data_path)


def _test_synthetic():
    """Test with synthetic data to verify transform chain."""
    from PIL import Image
    from torchvision.transforms import v2

    from object_detection_training.transforms.conversion import ToFloat32Tensor

    print("\n=== Synthetic Data Pipeline Test ===\n")

    # Create a synthetic image and target
    img = Image.new("RGB", (480, 640), color=(128, 128, 128))
    boxes_xyxy = torch.tensor(
        [
            [100.0, 150.0, 300.0, 400.0],  # A box in pixel XYXY
            [50.0, 50.0, 200.0, 200.0],
        ]
    )
    target = {
        "boxes": tv_tensors.BoundingBoxes(
            boxes_xyxy, format="XYXY", canvas_size=(640, 480)
        ),
        "labels": torch.tensor([0, 1], dtype=torch.int64),
        "image_id": torch.tensor([1]),
        "area": torch.tensor([50000.0, 22500.0]),
        "iscrowd": torch.zeros(2, dtype=torch.int64),
        "orig_size": torch.tensor([640, 480]),
        "size": torch.tensor([640, 480]),
    }

    print("1. Raw dataset output (XYXY pixel coords):")
    check_box_sanity(target["boxes"], "raw_xyxy", expected_range=640)

    # Apply post_mosaic_transforms (what the mosaic wrapper does)
    post_transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.0),  # Disable for deterministic test
            v2.Resize((640, 640)),
            ToFloat32Tensor(scale=False),
            v2.ConvertBoundingBoxFormat(format="CXCYWH"),
        ]
    )

    img_t, target_t = post_transforms(img, target)
    print("\n2. After transforms (should be CXCYWH pixel coords):")
    check_box_sanity(target_t["boxes"], "after_transforms", expected_range=640)

    # Verify CXCYWH values manually
    # Original XYXY: [100, 150, 300, 400]
    # After resize 480x640 -> 640x640: x scaled by 640/480=1.333, y unchanged
    # XYXY after resize: [133.3, 150, 400, 400]
    # CXCYWH: cx=(133.3+400)/2=266.7, cy=(150+400)/2=275, w=266.7, h=250
    print(f"\n  Expected first box (approx): cx~266.7, cy~275, w~266.7, h~250")
    actual = target_t["boxes"][0]
    print(
        f"  Actual first box: cx={actual[0]:.1f}, cy={actual[1]:.1f}, "
        f"w={actual[2]:.1f}, h={actual[3]:.1f}"
    )

    # Now simulate what SimOTA _get_in_boxes_info does
    print("\n3. SimOTA in-box check simulation:")
    _simulate_simota(target_t["boxes"])


def _test_mosaic_synthetic():
    """Test mosaic pipeline with synthetic dataset."""
    from torchvision.transforms import v2

    from object_detection_training.data.mosaic import MosaicMixupDataset
    from object_detection_training.transforms.conversion import ToFloat32Tensor

    print("\n=== Mosaic Pipeline Test (Synthetic) ===\n")

    class FakeDataset(torch.utils.data.Dataset):
        """Synthetic dataset returning PIL images with XYXY boxes."""

        def __len__(self):
            return 100

        def __getitem__(self, idx):
            # Random-sized image
            w, h = 480, 640
            from PIL import Image

            img = Image.new("RGB", (w, h), color=(100 + idx % 50, 128, 128))

            # 3 boxes in XYXY pixel coords (like DetectionDataset returns)
            boxes = torch.tensor(
                [
                    [50.0, 50.0, 200.0, 200.0],
                    [100.0, 300.0, 350.0, 500.0],
                    [250.0, 100.0, 400.0, 250.0],
                ]
            )
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    boxes, format="XYXY", canvas_size=(h, w)
                ),
                "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.tensor([22500.0, 50000.0, 22500.0]),
                "iscrowd": torch.zeros(3, dtype=torch.int64),
                "orig_size": torch.tensor([h, w]),
                "size": torch.tensor([h, w]),
            }
            return img, target

    post_transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.0),  # Disable for determinism
            ToFloat32Tensor(scale=False),
            v2.ConvertBoundingBoxFormat(format="CXCYWH"),
        ]
    )

    mosaic_ds = MosaicMixupDataset(
        FakeDataset(),
        input_height=640,
        input_width=640,
        mosaic_prob=1.0,
        mixup_prob=0.0,  # Disable mixup for clarity
        post_transforms=post_transforms,
        use_cache=True,
        max_cached_images=40,
    )

    for i in range(10):
        img_m, target_m = mosaic_ds[i]
        boxes = target_m["boxes"]
        print(
            f"  Sample {i}: img shape={img_m.shape}, "
            f"dtype={img_m.dtype}, range=[{img_m.min():.0f}, {img_m.max():.0f}]"
        )

        if boxes.numel() == 0:
            print(f"    WARNING: No boxes!")
            continue

        print(f"    boxes type={type(boxes).__name__}", end="")
        if isinstance(boxes, tv_tensors.BoundingBoxes):
            print(f", format={boxes.format}", end="")
        print(f", shape={boxes.shape}")
        print(f"    values: min={boxes.min():.1f}, max={boxes.max():.1f}")
        print(f"    first box: {boxes[0].tolist()}")

        # Verify CXCYWH: w and h should be positive
        if boxes.shape[1] == 4:
            w_vals = boxes[:, 2]
            h_vals = boxes[:, 3]
            if (w_vals <= 0).any() or (h_vals <= 0).any():
                print(f"    *** NEGATIVE WIDTH/HEIGHT DETECTED ***")
                print(f"    w: {w_vals.tolist()}")
                print(f"    h: {h_vals.tolist()}")

        n_matched = _simulate_simota(boxes)
        if n_matched == 0:
            print(f"    *** SimOTA MATCHED 0 ANCHORS ***")

    print("\n  Mosaic pipeline test complete.")


def _test_real(data_path: Path):
    """Test with real COCO data."""
    from torchvision.transforms import v2

    from object_detection_training.data.cache_dataset import CacheDataset
    from object_detection_training.data.coco_detection_dataset import (
        COCODetectionDataset,
    )
    from object_detection_training.data.mosaic import MosaicMixupDataset
    from object_detection_training.models.rfdetr.collate import collate_fn
    from object_detection_training.transforms.conversion import ToFloat32Tensor

    print("\n=== Real COCO Data Pipeline Test ===\n")

    # Create dataset WITHOUT transforms (as data module does for mosaic)
    dataset = COCODetectionDataset(
        root_path=str(data_path),
        split="train",
        transforms=None,
    )
    print(f"Dataset: {dataset}")
    print(f"  num_classes={dataset.num_classes}")
    print(f"  label_map (first 5): {dict(list(dataset.label_map.items())[:5])}")

    # Check raw dataset output
    print("\n--- Step 1: Raw dataset output ---")
    img, target = dataset[0]
    print(f"  Image: {img.size} (PIL)")
    check_box_sanity(target["boxes"], "raw_xyxy", expected_range=max(img.size))
    print(f"  Labels: {target['labels']}")

    # Create mosaic dataset (as data module does)
    post_transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ToFloat32Tensor(scale=False),
            v2.ConvertBoundingBoxFormat(format="CXCYWH"),
        ]
    )

    mosaic_dataset = MosaicMixupDataset(
        dataset,
        input_height=640,
        input_width=640,
        mosaic_prob=1.0,
        mixup_prob=1.0,
        post_transforms=post_transforms,
    )

    # Check mosaic output
    print("\n--- Step 2: After mosaic + post_transforms ---")
    for i in range(5):
        img_m, target_m = mosaic_dataset[i]
        print(f"\n  Sample {i}:")
        print(
            f"  Image: shape={img_m.shape}, dtype={img_m.dtype}, "
            f"min={img_m.min():.1f}, max={img_m.max():.1f}"
        )
        ok = check_box_sanity(target_m["boxes"], f"mosaic_{i}", expected_range=640)
        print(f"  Labels: {target_m['labels'][:5]}...")
        if not ok:
            print("  *** DATA PIPELINE BUG DETECTED ***")

        # SimOTA check
        if target_m["boxes"].numel() > 0:
            n_matched = _simulate_simota(target_m["boxes"])
            if n_matched == 0:
                print("  *** SimOTA MATCHED 0 ANCHORS — THIS IS THE BUG ***")

    # Check collated batch
    print("\n--- Step 3: After collation ---")
    batch = [mosaic_dataset[i] for i in range(4)]
    images, targets = collate_fn(batch)
    print(f"  Images: shape={images.tensors.shape}, dtype={images.tensors.dtype}")
    for i, t in enumerate(targets):
        print(f"  Target {i}:")
        check_box_sanity(t["boxes"], f"collated_{i}", expected_range=640)


def _simulate_simota(gt_boxes: torch.Tensor, input_size: int = 640) -> int:
    """Simulate SimOTA's _get_in_boxes_info to check anchor matching."""
    if gt_boxes.numel() == 0:
        print("    No GT boxes to match")
        return 0

    # Assume boxes are CXCYWH
    num_gt = gt_boxes.shape[0]

    # Build anchor grid (same as DINOX/YOLOX head)
    strides = [8, 16, 32]
    all_x_centers = []
    all_y_centers = []
    all_strides = []

    for stride in strides:
        h = input_size // stride
        w = input_size // stride
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        x_centers = (grid_x.flatten().float() + 0.5) * stride
        y_centers = (grid_y.flatten().float() + 0.5) * stride
        all_x_centers.append(x_centers)
        all_y_centers.append(y_centers)
        all_strides.append(torch.full_like(x_centers, stride))

    x_centers = torch.cat(all_x_centers)  # [N_anchors]
    y_centers = torch.cat(all_y_centers)
    anchor_strides = torch.cat(all_strides)
    total_anchors = x_centers.shape[0]

    # Check 1: anchor centers inside GT boxes (CXCYWH)
    # gt_boxes: [num_gt, 4] = [cx, cy, w, h]
    gt_l = (gt_boxes[:, 0] - 0.5 * gt_boxes[:, 2]).unsqueeze(1)  # [num_gt, 1]
    gt_r = (gt_boxes[:, 0] + 0.5 * gt_boxes[:, 2]).unsqueeze(1)
    gt_t = (gt_boxes[:, 1] - 0.5 * gt_boxes[:, 3]).unsqueeze(1)
    gt_b = (gt_boxes[:, 1] + 0.5 * gt_boxes[:, 3]).unsqueeze(1)

    x_c = x_centers.unsqueeze(0)  # [1, N_anchors]
    y_c = y_centers.unsqueeze(0)

    in_box = (
        (x_c > gt_l) & (x_c < gt_r) & (y_c > gt_t) & (y_c < gt_b)
    )  # [num_gt, N_anchors]
    in_box_any = in_box.sum(dim=0) > 0  # [N_anchors]

    # Check 2: within center radius
    center_radius = 2.5
    gt_cx = gt_boxes[:, 0].unsqueeze(1)
    gt_cy = gt_boxes[:, 1].unsqueeze(1)
    stride_row = anchor_strides.unsqueeze(0)

    in_center = (
        (x_c > gt_cx - center_radius * stride_row)
        & (x_c < gt_cx + center_radius * stride_row)
        & (y_c > gt_cy - center_radius * stride_row)
        & (y_c < gt_cy + center_radius * stride_row)
    )
    in_center_any = in_center.sum(dim=0) > 0

    # Combined
    fg_mask = in_box_any | in_center_any
    n_matched = fg_mask.sum().item()

    print(
        f"    SimOTA check: {num_gt} GT boxes, {total_anchors} anchors, "
        f"{n_matched} anchors in-box/in-center"
    )
    print(
        f"    GT box ranges: cx=[{gt_boxes[:, 0].min():.1f},{gt_boxes[:, 0].max():.1f}], "
        f"cy=[{gt_boxes[:, 1].min():.1f},{gt_boxes[:, 1].max():.1f}], "
        f"w=[{gt_boxes[:, 2].min():.1f},{gt_boxes[:, 2].max():.1f}], "
        f"h=[{gt_boxes[:, 3].min():.1f},{gt_boxes[:, 3].max():.1f}]"
    )
    print(
        f"    Anchor range: x=[{x_centers.min():.1f},{x_centers.max():.1f}], "
        f"y=[{y_centers.min():.1f},{y_centers.max():.1f}]"
    )

    return n_matched


if __name__ == "__main__":
    main()
