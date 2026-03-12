#!/usr/bin/env python3
"""Debug: run a full DINOX forward pass with synthetic data.

Tests whether SimOTA assignment succeeds or throws an exception.
Run with: pixi run python scripts/debug_dinox_forward.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
from torchvision import tv_tensors

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))


def main():
    from object_detection_training.models.dinox.dinox import DINOX
    from object_detection_training.models.dinox.dinox_head import DINOXHead
    from object_detection_training.models.yolox import YOLOPAFPN

    print("=== DINOX Forward Pass Debug ===\n")

    num_classes = 80
    width = 0.5
    depth = 0.33

    # Build model (same as training config)
    backbone = YOLOPAFPN(depth=depth, width=width, in_channels=[256, 512, 1024])
    head = DINOXHead(
        num_classes=num_classes,
        width=width,
        in_channels=[256, 512, 1024],
        use_dfl=False,
        use_soft_labels=False,
        use_mal=False,
        use_dual_head=False,
        assigner_type="simota",
    )
    model = DINOX(backbone=backbone, head=head)
    head.initialize_biases(prior_prob=1e-2)
    model.train()

    # Create synthetic input (640x640 image batch)
    images = torch.randn(2, 3, 640, 640)

    # Create targets in CXCYWH pixel coords (what the pipeline produces)
    targets = [
        {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(
                    [
                        [200.0, 300.0, 100.0, 150.0],  # cx, cy, w, h
                        [400.0, 200.0, 80.0, 60.0],
                        [100.0, 500.0, 200.0, 100.0],
                    ]
                ),
                format="CXCYWH",
                canvas_size=(640, 640),
            ),
            "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
            "image_id": torch.tensor([1]),
        },
        {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(
                    [
                        [320.0, 320.0, 640.0, 640.0],  # Large box
                        [50.0, 50.0, 30.0, 30.0],  # Small box
                    ]
                ),
                format="CXCYWH",
                canvas_size=(640, 640),
            ),
            "labels": torch.tensor([5, 10], dtype=torch.int64),
            "image_id": torch.tensor([2]),
        },
    ]

    print("Targets:")
    for i, t in enumerate(targets):
        print(
            f"  Image {i}: boxes={t['boxes'].shape}, "
            f"format={t['boxes'].format}, "
            f"labels={t['labels']}"
        )
        print(f"    boxes:\n{t['boxes']}")

    # Test with bf16 autocast (simulating training with bf16-mixed)
    print("\n--- Test with bf16 autocast ---")
    device = "cpu"  # Use CPU since we might not have CUDA
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device)
        images = images.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

    try:
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            outputs_bf16 = model(images, targets)
        num_fg_bf16 = outputs_bf16.get("num_fg", 0)
        print(f"  bf16 forward SUCCEEDED, num_fg = {num_fg_bf16}")
        print(f"  total_loss = {outputs_bf16['total_loss'].item():.6f}")
        print(f"  conf_loss = {outputs_bf16['conf_loss'].item():.6f}")
        if num_fg_bf16 <= 1:
            print("  *** bf16 CAUSED num_fg=0! This is likely the issue ***")
    except Exception as e:
        print(f"  bf16 forward FAILED: {type(e).__name__}: {e}")

    if device != "cpu":
        model = model.cpu()
        images = images.cpu()
        targets = [
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

    # Run forward pass
    print("\nRunning forward pass (fp32)...")
    try:
        outputs = model(images, targets)
        print("\nForward pass SUCCEEDED!")
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(
                    f"  {k}: shape={v.shape}, value={v.item() if v.numel() == 1 else 'tensor'}"
                )
            elif isinstance(v, (int, float)):
                print(f"  {k}: {v}")
            elif isinstance(v, (list, tuple)):
                print(f"  {k}: {type(v).__name__} len={len(v)}")
            else:
                print(f"  {k}: {type(v).__name__}")

        num_fg = outputs.get("num_fg", 0)
        print(f"\n  *** num_fg = {num_fg} ***")
        if isinstance(num_fg, (int, float)) and num_fg <= 1:
            print("  WARNING: num_fg <= 1, SimOTA may not be matching anchors!")

    except Exception as e:
        print(f"\nForward pass FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    # Also test with bare tensors (not tv_tensors) to compare
    print("\n\n--- Test with plain tensors (not tv_tensors.BoundingBoxes) ---")
    targets_plain = [
        {
            "boxes": torch.tensor(
                [
                    [200.0, 300.0, 100.0, 150.0],
                    [400.0, 200.0, 80.0, 60.0],
                    [100.0, 500.0, 200.0, 100.0],
                ]
            ),
            "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
            "image_id": torch.tensor([1]),
        },
        {
            "boxes": torch.tensor(
                [
                    [320.0, 320.0, 640.0, 640.0],
                    [50.0, 50.0, 30.0, 30.0],
                ]
            ),
            "labels": torch.tensor([5, 10], dtype=torch.int64),
            "image_id": torch.tensor([2]),
        },
    ]

    try:
        outputs_plain = model(images, targets_plain)
        num_fg_plain = outputs_plain.get("num_fg", 0)
        print(f"  Forward pass SUCCEEDED, num_fg = {num_fg_plain}")
        print(f"  total_loss = {outputs_plain['total_loss'].item():.6f}")
        print(f"  conf_loss = {outputs_plain['conf_loss'].item():.6f}")
        print(f"  iou_loss = {outputs_plain['iou_loss'].item():.6f}")
        print(f"  cls_loss = {outputs_plain['cls_loss'].item():.6f}")
    except Exception as e:
        print(f"  Forward pass FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
