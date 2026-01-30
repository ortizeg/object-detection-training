#!/usr/bin/env python
# ruff: noqa: T201
"""Verify YOLOX model weight loading correctness.

Compares:
1. Freshly initialized model vs model loaded from checkpoint
2. Forward pass outputs with and without checkpoint loading
3. State dict key matching between our model and official checkpoint

Usage:
    pixi run python scripts/verify_yolox_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_detection_training.models.yolox import (
    YOLOPAFPN,
    YOLOX,
    YOLOXHead,
)


def build_yolox_nano(num_classes: int = 80) -> YOLOX:
    """Build a YOLOX nano model (matches official config)."""
    depth, width, depthwise = 0.33, 0.25, True
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(
        depth=depth,
        width=width,
        in_channels=in_channels,
        depthwise=depthwise,
    )
    head = YOLOXHead(
        num_classes=num_classes,
        width=width,
        in_channels=in_channels,
        depthwise=depthwise,
    )
    model = YOLOX(backbone=backbone, head=head)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

    model.head.initialize_biases(prior_prob=1e-2)
    return model


def verify_checkpoint_loading():
    """Verify that checkpoint loading produces expected results."""
    print("=" * 60)
    print("YOLOX Weight Loading Verification")
    print("=" * 60)

    cache_dir = Path.home() / ".cache" / "yolox"
    checkpoint_path = cache_dir / "yolox_nano.pth"

    if not checkpoint_path.exists():
        print(f"\nCheckpoint not found at {checkpoint_path}")
        print("Downloading...")
        import urllib.request

        cache_dir.mkdir(parents=True, exist_ok=True)
        url = (
            "https://github.com/Megvii-BaseDetection/YOLOX"
            "/releases/download/0.1.1rc0/yolox_nano.pth"
        )
        urllib.request.urlretrieve(url, checkpoint_path)  # noqa: S310
        print("Downloaded.")

    # 1. Build fresh model (80 classes)
    model_fresh = build_yolox_nano(num_classes=80)
    n_params = sum(p.numel() for p in model_fresh.parameters())
    print(f"\n1. Fresh model built with {n_params} params")

    # 2. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    print(f"\n2. Checkpoint keys: {len(state_dict)}")
    model_sd = model_fresh.state_dict()
    print(f"   Model state_dict keys: {len(model_sd)}")

    # 3. Compare keys
    ckpt_keys = set(state_dict.keys())
    model_keys = set(model_sd.keys())
    common = model_keys & ckpt_keys

    print("\n3. Key comparison:")
    print(f"   Common keys: {len(common)}")

    missing = model_keys - ckpt_keys
    print(f"   Missing in checkpoint: {len(missing)}")
    for k in sorted(missing):
        print(f"     - {k}")

    extra = ckpt_keys - model_keys
    print(f"   Extra in checkpoint: {len(extra)}")
    for k in sorted(extra):
        print(f"     - {k}")

    # 4. Check shape matches
    mismatches = []
    for k in common:
        if state_dict[k].shape != model_sd[k].shape:
            mismatches.append(
                f"  {k}: ckpt={state_dict[k].shape} " f"vs model={model_sd[k].shape}"
            )

    print(f"\n4. Shape mismatches: {len(mismatches)}")
    for m in mismatches:
        print(m)

    # 5. Load weights and verify
    model_loaded = build_yolox_nano(num_classes=80)
    result = model_loaded.load_state_dict(state_dict, strict=False)
    print("\n5. Load result:")
    print(f"   Missing keys: {len(result.missing_keys)}")
    print(f"   Unexpected keys: {len(result.unexpected_keys)}")

    # 6. Compare forward pass
    print("\n6. Forward pass comparison:")
    dummy_input = torch.randn(1, 3, 416, 416)

    model_fresh.eval()
    model_loaded.eval()

    with torch.no_grad():
        out_fresh = model_fresh(dummy_input)
        out_loaded = model_loaded(dummy_input)

    print(f"   Fresh output shape: {out_fresh.shape}")
    print(f"   Loaded output shape: {out_loaded.shape}")

    diff = (out_fresh - out_loaded).abs().mean().item()
    print(f"   Mean absolute diff: {diff:.6f}")
    print(f"   Outputs differ (expected): {diff > 0.01}")

    # 7. Test with different num_classes
    print("\n7. Testing with num_classes=10 (basketball):")
    model_10 = build_yolox_nano(num_classes=10)
    model_10_sd = model_10.state_dict()

    matched, skipped_cls, skipped_shape = 0, 0, 0
    for k, v in state_dict.items():
        if k in model_10_sd:
            if "cls_preds" in k and v.shape[0] != 10:
                skipped_cls += 1
            elif v.shape == model_10_sd[k].shape:
                matched += 1
            else:
                skipped_shape += 1

    print(f"   Matched: {matched}")
    print(f"   Skipped (cls dim mismatch): {skipped_cls}")
    print(f"   Skipped (other shape mismatch): {skipped_shape}")

    # 8. Verify Lightning wrapper
    print("\n8. Testing Lightning wrapper:")
    from object_detection_training.models.yolox_lightning import (
        YOLOXLightningModel,
    )

    lightning_model = YOLOXLightningModel(
        variant="nano",
        num_classes=10,
        download_pretrained=True,
    )

    lightning_sd = lightning_model.model.state_dict()
    direct_sd = model_loaded.state_dict()

    bb_match = 0
    bb_total = 0
    for k in direct_sd:
        if "backbone" in k and k in lightning_sd:
            bb_total += 1
            if torch.equal(direct_sd[k], lightning_sd[k]):
                bb_match += 1

    print(f"   Backbone weight match: {bb_match}/{bb_total}")
    if bb_match == bb_total:
        print("   PASS: All backbone weights match!")
    else:
        print("   FAIL: Some backbone weights differ!")
        for k in direct_sd:
            if "backbone" not in k or k not in lightning_sd:
                continue
            if not torch.equal(direct_sd[k], lightning_sd[k]):
                d = (direct_sd[k] - lightning_sd[k]).abs().max()
                print(f"     {k}: max_diff={d:.8f}")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    verify_checkpoint_loading()
