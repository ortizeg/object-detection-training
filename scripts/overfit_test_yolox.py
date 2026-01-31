#!/usr/bin/env python
# ruff: noqa: T201
"""Quick overfit test for YOLOX nano on a small basketball subset.

Tests whether the model can overfit a small batch, verifying:
1. SimOTA assignment produces foreground matches (num_fg > 0)
2. Losses decrease over iterations
3. The model can actually learn

Usage:
    pixi run python scripts/overfit_test_yolox.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_detection_training.data.coco_data_module import COCODataModule
from object_detection_training.models.yolox_lightning import YOLOXLightningModel

DATA_ROOT = Path(
    "/Users/ortizeg/1Projects"
    "/\u26f9\ufe0f\u200d\u2642\ufe0f Next Play"
    "/data/basketball-player-detection-3-subset"
)


def main():
    print("=" * 60)
    print("YOLOX Nano Overfit Test")
    print("=" * 60)

    model = YOLOXLightningModel(
        variant="nano",
        num_classes=10,
        download_pretrained=True,
        learning_rate=0.01,
        weight_decay=5e-4,
        warmup_epochs=0,
        input_height=416,
        input_width=416,
    )
    model.train()

    dm = COCODataModule(
        train_path=str(DATA_ROOT / "train"),
        val_path=str(DATA_ROOT / "valid"),
        batch_size=4,
        num_workers=0,
        input_height=416,
        input_width=416,
        multi_scale=False,
        expanded_scales=False,
        skip_random_resize=True,
        patch_size=16,
        num_windows=2,
        pin_memory=False,
        persistent_workers=False,
        square_resize_div_64=True,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        selected_categories=[
            "ball",
            "ball-in-basket",
            "number",
            "player",
            "player-in-possession",
            "player-jump-shot",
            "player-layup-dunk",
            "player-shot-block",
            "referee",
            "rim",
        ],
    )

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    images, targets = batch

    img_tensor = images.tensors if hasattr(images, "tensors") else images

    print(f"\nBatch: {img_tensor.shape}, {len(targets)} targets")
    for i, t in enumerate(targets):
        n = t["boxes"].shape[0]
        labs = t["labels"].tolist()
        print(f"  Image {i}: {n} boxes, labels={labs}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    header = (
        f"{'Iter':<6} {'Loss':>10} {'IoU':>10} "
        f"{'Obj':>10} {'Cls':>10} {'L1':>10} {'num_fg':>8}"
    )
    print(f"\n{header}")
    print("-" * 70)

    losses_track = []
    for i in range(100):
        optimizer.zero_grad()
        outputs = model(img_tensor, targets)
        loss = outputs["total_loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARN: Loss is {loss.item()} at iter {i}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses_track.append(loss.item())

        if i % 5 == 0 or i < 10:
            iou_l = outputs["iou_loss"]
            obj_l = outputs["conf_loss"]
            cls_l = outputs["cls_loss"]
            l1_l = outputs["l1_loss"]
            nfg = outputs["num_fg"]
            iou_l = iou_l.item() if isinstance(iou_l, torch.Tensor) else iou_l
            obj_l = obj_l.item() if isinstance(obj_l, torch.Tensor) else obj_l
            cls_l = cls_l.item() if isinstance(cls_l, torch.Tensor) else cls_l
            l1_l = l1_l.item() if isinstance(l1_l, torch.Tensor) else l1_l
            nfg = nfg.item() if isinstance(nfg, torch.Tensor) else nfg
            print(
                f"{i:<6} {loss.item():>10.4f} {iou_l:>10.4f} "
                f"{obj_l:>10.4f} {cls_l:>10.4f} "
                f"{l1_l:>10.4f} {nfg:>8.1f}"
            )

    print("\n" + "=" * 60)
    print("OVERFIT TEST RESULTS")
    print("=" * 60)

    if len(losses_track) < 10:
        print("FAIL: Not enough valid iterations")
        return

    first_10 = sum(losses_track[:10]) / 10
    last_10 = sum(losses_track[-10:]) / 10
    reduction = (first_10 - last_10) / first_10 * 100

    print(f"First 10 avg loss:  {first_10:.4f}")
    print(f"Last 10 avg loss:   {last_10:.4f}")
    print(f"Loss reduction:     {reduction:.1f}%")

    if last_10 < first_10:
        print("\nPASS: Loss is decreasing - model is learning!")
    else:
        print("\nFAIL: Loss is NOT decreasing!")

    if reduction > 20:
        print("PASS: Significant loss reduction (>20%)")
    elif reduction > 5:
        print("WARN: Modest loss reduction (5-20%)")
    else:
        print("FAIL: Minimal loss reduction (<5%)")


if __name__ == "__main__":
    main()
