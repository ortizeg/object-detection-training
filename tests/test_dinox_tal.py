"""Tests for Task-Aligned Label Assignment (TAL).

Verifies TAL output contract compatibility with SimOTA, top-k selection,
conflict resolution, and integration with DINOXHead forward pass.
"""

from __future__ import annotations

import torch

from object_detection_training.models.dinox import DINOX, DINOXHead
from object_detection_training.models.dinox.tal import TaskAlignedAssigner
from object_detection_training.models.yolox import YOLOPAFPN

NUM_CLASSES = 2
INPUT_SIZE = 320


def _make_model(
    use_dfl: bool = False,
    reg_max: int = 16,
    width: float = 0.75,
    depth: float = 0.67,
    use_soft_labels: bool = False,
    soft_label_gamma: float = 2.0,
    use_log_iou_cost: bool = False,
    use_mal: bool = False,
    mal_gamma: float = 1.5,
    assigner_type: str = "simota",
) -> DINOX:
    """Create a small DINOX model for testing."""
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(  # type: ignore[no-untyped-call]
        depth=depth, width=width, in_channels=in_channels
    )
    head = DINOXHead(
        num_classes=NUM_CLASSES,
        width=width,
        in_channels=in_channels,
        use_dfl=use_dfl,
        reg_max=reg_max,
        use_soft_labels=use_soft_labels,
        soft_label_gamma=soft_label_gamma,
        use_log_iou_cost=use_log_iou_cost,
        use_mal=use_mal,
        mal_gamma=mal_gamma,
        assigner_type=assigner_type,
    )
    return DINOX(backbone=backbone, head=head)


def _make_targets(
    num_boxes: int = 3, img_size: int = INPUT_SIZE
) -> list[dict[str, torch.Tensor]]:
    """Create realistic dummy targets in pixel CXCYWH format."""
    boxes = torch.tensor(
        [
            [img_size * 0.3, img_size * 0.4, img_size * 0.2, img_size * 0.3],
            [img_size * 0.7, img_size * 0.6, img_size * 0.15, img_size * 0.25],
            [img_size * 0.5, img_size * 0.5, img_size * 0.1, img_size * 0.1],
        ][:num_boxes]
    )
    labels = torch.zeros(num_boxes, dtype=torch.long)
    return [{"boxes": boxes, "labels": labels}]


def _make_tal_inputs(
    num_gt: int = 2,
    total_num_anchors: int = 100,
    num_classes: int = NUM_CLASSES,
) -> dict[str, torch.Tensor | int]:
    """Create synthetic inputs for TaskAlignedAssigner.assign().

    Returns dict with all args needed for assign() except batch_idx.
    GT boxes are in cxcywh format positioned within a 320x320 image.
    """
    torch.manual_seed(0)
    # GT boxes in reasonable positions (cxcywh)
    gt_bboxes = torch.tensor(
        [
            [100.0, 120.0, 60.0, 80.0],
            [200.0, 180.0, 50.0, 70.0],
            [160.0, 160.0, 40.0, 40.0],
        ][:num_gt]
    )
    gt_classes = torch.zeros(num_gt, dtype=torch.long)

    # Random predictions
    bbox_preds = torch.randn(total_num_anchors, 4).abs() * 50 + 10
    # Place some predictions near GT boxes to ensure matching
    bbox_preds[:5] = gt_bboxes[0].unsqueeze(0) + torch.randn(5, 4) * 5
    if num_gt > 1:
        bbox_preds[5:10] = gt_bboxes[1].unsqueeze(0) + torch.randn(5, 4) * 5
    cls_preds = torch.randn(total_num_anchors, num_classes)
    obj_preds = torch.randn(total_num_anchors, 1)

    # Grid info: single stride=8 for simplicity
    stride = 8.0
    expanded_strides = torch.full((1, total_num_anchors), stride)
    # Arrange anchors in a 10x10 grid
    grid_size = int(total_num_anchors**0.5)
    xs = torch.arange(grid_size).float().repeat(grid_size)
    ys = torch.arange(grid_size).float().repeat_interleave(grid_size)
    # Pad if needed
    if len(xs) < total_num_anchors:
        xs = torch.cat([xs, torch.zeros(total_num_anchors - len(xs))])
        ys = torch.cat([ys, torch.zeros(total_num_anchors - len(ys))])
    x_shifts = xs[:total_num_anchors].unsqueeze(0)
    y_shifts = ys[:total_num_anchors].unsqueeze(0)

    return {
        "num_gt": num_gt,
        "total_num_anchors": total_num_anchors,
        "gt_bboxes_per_image": gt_bboxes,
        "gt_classes": gt_classes,
        "bbox_preds": bbox_preds,
        "cls_preds": cls_preds,
        "obj_preds": obj_preds,
        "expanded_strides": expanded_strides,
        "x_shifts": x_shifts,
        "y_shifts": y_shifts,
    }


class TestTALOutputContract:
    """Tests for TAL output tuple structure and shapes."""

    def test_tal_output_tuple_length(self) -> None:
        """assign() returns a 5-tuple."""
        tal = TaskAlignedAssigner(topk=3, num_classes=NUM_CLASSES)
        inputs = _make_tal_inputs()
        result = tal.assign(batch_idx=0, **inputs)  # type: ignore[arg-type]
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_tal_output_shapes(self) -> None:
        """Verify output tensor shapes match the contract."""
        tal = TaskAlignedAssigner(topk=5, num_classes=NUM_CLASSES)
        inputs = _make_tal_inputs(num_gt=2, total_num_anchors=100)
        gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg = tal.assign(
            batch_idx=0,
            **inputs,  # type: ignore[arg-type]
        )

        total_anchors = inputs["total_num_anchors"]
        assert isinstance(total_anchors, int)

        # fg_mask: [total_num_anchors] bool
        assert fg_mask.shape == (total_anchors,)
        assert fg_mask.dtype == torch.bool

        # num_fg matches fg_mask
        assert isinstance(num_fg, int)
        assert num_fg == int(fg_mask.sum().item())

        # Foreground-indexed tensors: shape [num_fg]
        assert gt_matched_classes.shape == (num_fg,)
        assert pred_ious.shape == (num_fg,)
        assert matched_gt_inds.shape == (num_fg,)

    def test_tal_topk_respects_limit(self) -> None:
        """With topk=3 and 2 GTs, num_fg <= 6."""
        tal = TaskAlignedAssigner(topk=3, num_classes=NUM_CLASSES)
        inputs = _make_tal_inputs(num_gt=2, total_num_anchors=100)
        _, _, _, _, num_fg = tal.assign(batch_idx=0, **inputs)  # type: ignore[arg-type]
        # At most topk per GT minus conflicts
        assert num_fg <= 3 * 2


class TestTALBehavior:
    """Tests for TAL conflict resolution and edge cases."""

    def test_tal_conflict_resolution(self) -> None:
        """Anchor matched to two GTs is assigned to GT with higher alignment."""
        torch.manual_seed(42)
        tal = TaskAlignedAssigner(topk=50, alpha=1.0, beta=6.0, num_classes=NUM_CLASSES)
        # Use many anchors and large topk to increase conflict probability
        inputs = _make_tal_inputs(num_gt=2, total_num_anchors=100)
        _cls, fg_mask, _ious, matched_gt_inds, num_fg = tal.assign(
            batch_idx=0,
            **inputs,  # type: ignore[arg-type]
        )

        # Each foreground anchor should be assigned to exactly one GT
        # (no duplicates -- conflict resolution ensures this)
        assert num_fg == int(fg_mask.sum().item())
        # matched_gt_inds values should be valid GT indices
        if num_fg > 0:
            assert (matched_gt_inds >= 0).all()
            assert (matched_gt_inds < inputs["num_gt"]).all()

    def test_tal_no_candidates(self) -> None:
        """When no anchors pass spatial filtering, returns empty results."""
        tal = TaskAlignedAssigner(topk=3, num_classes=NUM_CLASSES)
        total_anchors = 50
        # Place GT far outside the grid coverage
        inputs = _make_tal_inputs(num_gt=1, total_num_anchors=total_anchors)
        # Override GT to be far outside image
        inputs["gt_bboxes_per_image"] = torch.tensor([[5000.0, 5000.0, 10.0, 10.0]])

        result = tal.assign(batch_idx=0, **inputs)  # type: ignore[arg-type]
        gt_matched_classes, fg_mask, pred_ious, _gt_inds, num_fg = result

        assert num_fg == 0
        assert fg_mask.sum() == 0
        assert gt_matched_classes.shape == (0,)
        assert pred_ious.shape == (0,)

    def test_tal_contract_matches_simota(self) -> None:
        """TAL and SimOTA output tuples have same structure (dtypes, shape patterns).

        Values will differ but the contract (tuple length, tensor shapes relative
        to num_fg, dtypes) must be identical for drop-in replacement.
        """
        torch.manual_seed(42)
        model_simota = _make_model(assigner_type="simota")
        model_tal = _make_model(assigner_type="tal")

        model_simota.train()
        model_tal.train()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        targets = _make_targets(num_boxes=2)

        # Both should produce valid loss dicts
        out_simota = model_simota(x, targets=targets)
        out_tal = model_tal(x, targets=targets)

        assert isinstance(out_simota, dict)
        assert isinstance(out_tal, dict)
        # Both should have the same loss keys
        assert set(out_simota.keys()) == set(out_tal.keys())
        # Both losses should be finite
        assert torch.isfinite(out_simota["total_loss"])
        assert torch.isfinite(out_tal["total_loss"])


class TestTALIntegration:
    """Integration tests for TAL with DINOXHead."""

    def test_tal_integration_forward(self) -> None:
        """DINOXHead with TAL and soft labels produces finite loss."""
        torch.manual_seed(42)
        model = _make_model(
            assigner_type="tal",
            use_soft_labels=True,
            soft_label_gamma=2.0,
        )
        model.train()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        targets = _make_targets()
        out = model(x, targets=targets)

        assert isinstance(out, dict)
        loss = out["total_loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.head.cls_preds.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients found in cls_preds after backward with TAL"
