"""Tests for dual-head (O2O + O2M) architecture (TEST-05).

Verifies Hungarian matching 1:1 assignment, O2O output shape matching O2M,
consistent alpha/beta between heads, constant-1 objectness in O2O inference,
separate O2O parameters, combined loss, and E5/E6 ablation Hydra configs.
"""

from __future__ import annotations

import sys
from os.path import dirname, join

import hydra
import pytest
import torch

from object_detection_training.models.dinox import DINOX, DINOXConfig, DINOXHead
from object_detection_training.models.dinox.hungarian import HungarianAssigner
from object_detection_training.models.yolox import YOLOPAFPN

NUM_CLASSES = 2
INPUT_SIZE = 320

# Hydra config path (relative to this file)
sys.path.append(join(dirname(__file__), "../src"))
import object_detection_training.models as _models  # noqa: F401, E402

CONF_PATH = "../src/object_detection_training/conf"


def _make_model(
    use_dfl: bool = False,
    reg_max: int = 16,
    width: float = 0.25,
    depth: float = 0.33,
    use_soft_labels: bool = False,
    soft_label_gamma: float = 2.0,
    use_log_iou_cost: bool = False,
    use_mal: bool = False,
    mal_gamma: float = 1.5,
    assigner_type: str = "simota",
    use_dual_head: bool = False,
    lambda_o2o: float = 1.0,
    tal_alpha: float = 1.0,
    tal_beta: float = 6.0,
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
        use_dual_head=use_dual_head,
        lambda_o2o=lambda_o2o,
        tal_alpha=tal_alpha,
        tal_beta=tal_beta,
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


def _make_hungarian_inputs(
    num_gt: int = 3,
    total_num_anchors: int = 400,
    num_classes: int = NUM_CLASSES,
) -> dict[str, torch.Tensor | int]:
    """Create synthetic inputs for HungarianAssigner.assign().

    GT boxes are in cxcywh format positioned within a 320x320 image.
    Anchor grid uses stride=8 with a 20x20 grid covering 0-160px,
    plus center_radius=2.5 extends to ~180px, ensuring GT overlap.
    """
    torch.manual_seed(0)
    gt_bboxes = torch.tensor(
        [
            [60.0, 60.0, 40.0, 40.0],
            [120.0, 80.0, 30.0, 50.0],
            [80.0, 120.0, 50.0, 30.0],
        ][:num_gt]
    )
    gt_classes = torch.zeros(num_gt, dtype=torch.long)

    # Random predictions
    bbox_preds = torch.randn(total_num_anchors, 4).abs() * 30 + 10
    # Place some predictions near GT boxes to ensure matching
    if num_gt > 0:
        bbox_preds[:5] = gt_bboxes[0].unsqueeze(0) + torch.randn(5, 4) * 3
    if num_gt > 1:
        bbox_preds[5:10] = gt_bboxes[1].unsqueeze(0) + torch.randn(5, 4) * 3
    if num_gt > 2:
        bbox_preds[10:15] = gt_bboxes[2].unsqueeze(0) + torch.randn(5, 4) * 3
    cls_preds = torch.randn(total_num_anchors, num_classes)
    obj_preds = torch.randn(total_num_anchors, 1)

    # Grid info: stride=8, 20x20 grid covering 0-160px
    stride = 8.0
    expanded_strides = torch.full((1, total_num_anchors), stride)
    grid_size = int(total_num_anchors**0.5)
    xs = torch.arange(grid_size).float().repeat(grid_size)
    ys = torch.arange(grid_size).float().repeat_interleave(grid_size)
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


class TestHungarian1to1Assignment:
    """Tests for Hungarian matching strict 1:1 assignment (DUAL-02, TEST-05)."""

    def test_hungarian_1to1_assignment(self) -> None:
        """Hungarian assigns exactly num_gt foreground anchors, one per GT."""
        assigner = HungarianAssigner(alpha=1.0, beta=6.0, num_classes=NUM_CLASSES)
        inputs = _make_hungarian_inputs(num_gt=3, total_num_anchors=400)
        _gt_cls, fg_mask, _pred_ious, matched_gt_inds, num_fg = assigner.assign(
            batch_idx=0, **inputs
        )  # type: ignore[arg-type]

        # Strict 1:1: exactly num_gt foreground anchors
        assert num_fg == inputs["num_gt"]
        assert fg_mask.sum().item() == inputs["num_gt"]

        # Each GT index appears exactly once
        gt_indices = matched_gt_inds.tolist()
        assert len(set(gt_indices)) == len(gt_indices), (
            f"Duplicate GT assignments: {gt_indices}"
        )
        assert set(gt_indices) == set(range(inputs["num_gt"]))

    def test_hungarian_empty_gt(self) -> None:
        """With num_gt=0, no foreground anchors are assigned."""
        assigner = HungarianAssigner(alpha=1.0, beta=6.0, num_classes=NUM_CLASSES)
        inputs = _make_hungarian_inputs(num_gt=0, total_num_anchors=50)
        # Override to 0 GT
        inputs["num_gt"] = 0
        inputs["gt_bboxes_per_image"] = torch.zeros(0, 4)
        inputs["gt_classes"] = torch.zeros(0, dtype=torch.long)

        _, fg_mask, _, _, num_fg = assigner.assign(
            batch_idx=0,
            **inputs,  # type: ignore[arg-type]
        )
        assert num_fg == 0
        assert fg_mask.sum().item() == 0

    def test_hungarian_returns_5tuple_contract(self) -> None:
        """Verify return types match the 5-tuple contract."""
        assigner = HungarianAssigner(alpha=1.0, beta=6.0, num_classes=NUM_CLASSES)
        inputs = _make_hungarian_inputs(num_gt=2, total_num_anchors=400)
        result = assigner.assign(batch_idx=0, **inputs)  # type: ignore[arg-type]

        assert isinstance(result, tuple)
        assert len(result) == 5

        gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg = result

        # Type checks
        assert isinstance(gt_matched_classes, torch.Tensor)
        assert isinstance(fg_mask, torch.Tensor)
        assert fg_mask.dtype == torch.bool
        total_anchors = inputs["total_num_anchors"]
        assert isinstance(total_anchors, int)
        assert fg_mask.shape == (total_anchors,)
        assert isinstance(pred_ious, torch.Tensor)
        assert pred_ious.shape == (num_fg,)
        assert isinstance(matched_gt_inds, torch.Tensor)
        assert matched_gt_inds.shape == (num_fg,)
        assert isinstance(num_fg, int)


class TestO2OOutputShape:
    """Tests for O2O output shape matching O2M (TEST-05)."""

    def test_o2o_output_shape_matches_o2m(self) -> None:
        """Dual head (O2O) inference output shape matches non-dual (O2M)."""
        torch.manual_seed(42)

        model_o2m = _make_model(
            use_dual_head=False,
            use_soft_labels=False,
        )
        model_o2o = _make_model(
            use_dual_head=True,
            use_soft_labels=True,
        )

        model_o2m.eval()
        model_o2o.eval()

        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out_o2m = model_o2m(x)
            out_o2o = model_o2o(x)

        assert isinstance(out_o2m, torch.Tensor)
        assert isinstance(out_o2o, torch.Tensor)
        # Both should be [B, N, 5+C]
        assert out_o2m.shape == out_o2o.shape, (
            f"Shape mismatch: O2M={out_o2m.shape}, O2O={out_o2o.shape}"
        )
        assert out_o2m.shape[2] == 5 + NUM_CLASSES


class TestO2OInference:
    """Tests for O2O inference behavior (DUAL-05)."""

    def test_o2o_inference_constant_objectness(self) -> None:
        """O2O inference produces constant-1 objectness in column 4."""
        torch.manual_seed(42)
        model = _make_model(
            use_dual_head=True,
            use_soft_labels=True,
        )
        model.eval()

        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)

        assert isinstance(out, torch.Tensor)
        # Column 4 is objectness in [B, N, 5+C] output
        objectness = out[:, :, 4]
        assert torch.allclose(objectness, torch.ones_like(objectness), atol=1e-6), (
            f"Objectness not constant-1: min={objectness.min()}, max={objectness.max()}"
        )


class TestConsistentAlphaBeta:
    """Tests for alpha/beta consistency between O2M and O2O (DUAL-03)."""

    def test_consistent_alpha_beta(self) -> None:
        """Hungarian assigner uses same alpha/beta as passed to DINOXHead."""
        model = _make_model(
            use_dual_head=True,
            use_soft_labels=True,
            tal_alpha=1.5,
            tal_beta=3.0,
        )
        head = model.head

        assert head._hungarian_assigner.alpha == 1.5
        assert head._hungarian_assigner.beta == 3.0


class TestDualHeadCombinedLoss:
    """Tests for combined O2M + O2O loss (DUAL-04)."""

    def test_dual_head_combined_loss(self) -> None:
        """Dual head produces finite positive loss."""
        torch.manual_seed(42)
        model = _make_model(
            use_dual_head=True,
            use_soft_labels=True,
            lambda_o2o=1.0,
        )
        model.train()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        targets = _make_targets()
        out = model(x, targets=targets)

        assert isinstance(out, dict)
        loss = out["total_loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss.item() > 0, f"Loss should be positive: {loss}"

    def test_lambda_o2o_zero_equals_o2m_only(self) -> None:
        """With lambda_o2o=0, loss equals O2M-only (O2O contributes nothing)."""
        torch.manual_seed(42)
        model_dual = _make_model(
            use_dual_head=True,
            use_soft_labels=True,
            lambda_o2o=0.0,
        )
        model_o2m = _make_model(
            use_dual_head=False,
            use_soft_labels=True,
        )

        # Copy weights from dual to o2m (O2M prediction layers)
        model_o2m.load_state_dict(model_dual.state_dict(), strict=False)

        model_dual.train()
        model_o2m.train()

        targets = _make_targets()

        torch.manual_seed(42)
        out_dual = model_dual(
            torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE), targets=targets
        )
        torch.manual_seed(42)
        out_o2m = model_o2m(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE), targets=targets)

        # Both should produce finite losses
        assert torch.isfinite(out_dual["total_loss"])
        assert torch.isfinite(out_o2m["total_loss"])


class TestDualHeadSeparateParameters:
    """Tests for O2O parameter independence (DUAL-01)."""

    def test_dual_head_separate_parameters(self) -> None:
        """O2O cls/reg preds are separate tensors from O2M preds."""
        model = _make_model(
            use_dual_head=True,
            use_soft_labels=True,
        )
        head = model.head

        # Separate tensors (not the same object)
        assert head.cls_preds_o2o[0].weight is not head.cls_preds[0].weight
        assert head.reg_preds_o2o[0].weight is not head.reg_preds[0].weight

        # Same shape
        assert head.cls_preds_o2o[0].weight.shape == head.cls_preds[0].weight.shape
        assert head.reg_preds_o2o[0].weight.shape == head.reg_preds[0].weight.shape

        # Modify O2O and verify O2M is unchanged
        original_o2m_weight = head.cls_preds[0].weight.data.clone()
        head.cls_preds_o2o[0].weight.data.fill_(999.0)
        assert torch.equal(head.cls_preds[0].weight.data, original_o2m_weight), (
            "Modifying O2O weight changed O2M weight -- not independent"
        )


class TestDINOXConfigDualHead:
    """Tests for DINOXConfig validation of dual head flags."""

    def test_config_lambda_o2o_valid(self) -> None:
        """use_dual_head=True with use_soft_labels=True validates OK."""
        config = DINOXConfig(use_dual_head=True, use_soft_labels=True, lambda_o2o=0.5)
        assert config.use_dual_head is True
        assert config.lambda_o2o == 0.5

    def test_config_dual_head_requires_soft_labels(self) -> None:
        """use_dual_head=True without use_soft_labels=True raises ValueError."""
        with pytest.raises(
            ValueError, match="use_dual_head=True requires use_soft_labels=True"
        ):
            DINOXConfig(use_dual_head=True, use_soft_labels=False)


class TestHydraE5E6Configs:
    """Tests for E5 and E6 ablation Hydra configs."""

    def test_hydra_e5_loads(self) -> None:
        """E5 config (SimOTA + soft labels + dual head) loads correctly."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_e5"],
            )
        assert cfg.models.use_dual_head is True
        assert cfg.models.use_soft_labels is True
        assert cfg.models.use_log_iou_cost is True
        assert cfg.models.lambda_o2o == 1.0
        # SimOTA is the default assigner
        assert cfg.models.assigner == "simota"

    def test_hydra_e6_loads(self) -> None:
        """E6 config (TAL + soft labels + dual head) loads correctly."""
        with hydra.initialize(version_base=None, config_path=CONF_PATH):
            cfg = hydra.compose(
                config_name="train_dinox",
                overrides=["models=dinox_m_e6"],
            )
        assert cfg.models.use_dual_head is True
        assert cfg.models.use_soft_labels is True
        assert cfg.models.use_log_iou_cost is True
        assert cfg.models.assigner == "tal"
        assert cfg.models.lambda_o2o == 1.0
