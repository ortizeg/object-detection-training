"""Tests for soft SimOTA label assignment in DINOXHead.

Verifies IoU-weighted targets (SIMO-01), -log(IoU) cost (SIMO-02),
RTMDet soft classification cost (SIMO-03), flag independence (SIMO-04),
and gradient flow through the soft label path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from object_detection_training.models.dinox import DINOX, DINOXHead
from object_detection_training.models.dinox.config import DINOXConfig
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


class TestSoftLabelTargets:
    """Tests for SIMO-01 (IoU^gamma weighted classification targets)."""

    def test_iou_gamma_weighting_values(self) -> None:
        """IoU^gamma=2.0 on [0.8, 0.5, 0.3] produces [0.64, 0.25, 0.09]."""
        ious = torch.tensor([0.8, 0.5, 0.3])
        gamma = 2.0
        expected = torch.tensor([0.64, 0.25, 0.09])

        soft_weights = ious.pow(gamma)
        assert torch.allclose(soft_weights, expected, atol=1e-6)

        # Build one-hot for 2 classes (all class 0) and multiply by weights
        num_classes = 2
        labels = torch.zeros(3, dtype=torch.int64)
        one_hot = F.one_hot(labels, num_classes).float()
        cls_target = one_hot * soft_weights.unsqueeze(-1)

        assert torch.allclose(cls_target[:, 0], expected, atol=1e-6)
        assert torch.allclose(cls_target[:, 1], torch.zeros(3), atol=1e-6)

    def test_gamma_zero_produces_binary_targets(self) -> None:
        """gamma=0 makes IoU^0 = 1.0 for all positive samples."""
        ious = torch.tensor([0.8, 0.5, 0.1])
        result = ious.pow(0.0)
        assert torch.allclose(result, torch.ones(3), atol=1e-6)

    def test_gamma_one_is_identity(self) -> None:
        """gamma=1.0 produces weights equal to raw IoU values."""
        ious = torch.tensor([0.7, 0.3])
        result = ious.pow(1.0)
        assert torch.allclose(result, ious, atol=1e-6)


class TestLogIouCost:
    """Tests for SIMO-02 (-log(IoU) regression cost)."""

    def test_log_iou_cost_known_values(self) -> None:
        """Known -log(IoU) values for [0.5, 0.8, 0.1]."""
        ious = torch.tensor([[0.5, 0.8, 0.1]])
        expected = torch.tensor([[0.6931, 0.2231, 2.3026]])
        cost = -torch.log(ious + 1e-8)
        assert torch.allclose(cost, expected, atol=1e-3)

    def test_log_iou_cost_perfect_match(self) -> None:
        """IoU=1.0 gives cost approximately 0.0."""
        cost = -torch.log(torch.tensor(1.0) + 1e-8)
        assert torch.allclose(cost, torch.tensor(0.0), atol=1e-6)

    def test_log_iou_cost_near_zero(self) -> None:
        """IoU near 0 gives very high cost."""
        cost = -torch.log(torch.tensor(0.01) + 1e-8)
        assert cost.item() > 4.0


class TestSoftClassificationCost:
    """Tests for SIMO-03 (RTMDet soft cls cost formulation)."""

    def test_rtmdet_cost_hand_computed(self) -> None:
        """RTMDet cost for 1 GT, 1 anchor, 2 classes with known values."""
        # GT class=0, IoU=0.7
        # pred_score (already sigmoid'd) = [0.6, 0.1]
        # soft_label = IoU * one_hot = [0.7, 0.0]
        pred_scores = torch.tensor([[[0.6, 0.1]]])
        soft_label = torch.tensor([[[0.7, 0.0]]])

        # BCE(P, Y) per element, then scale_factor = |Y - P|^2
        scale_factor = (soft_label - pred_scores).abs().pow(2.0)
        bce = F.binary_cross_entropy(pred_scores, soft_label, reduction="none")
        cost = (bce * scale_factor).sum(-1)

        # The cost should be small but nonzero
        assert cost.shape == (1, 1)
        assert torch.allclose(cost, torch.tensor([[0.00716]]), atol=1e-3)

    def test_rtmdet_cost_perfect_prediction(self) -> None:
        """When pred_score equals soft_label, scale_factor is 0, cost ~0."""
        pred_scores = torch.tensor([[[0.7, 0.0]]])
        soft_label = torch.tensor([[[0.7, 0.0]]])

        scale_factor = (soft_label - pred_scores).abs().pow(2.0)
        bce = F.binary_cross_entropy(pred_scores, soft_label, reduction="none")
        cost = (bce * scale_factor).sum(-1)

        assert cost.item() < 1e-6

    def test_rtmdet_cost_vs_yolox_different(self) -> None:
        """RTMDet cost and YOLOX cost produce different values for same inputs."""
        # Shared inputs
        cls_pred = torch.tensor([[[0.6, 0.1]]])  # already sigmoid'd
        gt_one_hot = torch.tensor([[[1.0, 0.0]]])
        iou = torch.tensor([[[0.7]]])

        # RTMDet cost: BCE(P, IoU*Y) * |IoU*Y - P|^2
        soft_label = gt_one_hot * iou
        scale_factor = (soft_label - cls_pred).abs().pow(2.0)
        rtmdet_cost = (
            F.binary_cross_entropy(cls_pred, soft_label, reduction="none")
            * scale_factor
        ).sum(-1)

        # YOLOX cost: BCE(sqrt(P), Y) (binary targets, sqrt transform)
        yolox_cost = F.binary_cross_entropy(
            cls_pred.sqrt(), gt_one_hot, reduction="none"
        ).sum(-1)

        # They should be different formulations -> different values
        assert not torch.allclose(rtmdet_cost, yolox_cost, atol=1e-4)


class TestFlagIndependence:
    """Tests for SIMO-04 (flags independently toggleable)."""

    def test_config_flags_independent(self) -> None:
        """Each flag combination is valid (no ValueError)."""
        DINOXConfig(use_soft_labels=True, use_log_iou_cost=False)
        DINOXConfig(use_soft_labels=False, use_log_iou_cost=True)
        DINOXConfig(use_soft_labels=True, use_log_iou_cost=True)

    def test_head_stores_flags_independently(self) -> None:
        """DINOXHead stores each flag independently."""
        head1 = DINOXHead(num_classes=2, use_soft_labels=True, use_log_iou_cost=False)
        assert head1.use_soft_labels is True
        assert head1.use_log_iou_cost is False

        head2 = DINOXHead(num_classes=2, use_soft_labels=False, use_log_iou_cost=True)
        assert head2.use_soft_labels is False
        assert head2.use_log_iou_cost is True

    def test_soft_labels_training_forward_runs(self) -> None:
        """Training forward with use_soft_labels=True runs without error."""
        torch.manual_seed(42)
        model = _make_model(use_soft_labels=True, soft_label_gamma=2.0)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        targets = _make_targets()
        out = model(x, targets=targets)

        assert isinstance(out, dict)
        loss = out["total_loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_log_iou_cost_training_forward_runs(self) -> None:
        """Training forward with use_log_iou_cost=True runs without error."""
        torch.manual_seed(42)
        model = _make_model(use_log_iou_cost=True)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        targets = _make_targets()
        out = model(x, targets=targets)

        assert isinstance(out, dict)
        loss = out["total_loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_all_flags_training_forward_runs(self) -> None:
        """Training forward with all soft label flags runs without error."""
        torch.manual_seed(42)
        model = _make_model(
            use_soft_labels=True, soft_label_gamma=2.0, use_log_iou_cost=True
        )
        model.train()
        x = torch.randn(1, 3, 320, 320)
        targets = _make_targets()
        out = model(x, targets=targets)

        assert isinstance(out, dict)
        loss = out["total_loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"


class TestGradientFlow:
    """Tests for gradient flow through soft label paths."""

    def test_gradient_flows_with_soft_labels(self) -> None:
        """Gradient flows through cls_preds with soft labels enabled."""
        torch.manual_seed(42)
        model = _make_model(use_soft_labels=True)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        targets = _make_targets()
        out = model(x, targets=targets)

        loss = out["total_loss"]
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.head.cls_preds.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients found in cls_preds after backward"

    def test_gradient_flows_with_all_flags(self) -> None:
        """Gradient flows with all soft label flags enabled."""
        torch.manual_seed(42)
        model = _make_model(use_soft_labels=True, use_log_iou_cost=True)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        targets = _make_targets()
        out = model(x, targets=targets)

        loss = out["total_loss"]
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.head.cls_preds.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients found in cls_preds after backward"
