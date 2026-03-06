"""Tests for Matchability-Aware Loss (MAL) weighting functions (TEST-03).

Verifies MAL boundary conditions (weight=1.0 at matchability=1.0),
gradient amplification (weight>1.0 at low matchability), monotonicity,
and gradient flow through the matchability computation.
"""

from __future__ import annotations

import torch

from object_detection_training.models.dinox import DINOX, DINOXHead
from object_detection_training.models.dinox.mal import mal_weight, matchability_score
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


class TestMatchabilityScore:
    """Tests for matchability_score function."""

    def test_matchability_perfect_match(self) -> None:
        """When ious=1.0 and cls_scores=1.0, matchability=1.0."""
        ious = torch.tensor([1.0, 1.0])
        cls_scores = torch.tensor([1.0, 1.0])
        result = matchability_score(ious, cls_scores, gamma=1.5)
        expected = torch.ones(2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_matchability_zero_iou(self) -> None:
        """When ious=0.0, matchability=0.0 regardless of cls_scores."""
        ious = torch.tensor([0.0, 0.0])
        cls_scores = torch.tensor([0.5, 0.9])
        result = matchability_score(ious, cls_scores, gamma=1.5)
        expected = torch.zeros(2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_matchability_formula(self) -> None:
        """Hand-computed: ious=0.5, cls=0.8, gamma=1.5 -> ~0.3953."""
        ious = torch.tensor([0.5])
        cls_scores = torch.tensor([0.8])
        result = matchability_score(ious, cls_scores, gamma=1.5)
        # 0.5^1.5 * 0.8^(-0.5) = 0.35355 * 1.11803 = 0.39528
        expected = torch.tensor([0.5**1.5 * 0.8 ** (1.0 - 1.5)])
        assert torch.allclose(result, expected, atol=1e-4)

    def test_matchability_batch(self) -> None:
        """Verify vectorized computation on multi-element tensors."""
        ious = torch.tensor([0.3, 0.5, 0.7, 0.9])
        cls_scores = torch.tensor([0.6, 0.8, 0.4, 0.95])
        gamma = 1.5
        result = matchability_score(ious, cls_scores, gamma=gamma)
        expected = ious.pow(gamma) * cls_scores.pow(1.0 - gamma)
        assert result.shape == (4,)
        assert torch.allclose(result, expected, atol=1e-6)


class TestMALWeight:
    """Tests for mal_weight function."""

    def test_mal_weight_at_one(self) -> None:
        """matchability=1.0 -> weight=1.0 (BCE equivalence, TEST-03 boundary)."""
        m = torch.tensor([1.0])
        w = mal_weight(m)
        assert torch.allclose(w, torch.tensor([1.0]), atol=1e-6)

    def test_mal_weight_at_zero(self) -> None:
        """matchability=0.0 -> weight=2.0 (maximum amplification)."""
        m = torch.tensor([0.0])
        w = mal_weight(m)
        assert torch.allclose(w, torch.tensor([2.0]), atol=1e-6)

    def test_mal_weight_monotonic(self) -> None:
        """Weights are monotonically decreasing as matchability increases."""
        matchabilities = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = mal_weight(matchabilities)
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], (
                f"Not monotonically decreasing at index {i}: "
                f"{weights[i].item()} <= {weights[i + 1].item()}"
            )

    def test_mal_weight_range(self) -> None:
        """All weights in [1.0, 2.0] for matchability in [0, 1]."""
        matchabilities = torch.linspace(0.0, 1.0, steps=100)
        weights = mal_weight(matchabilities)
        assert (weights >= 1.0 - 1e-6).all(), f"Weight below 1.0: {weights.min()}"
        assert (weights <= 2.0 + 1e-6).all(), f"Weight above 2.0: {weights.max()}"


class TestMALGradient:
    """Tests for gradient flow through MAL computation."""

    def test_mal_gradient_flows_through_cls_scores(self) -> None:
        """Gradient flows through cls_scores in matchability computation."""
        cls_scores = torch.tensor([0.5, 0.8], requires_grad=True)
        ious = torch.tensor([0.6, 0.7])  # no grad
        m = matchability_score(ious, cls_scores, gamma=1.5)
        w = mal_weight(m)
        w.sum().backward()
        assert cls_scores.grad is not None
        assert cls_scores.grad.abs().sum() > 0

    def test_mal_gradient_transparent_for_ious(self) -> None:
        """matchability_score does not detach ious -- gradient flows if requested.

        The real detachment happens at the caller (dinox_head._get_losses).
        This test verifies the function itself is gradient-transparent.
        """
        ious = torch.tensor([0.6, 0.7], requires_grad=True)
        cls_scores = torch.tensor([0.5, 0.8], requires_grad=True)
        m = matchability_score(ious, cls_scores, gamma=1.5)
        m.sum().backward()
        assert ious.grad is not None, (
            "ious gradient should flow (function is transparent)"
        )

    def test_mal_integration_forward_backward(self) -> None:
        """Full forward+backward with MAL enabled succeeds without errors."""
        torch.manual_seed(42)
        model = _make_model(use_soft_labels=True, use_mal=True, mal_gamma=1.5)
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
        assert has_grad, "No gradients found in cls_preds after backward with MAL"
