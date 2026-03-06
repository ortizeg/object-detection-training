"""Tests for DFLModule and distribution_focal_loss.

Verifies DFL output shapes, integral decode correctness, gradient flow,
buffer registration, and distribution_focal_loss known-value accuracy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from object_detection_training.models.dinox import DFLModule, distribution_focal_loss


class TestDFLModule:
    """Unit tests for DFLModule integral decoding."""

    def test_output_shape(self) -> None:
        """DFLModule(16) maps [10, 68] -> [10, 4]."""
        dfl = DFLModule(16)
        out = dfl(torch.randn(10, 68))
        assert out.shape == (10, 4)

    def test_output_shape_batched(self) -> None:
        """DFLModule(16) maps [2, 10, 68] -> [2, 10, 4]."""
        dfl = DFLModule(16)
        out = dfl(torch.randn(2, 10, 68))
        assert out.shape == (2, 10, 4)

    def test_integral_decode_known_input(self) -> None:
        """One-hot distribution at bin 8 should produce ~8.0 for all 4 edges."""
        dfl = DFLModule(16)
        # Create input: very large logit at position 8, small elsewhere
        # For each of the 4 edges, we need 17 bins -> 68 total
        x = torch.full((1, 68), -100.0)
        for edge in range(4):
            x[0, edge * 17 + 8] = 100.0  # bin 8 for each edge

        out = dfl(x)
        assert out.shape == (1, 4)
        torch.testing.assert_close(out, torch.full((1, 4), 8.0), atol=0.01, rtol=0.0)

    def test_uniform_distribution_produces_midpoint(self) -> None:
        """Uniform logits (all zeros) -> midpoint = reg_max/2 = 8.0."""
        dfl = DFLModule(16)
        x = torch.zeros(1, 68)
        out = dfl(x)
        # Uniform softmax over [0..16]: mean = 16/2 = 8.0
        torch.testing.assert_close(out, torch.full((1, 4), 8.0), atol=0.01, rtol=0.0)

    def test_project_buffer_is_registered(self) -> None:
        """'project' must be in named_buffers (critical for ONNX export)."""
        dfl = DFLModule(16)
        buffer_names = [name for name, _ in dfl.named_buffers()]
        assert "project" in buffer_names

    def test_gradient_flows(self) -> None:
        """Gradients flow through DFLModule."""
        dfl = DFLModule(16)
        x = torch.randn(5, 68, requires_grad=True)
        out = dfl(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_different_reg_max(self) -> None:
        """DFLModule(8) maps [5, 36] -> [5, 4]."""
        dfl = DFLModule(8)
        out = dfl(torch.randn(5, 36))
        assert out.shape == (5, 4)


class TestDistributionFocalLoss:
    """Unit tests for distribution_focal_loss function."""

    def test_output_shape(self) -> None:
        """distribution_focal_loss returns shape [N]."""
        pred = torch.randn(10, 17)
        target = torch.rand(10) * 15.99
        loss = distribution_focal_loss(pred, target)
        assert loss.shape == (10,)

    def test_gradient_flows(self) -> None:
        """Gradients flow through distribution_focal_loss."""
        pred = torch.randn(10, 17, requires_grad=True)
        target = torch.rand(10) * 15.99
        loss = distribution_focal_loss(pred, target)
        loss.sum().backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_integer_target(self) -> None:
        """For target=5.0, loss equals standard CE at bin 5."""
        pred = torch.randn(1, 17)
        target = torch.tensor([5.0])

        dfl_loss = distribution_focal_loss(pred, target)
        ce_loss = F.cross_entropy(pred, torch.tensor([5]), reduction="none")

        # For integer target: weight_left=1.0, weight_right=0.0
        # So DFL loss = 1.0 * CE(bin=5) + 0.0 * CE(bin=6) = CE(bin=5)
        torch.testing.assert_close(dfl_loss, ce_loss, atol=1e-5, rtol=1e-5)

    def test_midpoint_target(self) -> None:
        """For target=5.5, loss is average of CE at bins 5 and 6."""
        pred = torch.randn(1, 17)
        target = torch.tensor([5.5])

        dfl_loss = distribution_focal_loss(pred, target)
        ce_5 = F.cross_entropy(pred, torch.tensor([5]), reduction="none")
        ce_6 = F.cross_entropy(pred, torch.tensor([6]), reduction="none")
        expected = 0.5 * ce_5 + 0.5 * ce_6

        torch.testing.assert_close(dfl_loss, expected, atol=1e-5, rtol=1e-5)

    def test_boundary_target_zero(self) -> None:
        """For target=0.0, all weight on bin 0."""
        pred = torch.randn(1, 17)
        target = torch.tensor([0.0])

        dfl_loss = distribution_focal_loss(pred, target)
        ce_0 = F.cross_entropy(pred, torch.tensor([0]), reduction="none")

        torch.testing.assert_close(dfl_loss, ce_0, atol=1e-5, rtol=1e-5)

    def test_boundary_target_near_max(self) -> None:
        """For target near reg_max-1, weight concentrates on last bins."""
        pred = torch.randn(1, 17)
        # target = 15.0 exactly -> all weight on bin 15
        target = torch.tensor([15.0])

        dfl_loss = distribution_focal_loss(pred, target)
        ce_15 = F.cross_entropy(pred, torch.tensor([15]), reduction="none")

        torch.testing.assert_close(dfl_loss, ce_15, atol=1e-5, rtol=1e-5)
