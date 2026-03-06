"""Tests for DINOXHead architecture, forward pass, and loss computation.

Verifies architecture equivalence with YOLOXHead (non-DFL mode),
output shapes, loss dict contract, and gradient flow.
"""

from __future__ import annotations

import torch

from object_detection_training.models.dinox import DINOX, DINOXHead
from object_detection_training.models.yolox import YOLOPAFPN, YOLOXHead

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


class TestDINOXHeadArchitecture:
    """Verify DINOXHead architecture properties."""

    def test_non_dfl_matches_yolox_keys(self) -> None:
        """DINOXHead(use_dfl=False) state_dict keys match YOLOXHead."""
        dinox_head = DINOXHead(num_classes=NUM_CLASSES, width=0.75, use_dfl=False)
        yolox_head = YOLOXHead(num_classes=NUM_CLASSES, width=0.75)

        dinox_keys = set(dinox_head.state_dict().keys())
        yolox_keys = set(yolox_head.state_dict().keys())

        assert dinox_keys == yolox_keys, (
            f"Key mismatch: "
            f"dinox_only={dinox_keys - yolox_keys}, "
            f"yolox_only={yolox_keys - dinox_keys}"
        )

        # Also verify shapes match
        dinox_sd = dinox_head.state_dict()
        yolox_sd = yolox_head.state_dict()
        for key in dinox_keys:
            assert dinox_sd[key].shape == yolox_sd[key].shape, (
                f"Shape mismatch at {key}: "
                f"dinox={dinox_sd[key].shape} vs yolox={yolox_sd[key].shape}"
            )

    def test_dfl_reg_preds_shape(self) -> None:
        """DFL head reg_preds output channels = 4*(reg_max+1) = 68."""
        head = DINOXHead(num_classes=NUM_CLASSES, width=0.75, use_dfl=True, reg_max=16)
        # Each reg_pred Conv2d should have 68 output channels
        for reg_pred in head.reg_preds:
            assert reg_pred.out_channels == 68

    def test_non_dfl_reg_preds_shape(self) -> None:
        """Non-DFL head reg_preds output channels = 4."""
        head = DINOXHead(num_classes=NUM_CLASSES, width=0.75, use_dfl=False)
        for reg_pred in head.reg_preds:
            assert reg_pred.out_channels == 4

    def test_has_dfl_module_when_enabled(self) -> None:
        """DINOXHead(use_dfl=True) has self.dfl attribute."""
        head = DINOXHead(num_classes=NUM_CLASSES, use_dfl=True)
        assert hasattr(head, "dfl")

    def test_no_dfl_module_when_disabled(self) -> None:
        """DINOXHead(use_dfl=False) does NOT have self.dfl attribute."""
        head = DINOXHead(num_classes=NUM_CLASSES, use_dfl=False)
        assert not hasattr(head, "dfl")


class TestDINOXHeadForward:
    """Verify DINOXHead forward pass output shapes."""

    def test_inference_output_shape_no_dfl(self) -> None:
        """Non-DFL inference: output [1, N, 5+C]."""
        model = _make_model(use_dfl=False)
        model.eval()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 5 + NUM_CLASSES

    def test_inference_output_shape_dfl(self) -> None:
        """DFL inference: output [1, N, 5+C] (integral baked in)."""
        model = _make_model(use_dfl=True)
        model.eval()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 5 + NUM_CLASSES

    def test_inference_anchor_count_matches(self) -> None:
        """DFL and non-DFL produce same anchor count N."""
        model_no_dfl = _make_model(use_dfl=False)
        model_dfl = _make_model(use_dfl=True)
        model_no_dfl.eval()
        model_dfl.eval()

        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out_no_dfl = model_no_dfl(x)
            out_dfl = model_dfl(x)

        assert out_no_dfl.shape[1] == out_dfl.shape[1]

    def test_non_dfl_output_range(self) -> None:
        """In eval mode, obj_conf and cls values are in reasonable range."""
        model = _make_model(use_dfl=False)
        model.eval()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)

        # obj_conf (column 4) should be sigmoid -> [0, 1]
        obj_conf = out[0, :, 4]
        assert obj_conf.min() >= 0.0
        assert obj_conf.max() <= 1.0

        # cls values (columns 5+) should be sigmoid -> [0, 1]
        cls_vals = out[0, :, 5:]
        assert cls_vals.min() >= 0.0
        assert cls_vals.max() <= 1.0


class TestDINOXHeadLoss:
    """Verify DINOXHead training loss computation."""

    def test_training_returns_loss_dict(self) -> None:
        """Training mode returns dict with expected loss keys."""
        model = _make_model(use_dfl=False)
        model.train()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        targets = _make_targets()
        out = model(x, targets=targets)

        assert isinstance(out, dict)
        expected_keys = {
            "total_loss",
            "iou_loss",
            "conf_loss",
            "cls_loss",
            "l1_loss",
            "num_fg",
        }
        assert expected_keys.issubset(set(out.keys())), (
            f"Missing keys: {expected_keys - set(out.keys())}"
        )

    def test_training_dfl_returns_loss_dict(self) -> None:
        """DFL training mode returns same loss dict structure."""
        model = _make_model(use_dfl=True)
        model.train()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        targets = _make_targets()
        out = model(x, targets=targets)

        assert isinstance(out, dict)
        expected_keys = {
            "total_loss",
            "iou_loss",
            "conf_loss",
            "cls_loss",
            "l1_loss",
            "num_fg",
        }
        assert expected_keys.issubset(set(out.keys()))

    def test_loss_is_scalar(self) -> None:
        """All loss values should be scalar tensors."""
        model = _make_model(use_dfl=False)
        model.train()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        targets = _make_targets()
        out = model(x, targets=targets)

        for key in ["total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss"]:
            v = out[key]
            assert isinstance(v, torch.Tensor), f"{key} is not a tensor"
            assert v.ndim <= 1, f"{key} has ndim={v.ndim}, expected scalar"

    def test_loss_gradient_flows(self) -> None:
        """total_loss.backward() succeeds without error."""
        model = _make_model(use_dfl=False)
        model.train()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        targets = _make_targets()
        out = model(x, targets=targets)

        loss = out["total_loss"]
        loss.backward()
        # Verify some gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients found after backward"
