"""Tests for bounding box conversion utilities."""

from __future__ import annotations

import torch

from object_detection_training.utils.boxes import (
    box_iou_1_to_n,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
)


class TestBoxIou1ToN:
    """Tests for box_iou_1_to_n."""

    def test_identical_boxes(self) -> None:
        """IoU of identical boxes should be 1.0."""
        box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        iou = box_iou_1_to_n(box, boxes)
        torch.testing.assert_close(iou, torch.tensor([1.0]))

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes should have IoU of 0."""
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        iou = box_iou_1_to_n(box, boxes)
        torch.testing.assert_close(iou, torch.tensor([0.0]), atol=1e-5, rtol=1e-5)

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes should have 0 < IoU < 1."""
        box = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        iou = box_iou_1_to_n(box, boxes)
        # Intersection: 10x10=100, Union: 400+400-100=700
        expected = torch.tensor([100.0 / 700.0])
        torch.testing.assert_close(iou, expected, atol=1e-5, rtol=1e-5)

    def test_multiple_boxes(self) -> None:
        """Compute IoU against multiple boxes at once."""
        box = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        boxes = torch.tensor(
            [
                [0.0, 0.0, 20.0, 20.0],  # identical -> 1.0
                [50.0, 50.0, 70.0, 70.0],  # no overlap -> 0.0
            ]
        )
        iou = box_iou_1_to_n(box, boxes)
        assert iou.shape == (2,)
        torch.testing.assert_close(iou[0], torch.tensor(1.0))
        torch.testing.assert_close(iou[1], torch.tensor(0.0), atol=1e-5, rtol=1e-5)

    def test_empty_boxes(self) -> None:
        """Empty target boxes should return empty tensor."""
        box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        boxes = torch.zeros(0, 4)
        iou = box_iou_1_to_n(box, boxes)
        assert iou.shape == (0,)

    def test_flat_box_input(self) -> None:
        """Box as shape (4,) should also work."""
        box = torch.tensor([0.0, 0.0, 20.0, 20.0])
        boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        iou = box_iou_1_to_n(box, boxes)
        torch.testing.assert_close(iou, torch.tensor([1.0]))


class TestCxcywhToXyxy:
    """Tests for cxcywh_to_xyxy conversion."""

    def test_single_box(self) -> None:
        """Convert a single box from center format to corners."""
        boxes = torch.tensor([[50.0, 50.0, 20.0, 30.0]])
        result = cxcywh_to_xyxy(boxes)
        expected = torch.tensor([[40.0, 35.0, 60.0, 65.0]])
        torch.testing.assert_close(result, expected)

    def test_multiple_boxes(self) -> None:
        """Convert multiple boxes."""
        boxes = torch.tensor(
            [
                [50.0, 50.0, 20.0, 30.0],
                [100.0, 100.0, 40.0, 60.0],
            ]
        )
        result = cxcywh_to_xyxy(boxes)
        expected = torch.tensor(
            [
                [40.0, 35.0, 60.0, 65.0],
                [80.0, 70.0, 120.0, 130.0],
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_empty_tensor(self) -> None:
        """Empty tensor is returned as-is."""
        boxes = torch.zeros(0, 4)
        result = cxcywh_to_xyxy(boxes)
        assert result.shape == (0, 4)
        assert result.numel() == 0

    def test_batch_dimensions(self) -> None:
        """Supports arbitrary leading batch dimensions."""
        boxes = torch.tensor([[[50.0, 50.0, 20.0, 30.0]]])
        result = cxcywh_to_xyxy(boxes)
        assert result.shape == (1, 1, 4)
        expected = torch.tensor([[[40.0, 35.0, 60.0, 65.0]]])
        torch.testing.assert_close(result, expected)


class TestXyxyToCxcywh:
    """Tests for xyxy_to_cxcywh conversion."""

    def test_single_box(self) -> None:
        """Convert a single box from corners to center format."""
        boxes = torch.tensor([[40.0, 35.0, 60.0, 65.0]])
        result = xyxy_to_cxcywh(boxes)
        expected = torch.tensor([[50.0, 50.0, 20.0, 30.0]])
        torch.testing.assert_close(result, expected)

    def test_empty_tensor(self) -> None:
        """Empty tensor is returned as-is."""
        boxes = torch.zeros(0, 4)
        result = xyxy_to_cxcywh(boxes)
        assert result.shape == (0, 4)

    def test_roundtrip_cxcywh_xyxy(self) -> None:
        """cxcywh -> xyxy -> cxcywh roundtrip is identity."""
        original = torch.tensor(
            [
                [50.0, 60.0, 30.0, 40.0],
                [100.0, 200.0, 50.0, 80.0],
            ]
        )
        roundtripped = xyxy_to_cxcywh(cxcywh_to_xyxy(original))
        torch.testing.assert_close(roundtripped, original)

    def test_roundtrip_xyxy_cxcywh(self) -> None:
        """xyxy -> cxcywh -> xyxy roundtrip is identity."""
        original = torch.tensor(
            [
                [10.0, 20.0, 100.0, 200.0],
                [50.0, 60.0, 150.0, 250.0],
            ]
        )
        roundtripped = cxcywh_to_xyxy(xyxy_to_cxcywh(original))
        torch.testing.assert_close(roundtripped, original)
