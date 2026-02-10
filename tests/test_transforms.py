"""Tests for custom v2 transforms."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from object_detection_training.transforms import (
    MultiScaleRandomResize,
    MultiScaleResize,
    NormalizeBoxCoords,
    RandomSizeCrop,
    ToFloat32Tensor,
    compute_multi_scale_scales,
)


def _make_pil_image(width: int = 640, height: int = 480) -> Image.Image:
    """Create a simple test RGB PIL image."""
    return Image.fromarray(
        torch.randint(0, 256, (height, width, 3), dtype=torch.uint8).numpy()
    )


def _make_boxes(
    n: int = 3, width: int = 640, height: int = 480
) -> tv_tensors.BoundingBoxes:
    """Create test BoundingBoxes in XYXY format."""
    boxes = torch.tensor(
        [
            [10.0, 20.0, 100.0, 150.0],
            [200.0, 100.0, 400.0, 300.0],
            [50.0, 50.0, 200.0, 200.0],
        ][:n],
        dtype=torch.float32,
    )
    return tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width))


# -- MultiScaleResize -------------------------------------------------


class TestMultiScaleResize:
    """Tests for MultiScaleResize transform."""

    def test_scale_list_matches_compute_function(self) -> None:
        """Scale list should match compute_multi_scale_scales()."""
        t = MultiScaleResize(
            base_resolution=560,
            expanded_scales=True,
            patch_size=16,
            num_windows=2,
        )
        expected = compute_multi_scale_scales(
            560, expanded_scales=True, patch_size=16, num_windows=2
        )
        assert t.scales == expected

    def test_skip_random_resize_uses_last_scale(self) -> None:
        """skip_random_resize=True should keep only the largest scale."""
        scales = compute_multi_scale_scales(560, True, 16, 2)
        t = MultiScaleResize(
            base_resolution=560,
            expanded_scales=True,
            patch_size=16,
            num_windows=2,
            skip_random_resize=True,
        )
        assert t.scales == [scales[-1]]

    def test_resizes_pil_image(self) -> None:
        """Should produce a square image at one of the valid scales."""
        t = MultiScaleResize(base_resolution=560, patch_size=16, num_windows=2)
        img = _make_pil_image(640, 480)
        boxes = _make_boxes(2, 640, 480)

        result = t(img, boxes)
        out_img = result[0]

        # PIL in -> PIL out (v2 preserves input type)
        assert isinstance(out_img, Image.Image)
        w, h = out_img.size  # PIL uses (width, height)
        assert h == w, "Output should be square"
        assert h in t.scales, f"Output size {h} not in {t.scales}"

    def test_transforms_bounding_boxes(self) -> None:
        """BoundingBoxes should be resized along with the image."""
        t = MultiScaleResize(
            base_resolution=560,
            expanded_scales=False,
            patch_size=16,
            num_windows=2,
            skip_random_resize=True,  # deterministic single scale
        )
        img = _make_pil_image(640, 480)
        boxes = _make_boxes(2, 640, 480)

        _out_img, out_boxes = t(img, boxes)
        scale = t.scales[0]

        assert isinstance(out_boxes, tv_tensors.BoundingBoxes)
        assert out_boxes.canvas_size == (scale, scale)
        # Boxes should have been scaled from 640x480 to scale x scale
        assert out_boxes.shape == (2, 4)


# -- MultiScaleRandomResize -------------------------------------------


class TestMultiScaleRandomResize:
    """Tests for aspect-ratio preserving multi-scale resize."""

    def test_preserves_aspect_ratio(self) -> None:
        """Output should preserve aspect ratio within max_size."""
        t = MultiScaleRandomResize(
            base_resolution=560,
            skip_random_resize=True,
            max_size=1333,
        )
        img = _make_pil_image(800, 600)
        boxes = _make_boxes(1, 800, 600)

        out_img, _out_boxes = t(img, boxes)
        # PIL in -> PIL out
        w, h = out_img.size
        orig_ratio = 800 / 600
        out_ratio = w / h
        assert abs(orig_ratio - out_ratio) < 0.05


# -- RandomSizeCrop ----------------------------------------------------


class TestRandomSizeCrop:
    """Tests for DETR-style random size crop."""

    def test_crop_within_bounds(self) -> None:
        """Crop size should be within [min_size, max_size]."""
        t = RandomSizeCrop(min_size=100, max_size=300)
        img = _make_pil_image(640, 480)
        boxes = _make_boxes(2, 640, 480)

        out_img, _out_boxes = t(img, boxes)
        # PIL in -> PIL out
        w, h = out_img.size
        assert 100 <= h <= 480
        assert 100 <= w <= 640

    def test_boxes_clipped_to_crop(self) -> None:
        """Boxes should be clipped to the crop region."""
        t = RandomSizeCrop(min_size=100, max_size=200)
        img = _make_pil_image(640, 480)
        boxes = _make_boxes(3, 640, 480)

        out_img, out_boxes = t(img, boxes)
        w, h = out_img.size

        if isinstance(out_boxes, tv_tensors.BoundingBoxes):
            # All box coords should be within crop dimensions
            assert out_boxes[:, 0].min() >= 0
            assert out_boxes[:, 1].min() >= 0
            assert out_boxes[:, 2].max() <= w
            assert out_boxes[:, 3].max() <= h


# -- NormalizeBoxCoords ------------------------------------------------


class TestNormalizeBoxCoords:
    """Tests for NormalizeBoxCoords."""

    def test_normalizes_to_unit_range(self) -> None:
        """Box coords should be in [0, 1] after normalization."""
        t = NormalizeBoxCoords()
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([[0.0, 0.0, 320.0, 240.0]]),
            format="XYXY",
            canvas_size=(480, 640),
        )
        result = t(boxes)
        expected = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        torch.testing.assert_close(result, expected)

    def test_returns_plain_tensor(self) -> None:
        """Output should be a plain Tensor, NOT BoundingBoxes."""
        t = NormalizeBoxCoords()
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
            format="XYXY",
            canvas_size=(400, 400),
        )
        result = t(boxes)
        assert type(result) is torch.Tensor
        assert not isinstance(result, tv_tensors.BoundingBoxes)

    def test_passes_through_non_boxes(self) -> None:
        """Non-BoundingBoxes inputs should pass through unchanged."""
        t = NormalizeBoxCoords()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = t(tensor)
        torch.testing.assert_close(result, tensor)

    def test_works_with_cxcywh_format(self) -> None:
        """Should correctly normalize CXCYWH format boxes."""
        t = NormalizeBoxCoords()
        # Center at (320, 240), width=640, height=480 (full image)
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([[320.0, 240.0, 640.0, 480.0]]),
            format="CXCYWH",
            canvas_size=(480, 640),
        )
        result = t(boxes)
        expected = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        torch.testing.assert_close(result, expected)


# -- ToFloat32Tensor ---------------------------------------------------


class TestToFloat32Tensor:
    """Tests for ToFloat32Tensor."""

    def test_converts_pil_to_float32(self) -> None:
        """PIL image should become a float32 tensor."""
        t = ToFloat32Tensor(scale=False)
        img = _make_pil_image(64, 48)

        result = t(img)
        # When passed a single input, result may be a tuple or single tensor
        out_img = result[0] if isinstance(result, tuple) else result

        assert out_img.dtype == torch.float32
        assert out_img.shape == (3, 48, 64)

    def test_no_scale_keeps_255_range(self) -> None:
        """scale=False should keep pixel values in 0-255 range."""
        t = ToFloat32Tensor(scale=False)
        # Create image with known max pixel value
        img = Image.fromarray(torch.full((32, 32, 3), 200, dtype=torch.uint8).numpy())

        result = t(img)
        out_img = result[0] if isinstance(result, tuple) else result

        assert out_img.max() == 200.0

    def test_scale_normalizes_to_unit_range(self) -> None:
        """scale=True should normalize to [0, 1]."""
        t = ToFloat32Tensor(scale=True)
        img = Image.fromarray(torch.full((32, 32, 3), 255, dtype=torch.uint8).numpy())

        result = t(img)
        out_img = result[0] if isinstance(result, tuple) else result

        torch.testing.assert_close(out_img.max(), torch.tensor(1.0))

    def test_handles_boxes_passthrough(self) -> None:
        """BoundingBoxes should pass through unchanged."""
        t = ToFloat32Tensor(scale=False)
        img = _make_pil_image(64, 48)
        boxes = _make_boxes(2, 64, 48)

        _out_img, out_boxes = t(img, boxes)
        assert isinstance(out_boxes, tv_tensors.BoundingBoxes)
        assert out_boxes.shape == (2, 4)


# -- Integration: v2.Compose pipeline ---------------------------------


class TestComposePipeline:
    """Test that custom transforms work within v2.Compose pipelines."""

    def test_yolox_style_pipeline(self) -> None:
        """YOLOX-style pipeline: flip -> resize -> to_tensor."""
        pipeline = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.0),  # deterministic: no flip
                v2.Resize(size=[320, 320]),
                ToFloat32Tensor(scale=False),
            ]
        )

        img = _make_pil_image(640, 480)
        boxes = _make_boxes(2, 640, 480)

        out_img, _out_boxes = pipeline(img, boxes)
        assert out_img.dtype == torch.float32
        assert out_img.shape[-2:] == (320, 320)

    def test_rfdetr_style_pipeline(self) -> None:
        """RF-DETR-style pipeline with normalization and box coord normalization."""
        pipeline = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.0),
                v2.Resize(size=[560, 560]),
                v2.SanitizeBoundingBoxes(),
                ToFloat32Tensor(scale=False),
                v2.Normalize(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                ),
                v2.ConvertBoundingBoxFormat(format="CXCYWH"),
                NormalizeBoxCoords(),
            ]
        )

        img = _make_pil_image(640, 480)
        boxes = _make_boxes(2, 640, 480)
        labels = torch.tensor([0, 1], dtype=torch.int64)

        # v2.SanitizeBoundingBoxes needs labels in target dict
        target = {
            "boxes": boxes,
            "labels": labels,
        }

        out_img, out_target = pipeline(img, target)
        assert out_img.dtype == torch.float32
        assert out_img.shape[-2:] == (560, 560)

        # Boxes should be normalized to [0, 1] and plain tensor
        out_boxes = out_target["boxes"]
        assert type(out_boxes) is torch.Tensor
        assert not isinstance(out_boxes, tv_tensors.BoundingBoxes)
        assert out_boxes.min() >= 0.0
        assert out_boxes.max() <= 1.0
