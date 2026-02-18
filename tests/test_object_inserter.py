"""Tests for ObjectInserter transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from object_detection_training.schemas.crop_manifest import (
    CropMetadata,
    RLEMask,
    encode_rle,
)
from object_detection_training.transforms.object_inserter import ObjectInserter


def _make_crop_dir(
    tmp_path: Path,
    category: str = "ball",
    n_crops: int = 3,
    crop_size: int = 30,
) -> Path:
    """Create a fake crop directory with PNG + JSON pairs."""
    cat_dir = tmp_path / "crops" / category
    cat_dir.mkdir(parents=True)

    for i in range(n_crops):
        # Create a solid-color crop
        crop_img = Image.fromarray(
            np.full((crop_size, crop_size, 3), 200, dtype=np.uint8)
        )
        mask = np.ones((crop_size, crop_size), dtype=bool)
        rle = encode_rle(mask)

        metadata = CropMetadata(
            filename=f"{i}.png",
            mask=RLEMask(counts=rle.counts, height=rle.height, width=rle.width),
            bbox_x=10.0,
            bbox_y=10.0,
            bbox_w=float(crop_size),
            bbox_h=float(crop_size),
            source_image="test.jpg",
            annotation_id=i,
            category_name=category,
        )

        crop_img.save(cat_dir / f"{i}.png")
        metadata.save_json(cat_dir / f"{i}.json")

    return tmp_path / "crops"


def _make_image_and_target(
    width: int = 200, height: int = 200, n_boxes: int = 1
) -> tuple[Image.Image, dict]:
    """Create a test PIL image and target dict."""
    img = Image.fromarray(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))

    boxes_data = [[10.0, 10.0, 50.0, 50.0]][:n_boxes]
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor(boxes_data, dtype=torch.float32),
        format="XYXY",
        canvas_size=(height, width),
    )

    target = {
        "boxes": boxes,
        "labels": torch.tensor([0] * n_boxes, dtype=torch.int64),
        "area": torch.tensor([1600.0] * n_boxes, dtype=torch.float32),
        "iscrowd": torch.zeros(n_boxes, dtype=torch.int64),
    }
    return img, target


class TestObjectInserter:
    """Tests for ObjectInserter transform."""

    def test_no_insertion_when_p_zero(self, tmp_path: Path) -> None:
        """p=0 should never insert objects."""
        crops_dir = _make_crop_dir(tmp_path)
        inserter = ObjectInserter(
            crops_dir=str(crops_dir),
            category_to_label={"ball": 1},
            p=0.0,
        )
        img, target = _make_image_and_target()
        _out_img, out_target = inserter(img, target)
        assert len(out_target["labels"]) == 1

    def test_inserts_when_p_one(self, tmp_path: Path) -> None:
        """p=1 should always insert at least one object."""
        crops_dir = _make_crop_dir(tmp_path, crop_size=20)
        inserter = ObjectInserter(
            crops_dir=str(crops_dir),
            category_to_label={"ball": 1},
            p=1.0,
            max_objects_per_image=1,
        )
        img, target = _make_image_and_target(width=300, height=300, n_boxes=0)
        _out_img, out_target = inserter(img, target)
        # Should have inserted at least one object (no existing boxes to block)
        assert len(out_target["labels"]) >= 1

    def test_boxes_and_labels_appended(self, tmp_path: Path) -> None:
        """New boxes and labels should be appended to target."""
        crops_dir = _make_crop_dir(tmp_path, crop_size=15)
        inserter = ObjectInserter(
            crops_dir=str(crops_dir),
            category_to_label={"ball": 2},
            p=1.0,
            max_objects_per_image=1,
            iou_threshold=1.0,  # Allow overlap for determinism
        )
        img, target = _make_image_and_target(width=300, height=300)
        original_n = len(target["labels"])
        _out_img, out_target = inserter(img, target)

        assert len(out_target["labels"]) > original_n
        assert len(out_target["boxes"]) == len(out_target["labels"])
        assert len(out_target["area"]) == len(out_target["labels"])
        assert len(out_target["iscrowd"]) == len(out_target["labels"])
        # New label should be 2
        assert out_target["labels"][-1].item() == 2

    def test_iou_threshold_prevents_occlusion(self, tmp_path: Path) -> None:
        """With very tight box coverage, insertion should be skipped."""
        crops_dir = _make_crop_dir(tmp_path, crop_size=180)
        inserter = ObjectInserter(
            crops_dir=str(crops_dir),
            category_to_label={"ball": 1},
            p=1.0,
            max_objects_per_image=1,
            iou_threshold=0.01,
            max_placement_retries=5,
        )
        # Image almost entirely covered by existing box
        img, target = _make_image_and_target(width=200, height=200)
        target["boxes"] = tv_tensors.BoundingBoxes(
            torch.tensor([[0.0, 0.0, 195.0, 195.0]]),
            format="XYXY",
            canvas_size=(200, 200),
        )
        _out_img, out_target = inserter(img, target)
        # Should not have inserted (no valid position)
        assert len(out_target["labels"]) == 1

    def test_crop_larger_than_image_scaled_down(self, tmp_path: Path) -> None:
        """Crops larger than max_crop_ratio get scaled down."""
        crops_dir = _make_crop_dir(tmp_path, crop_size=100)
        inserter = ObjectInserter(
            crops_dir=str(crops_dir),
            category_to_label={"ball": 1},
            p=1.0,
            max_objects_per_image=1,
            max_crop_ratio=0.3,
            iou_threshold=1.0,
        )
        img, target = _make_image_and_target(width=400, height=400, n_boxes=0)
        _out_img, out_target = inserter(img, target)
        n_labels = len(out_target["labels"])
        if n_labels > 0:
            # Inserted box should be within max_crop_ratio of image
            boxes = out_target["boxes"]
            last_box = boxes[n_labels - 1]
            box_w = last_box[2] - last_box[0]
            box_h = last_box[3] - last_box[1]
            assert box_w <= 400 * 0.3 + 5  # small tolerance for rounding
            assert box_h <= 400 * 0.3 + 5

    def test_passthrough_non_pil(self, tmp_path: Path) -> None:
        """Non-PIL input should pass through unchanged."""
        crops_dir = _make_crop_dir(tmp_path)
        inserter = ObjectInserter(
            crops_dir=str(crops_dir),
            category_to_label={"ball": 1},
            p=1.0,
        )
        tensor = torch.randn(3, 64, 64)
        target = {"boxes": torch.zeros(0, 4)}
        out_tensor, _out_target = inserter(tensor, target)
        torch.testing.assert_close(out_tensor, tensor)

    def test_works_in_compose_pipeline(self, tmp_path: Path) -> None:
        """ObjectInserter should work inside v2.Compose."""
        crops_dir = _make_crop_dir(tmp_path, crop_size=20)
        pipeline = v2.Compose(
            [
                ObjectInserter(
                    crops_dir=str(crops_dir),
                    category_to_label={"ball": 1},
                    p=1.0,
                    max_objects_per_image=1,
                    iou_threshold=1.0,
                ),
                v2.RandomHorizontalFlip(p=0.0),
            ]
        )
        img, target = _make_image_and_target(width=200, height=200, n_boxes=0)
        out_img, _out_target = pipeline(img, target)
        assert isinstance(out_img, Image.Image)

    def test_empty_crops_dir(self, tmp_path: Path) -> None:
        """Empty crops directory should passthrough without error."""
        empty_dir = tmp_path / "empty_crops"
        empty_dir.mkdir()
        inserter = ObjectInserter(
            crops_dir=str(empty_dir),
            category_to_label={"ball": 1},
            p=1.0,
        )
        img, target = _make_image_and_target()
        _out_img, out_target = inserter(img, target)
        assert len(out_target["labels"]) == 1
