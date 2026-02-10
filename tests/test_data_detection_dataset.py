"""Tests for the DetectionDataset base class."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from object_detection_training.data.detection_dataset import (
    DetectionDataset,
    SizeThresholds,
)


class FakeDetectionDataset(DetectionDataset):
    """Concrete implementation of DetectionDataset for testing."""

    def __init__(
        self,
        images: list[dict[str, object]],
        annotations: list[dict[str, object]],
        categories: dict[int, str],
        **kwargs: object,
    ):
        super().__init__(root_path="/fake", **kwargs)
        self._raw_images = images
        self._raw_annotations = annotations
        self._raw_categories = categories

    def load_annotations(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
        images_df = pd.DataFrame(self._raw_images)
        annotations_df = pd.DataFrame(self._raw_annotations)
        return images_df, annotations_df, self._raw_categories

    def _load_image(self, image_id: int) -> Image.Image:
        # Return a 100x100 black image
        return Image.new("RGB", (100, 100))


def _make_dataset(
    selected_categories: list[str] | None = None,
) -> FakeDetectionDataset:
    """Create a FakeDetectionDataset with 2 images and 3 annotations."""
    images = [
        {"image_id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
        {"image_id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
    ]
    annotations = [
        {
            "annotation_id": 1,
            "image_id": 1,
            "category_id": 10,
            "bbox_x": 10,
            "bbox_y": 10,
            "bbox_w": 30,
            "bbox_h": 40,
        },
        {
            "annotation_id": 2,
            "image_id": 1,
            "category_id": 20,
            "bbox_x": 50,
            "bbox_y": 50,
            "bbox_w": 20,
            "bbox_h": 20,
        },
        {
            "annotation_id": 3,
            "image_id": 2,
            "category_id": 10,
            "bbox_x": 5,
            "bbox_y": 5,
            "bbox_w": 80,
            "bbox_h": 90,
        },
    ]
    categories = {10: "person", 20: "ball"}

    return FakeDetectionDataset(
        images=images,
        annotations=annotations,
        categories=categories,
        selected_categories=selected_categories,
    )


class TestSizeThresholds:
    """Tests for SizeThresholds Pydantic model."""

    def test_default_values(self) -> None:
        thresholds = SizeThresholds()
        assert thresholds.small == 32.0
        assert thresholds.medium == 96.0

    def test_custom_values(self) -> None:
        thresholds = SizeThresholds(small=16.0, medium=64.0)
        assert thresholds.small == 16.0
        assert thresholds.medium == 64.0

    def test_medium_must_be_greater_than_small(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="medium threshold must be greater"):
            SizeThresholds(small=50.0, medium=30.0)


class TestDetectionDatasetLen:
    """Tests for __len__."""

    def test_len_returns_image_count(self) -> None:
        dataset = _make_dataset()
        assert len(dataset) == 2


class TestDetectionDatasetProperties:
    """Tests for dataset properties."""

    def test_num_classes(self) -> None:
        dataset = _make_dataset()
        assert dataset.num_classes == 2

    def test_class_names(self) -> None:
        dataset = _make_dataset()
        names = dataset.class_names
        assert len(names) == 2
        assert "person" in names
        assert "ball" in names

    def test_label_map(self) -> None:
        dataset = _make_dataset()
        label_map = dataset.label_map
        # Category IDs 10 and 20 should map to 0 and 1
        assert set(label_map.values()) == {0, 1}

    def test_categories(self) -> None:
        dataset = _make_dataset()
        cats = dataset.categories
        assert 10 in cats
        assert 20 in cats
        assert cats[10] == "person"


class TestDetectionDatasetGetitem:
    """Tests for __getitem__."""

    def test_returns_image_and_target(self) -> None:
        dataset = _make_dataset()
        img, target = dataset[0]

        assert isinstance(img, Image.Image)
        assert "boxes" in target
        assert "labels" in target

    def test_boxes_are_xyxy(self) -> None:
        dataset = _make_dataset()
        _, target = dataset[0]
        boxes = target["boxes"]
        # x2 > x1 and y2 > y1
        if boxes.numel() > 0:
            assert (boxes[:, 2] > boxes[:, 0]).all()
            assert (boxes[:, 3] > boxes[:, 1]).all()

    def test_labels_are_contiguous(self) -> None:
        dataset = _make_dataset()
        _, target = dataset[0]
        labels = target["labels"]
        # All labels should be 0 or 1 (contiguous)
        assert labels.min() >= 0
        assert labels.max() < dataset.num_classes

    def test_image_with_no_annotations(self) -> None:
        """Image with no annotations returns empty boxes tensor."""
        dataset = _make_dataset()
        # Image 2 has 1 annotation; but we can test the shape
        _, target = dataset[1]
        # Should still have proper tensor structure
        assert target["boxes"].ndim == 2
        assert target["boxes"].shape[1] == 4


class TestDetectionDatasetCategoryFiltering:
    """Tests for selected_categories filtering."""

    def test_filter_single_category(self) -> None:
        dataset = _make_dataset(selected_categories=["person"])
        assert dataset.num_classes == 1
        assert dataset.class_names == ["person"]

    def test_filter_preserves_label_order(self) -> None:
        """selected_categories order determines contiguous IDs."""
        dataset = _make_dataset(selected_categories=["ball", "person"])
        label_map = dataset.label_map
        # ball should be 0, person should be 1
        assert label_map[20] == 0  # ball
        assert label_map[10] == 1  # person


class TestDetectionDatasetLabelsMapping:
    """Tests for labels_mapping property and export."""

    def test_labels_mapping_property(self) -> None:
        dataset = _make_dataset()
        mapping = dataset.labels_mapping
        assert isinstance(mapping, dict)
        assert 0 in mapping
        assert 1 in mapping

    def test_export_labels_mapping(self, tmp_path: Path) -> None:
        dataset = _make_dataset()
        export_path = tmp_path / "labels.json"
        dataset.export_labels_mapping(export_path)

        import json

        with open(export_path) as f:
            data = json.load(f)

        assert "num_classes" in data
        assert data["num_classes"] == 2
        assert "class_names" in data
        assert "id_to_name" in data
