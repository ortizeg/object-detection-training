"""Tests for DatasetStatistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from PIL import Image

from object_detection_training.data.dataset_stats import DatasetStatistics
from object_detection_training.data.detection_dataset import DetectionDataset


class FakeDataset(DetectionDataset):
    """Minimal detection dataset for statistics tests."""

    def __init__(self) -> None:
        super().__init__(root_path="/fake", split="train")
        self._raw_images = [
            {"image_id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"image_id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
        ]
        self._raw_anns = [
            {
                "annotation_id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 20,
                "bbox_h": 25,
            },
            {
                "annotation_id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox_x": 50,
                "bbox_y": 50,
                "bbox_w": 40,
                "bbox_h": 40,
            },
            {
                "annotation_id": 3,
                "image_id": 2,
                "category_id": 0,
                "bbox_x": 5,
                "bbox_y": 5,
                "bbox_w": 80,
                "bbox_h": 90,
            },
        ]
        self._raw_cats = {0: "person", 1: "ball"}

    def load_annotations(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
        return (
            pd.DataFrame(self._raw_images),
            pd.DataFrame(self._raw_anns),
            self._raw_cats,
        )

    def _load_image(self, image_id: int) -> Image.Image:
        return Image.new("RGB", (100, 100))


class TestDatasetStatisticsSummary:
    """Tests for summary statistics."""

    def test_summary_keys(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        summary = stats.summary()

        assert "num_images" in summary
        assert "num_annotations" in summary
        assert "num_classes" in summary
        assert "annotations_per_image" in summary
        assert "size_distribution" in summary
        assert "area_stats" in summary

    def test_summary_values(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        summary = stats.summary()

        assert summary["num_images"] == 2
        assert summary["num_annotations"] == 3
        assert summary["num_classes"] == 2

    def test_summary_cached(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        s1 = stats.summary()
        s2 = stats.summary()
        assert s1 is s2  # same object


class TestDatasetStatisticsClassDistribution:
    """Tests for class distribution."""

    def test_class_distribution_dataframe(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        dist = stats.class_distribution()

        assert isinstance(dist, pd.DataFrame)
        assert "category_name" in dist.columns
        assert "count" in dist.columns
        assert "percentage" in dist.columns

    def test_class_distribution_counts(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        dist = stats.class_distribution()

        person_count = dist[dist["category_name"] == "person"]["count"].values[0]
        ball_count = dist[dist["category_name"] == "ball"]["count"].values[0]
        assert person_count == 2
        assert ball_count == 1


class TestClassDistributionForImages:
    """Tests for class_distribution_for_images() with subset/duplicate image IDs."""

    def test_single_image(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        # Image 1 has: person (ann 1), ball (ann 2)
        dist = stats.class_distribution_for_images([1])

        assert isinstance(dist, pd.DataFrame)
        assert set(dist.columns) == {"category_name", "count", "percentage"}
        person_row = dist[dist["category_name"] == "person"]
        ball_row = dist[dist["category_name"] == "ball"]
        assert person_row["count"].values[0] == 1
        assert ball_row["count"].values[0] == 1

    def test_duplicate_image_ids(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        # Image 1 sampled 3 times: 3 * (1 person + 1 ball) = 3 person, 3 ball
        dist = stats.class_distribution_for_images([1, 1, 1])

        person_count = dist[dist["category_name"] == "person"]["count"].values[0]
        ball_count = dist[dist["category_name"] == "ball"]["count"].values[0]
        assert person_count == 3
        assert ball_count == 3

    def test_mixed_images(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        # Image 1: person, ball. Image 2: person.
        # Sample: [1, 2, 2] -> person: 1+2=3, ball: 1+0=1
        dist = stats.class_distribution_for_images([1, 2, 2])

        person_count = dist[dist["category_name"] == "person"]["count"].values[0]
        ball_count = dist[dist["category_name"] == "ball"]["count"].values[0]
        assert person_count == 3
        assert ball_count == 1

    def test_percentages_sum_to_100(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        dist = stats.class_distribution_for_images([1, 2])

        total_pct = dist["percentage"].sum()
        assert abs(total_pct - 100.0) < 0.1

    def test_empty_image_ids(self) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        dist = stats.class_distribution_for_images([])
        assert len(dist) == 0


class TestDatasetStatisticsPlots:
    """Tests for plot generation."""

    def test_box_size_histogram(self, tmp_path: Path) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        save_path = tmp_path / "box_hist.png"
        fig = stats.box_size_histogram(save_path=save_path)

        assert save_path.exists()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_class_distribution_chart(self, tmp_path: Path) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        save_path = tmp_path / "class_dist.png"
        fig = stats.class_distribution_chart(save_path=save_path)

        assert save_path.exists()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestDatasetStatisticsExport:
    """Tests for export functionality."""

    def test_export_summary_json(self, tmp_path: Path) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        save_path = tmp_path / "summary.json"
        stats.export_summary_json(save_path)

        assert save_path.exists()

        import json

        with open(save_path) as f:
            data = json.load(f)
        assert data["num_images"] == 2

    def test_generate_report(self, tmp_path: Path) -> None:
        ds = FakeDataset()
        stats = DatasetStatistics(ds)
        stats.generate_report(tmp_path)

        assert (tmp_path / "train_stats.json").exists()
        assert (tmp_path / "train_box_sizes.png").exists()
        assert (tmp_path / "train_class_dist.png").exists()
