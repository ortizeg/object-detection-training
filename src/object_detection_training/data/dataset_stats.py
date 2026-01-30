"""
Dataset statistics and visualization module.

Computes and visualizes statistics for detection datasets:
- Box size distribution histograms
- Class distribution charts
- Per-image annotation counts
- Summary statistics

Designed to be called from callbacks or standalone analysis scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from object_detection_training.data.detection_dataset import DetectionDataset


class DatasetStatistics:
    """Compute and visualize detection dataset statistics."""

    def __init__(self, dataset: DetectionDataset):
        """Initialize with a detection dataset.

        Args:
            dataset: A DetectionDataset instance (already loaded).
        """
        self.dataset = dataset
        self._summary_cache: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics.

        Returns:
            Dict with total counts and basic stats.
        """
        if self._summary_cache is not None:
            return self._summary_cache

        df = self.dataset.annotations_df

        self._summary_cache = {
            "split": self.dataset.split,
            "num_images": len(self.dataset.images_df),
            "num_annotations": len(df),
            "num_classes": self.dataset.num_classes,
            "class_names": list(self.dataset.class_names),
            "annotations_per_image": {
                "mean": df.groupby("image_id").size().mean(),
                "std": df.groupby("image_id").size().std(),
                "min": df.groupby("image_id").size().min(),
                "max": df.groupby("image_id").size().max(),
            },
            "size_distribution": df["size_class"].value_counts().to_dict(),
            "area_stats": {
                "mean": df["area"].mean(),
                "std": df["area"].std(),
                "min": df["area"].min(),
                "max": df["area"].max(),
                "median": df["area"].median(),
            },
        }

        return self._summary_cache

    def class_distribution(self) -> pd.DataFrame:
        """Get class distribution as a DataFrame.

        Returns:
            DataFrame with columns: category_name, count, percentage.
        """
        df = self.dataset.annotations_df

        counts = df["category_name"].value_counts()
        total = len(df)

        result = pd.DataFrame(
            {
                "category_name": counts.index,
                "count": counts.values,
                "percentage": (counts.values / total * 100).round(2),
            }
        )

        return result.sort_values("count", ascending=False).reset_index(drop=True)

    def box_size_histogram(
        self,
        save_path: Path | None = None,
        figsize: tuple[int, int] = (12, 6),
        bins: int = 50,
    ) -> plt.Figure:
        """Generate histogram of box sizes with small/medium/large regions.

        Args:
            save_path: Optional path to save the figure.
            figsize: Figure size.
            bins: Number of histogram bins.

        Returns:
            Matplotlib Figure object.
        """
        df = self.dataset.annotations_df
        thresholds = self.dataset.size_thresholds

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left plot: max_dim distribution
        ax1 = axes[0]
        max_dims = df["max_dim"].values

        ax1.hist(max_dims, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")

        # Add threshold lines
        ax1.axvline(
            thresholds.small,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Small ≤ {thresholds.small}px",
        )
        ax1.axvline(
            thresholds.medium,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Medium ≤ {thresholds.medium}px",
        )

        # Add size class counts as text (upper left to avoid legend overlap)
        size_counts = df["size_class"].value_counts()
        total = len(df)
        small_cnt = size_counts.get("small", 0)
        med_cnt = size_counts.get("medium", 0)
        large_cnt = size_counts.get("large", 0)
        text = "\n".join(
            [
                f"Small: {small_cnt:,} ({small_cnt / total * 100:.1f}%)",
                f"Medium: {med_cnt:,} ({med_cnt / total * 100:.1f}%)",
                f"Large: {large_cnt:,} ({large_cnt / total * 100:.1f}%)",
            ]
        )
        ax1.text(
            0.02,
            0.95,
            text,
            transform=ax1.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax1.set_xlabel("Max Dimension (pixels)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Box Size Distribution - {self.dataset.split}")
        ax1.legend(loc="upper right")

        # Right plot: size class bar chart
        ax2 = axes[1]
        size_order = ["small", "medium", "large"]
        counts = [size_counts.get(s, 0) for s in size_order]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        bars = ax2.bar(size_order, counts, color=colors, edgecolor="black")

        # Add count labels on bars
        for bar, count in zip(bars, counts, strict=True):
            height = bar.get_height()
            ax2.annotate(
                f"{count:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax2.set_xlabel("Size Class")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Size Class Distribution - {self.dataset.split}")

        # Add threshold info
        thresh_text = (
            f"Thresholds: small ≤ {thresholds.small}px, medium ≤ {thresholds.medium}px"
        )
        ax2.text(
            0.5,
            -0.12,
            thresh_text,
            transform=ax2.transAxes,
            ha="center",
            fontsize=9,
            style="italic",
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved box size histogram to {save_path}")

        return fig

    def class_distribution_chart(
        self,
        save_path: Path | None = None,
        figsize: tuple[int, int] = (12, 6),
        horizontal: bool = False,
    ) -> plt.Figure:
        """Generate bar chart of class distribution.

        Args:
            save_path: Optional path to save the figure.
            figsize: Figure size.
            horizontal: If True, use horizontal bars.

        Returns:
            Matplotlib Figure object.
        """
        dist_df = self.class_distribution()

        fig, ax = plt.subplots(figsize=figsize)

        if horizontal:
            bars = ax.barh(
                dist_df["category_name"],
                dist_df["count"],
                color="steelblue",
                edgecolor="black",
            )
            ax.set_xlabel("Count")
            ax.set_ylabel("Category")
            ax.invert_yaxis()  # Put the most frequent classes at the top

            # Add count labels
            for bar in bars:
                width = bar.get_width()
                ax.annotate(
                    f"{int(width):,}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
        else:
            bars = ax.bar(
                dist_df["category_name"],
                dist_df["count"],
                color="steelblue",
                edgecolor="black",
            )
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", labelrotation=45)

            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_title(f"Class Distribution - {self.dataset.split}")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved class distribution chart to {save_path}")

        return fig

    def per_image_stats(self) -> pd.DataFrame:
        """Compute per-image statistics.

        Returns:
            DataFrame with image_id and various statistics.
        """
        df = self.dataset.annotations_df

        stats = df.groupby("image_id").agg(
            {
                "annotation_id": "count",
                "area": ["mean", "sum"],
                "max_dim": ["mean", "max"],
            }
        )

        # Flatten column names
        stats.columns = [
            "num_annotations",
            "mean_area",
            "total_area",
            "mean_max_dim",
            "max_max_dim",
        ]

        # Add size class counts per image
        size_pivot = df.pivot_table(
            index="image_id",
            columns="size_class",
            values="annotation_id",
            aggfunc="count",
            fill_value=0,
        )
        size_pivot.columns = [f"num_{col}" for col in size_pivot.columns]

        stats = stats.join(size_pivot)

        return stats.reset_index()

    def export_summary_json(self, save_path: Path) -> None:
        """Export summary statistics to JSON.

        Args:
            save_path: Path to save JSON file.
        """
        import json

        summary = self.summary()

        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj

        converted = {
            k: (
                convert(v)
                if not isinstance(v, dict)
                else {kk: convert(vv) for kk, vv in v.items()}
            )
            for k, v in summary.items()
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(converted, f, indent=2)

        logger.info(f"Exported summary to {save_path}")

    def generate_report(self, output_dir: Path, prefix: str | None = None) -> None:
        """Generate all statistics reports (JSON, histograms, charts).

        Args:
            output_dir: Directory to save the reports.
            prefix: Optional prefix for filenames. Default is dataset split name.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = prefix or self.dataset.split or "dataset"

        # 1. Export JSON summary
        self.export_summary_json(output_dir / f"{name}_stats.json")

        # 2. Generate box size histogram
        self.box_size_histogram(output_dir / f"{name}_box_sizes.png")

        # 3. Generate class distribution chart
        self.class_distribution_chart(output_dir / f"{name}_class_dist.png")

        logger.info(f"Full statistics report generated in {output_dir}")
