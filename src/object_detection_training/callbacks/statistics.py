"""
Dataset statistics callback for PyTorch Lightning.

Computes and visualizes dataset statistics at the start of training
using the DatasetStatistics class.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from object_detection_training.data.dataset_stats import DatasetStatistics


class DatasetStatisticsCallback(L.Callback):
    """
    Callback to compute and visualize dataset statistics at the start of training.

    Uses the DatasetStatistics class for computing box size distributions,
    class distributions, and per-image annotation counts.
    """

    def __init__(self, output_dir: str = "outputs"):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute statistics for train, val, and test sets."""
        logger.info("Computing dataset statistics...")

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            logger.warning("No datamodule found in trainer. Skipping statistics.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats_list = []

        # Process each split
        for split in ["train", "val", "test"]:
            detection_dataset = self._get_detection_dataset(datamodule, split)
            if detection_dataset is not None:
                stats = DatasetStatistics(detection_dataset)
                stats_list.append((split, stats))

        if not stats_list:
            logger.warning("No detection datasets found to compute statistics.")
            return

        # Generate summary table
        self._print_summary_table(stats_list)

        # Generate class distribution table
        self._print_class_table(stats_list)

        # Generate reports for each split
        for split, stats in stats_list:
            stats.generate_report(self.output_dir / split)

        logger.info(f"Dataset statistics exported to {self.output_dir}")

    def _get_detection_dataset(self, datamodule, split: str):
        """Get the underlying DetectionDataset for a split."""
        # Try to get detection dataset directly from COCODataModule
        if split == "train" and hasattr(datamodule, "train_detection_dataset"):
            return datamodule.train_detection_dataset

        # For val/test, we need to create them
        if hasattr(datamodule, "_create_detection_dataset"):
            try:
                if split == "train":
                    return datamodule._create_detection_dataset(
                        datamodule.train_path, "train"
                    )
                elif split == "val":
                    return datamodule._create_detection_dataset(
                        datamodule.val_path, "val"
                    )
                elif split == "test" and datamodule.test_path:
                    return datamodule._create_detection_dataset(
                        datamodule.test_path, "test"
                    )
            except Exception as e:
                logger.debug(f"Could not create detection dataset for {split}: {e}")

        return None

    def _print_summary_table(self, stats_list):
        """Print a summary table with rich formatting."""
        console = Console()
        table = Table(
            title="Dataset Summary",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        table.add_column("Split", style="cyan")
        table.add_column("Images", justify="right", style="green")
        table.add_column("Annotations", justify="right", style="green")
        table.add_column("Classes", justify="right", style="green")
        table.add_column("Avg/Image", justify="right", style="yellow")

        total_images = 0
        total_anns = 0

        for split, stats in stats_list:
            summary = stats.summary()
            total_images += summary["num_images"]
            total_anns += summary["num_annotations"]
            avg_per_img = summary["annotations_per_image"]["mean"]

            table.add_row(
                split.capitalize(),
                str(summary["num_images"]),
                str(summary["num_annotations"]),
                str(summary["num_classes"]),
                f"{avg_per_img:.1f}",
            )

        table.add_section()
        table.add_row(
            "Total",
            str(total_images),
            str(total_anns),
            "-",
            f"{total_anns / total_images:.1f}" if total_images > 0 else "-",
            style="bold",
        )

        logger.info("\nDataset Summary:")
        console.print(table)

    def _print_class_table(self, stats_list):
        """Print class distribution table."""
        if not stats_list:
            return

        console = Console()
        table = Table(
            title="Class Distribution",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        table.add_column("Class", style="cyan")
        for split, _ in stats_list:
            table.add_column(split.capitalize(), justify="right", style="green")
        table.add_column("Total", justify="right", style="yellow")

        # Get class names from first dataset
        first_stats = stats_list[0][1]
        class_names = first_stats.dataset.class_names

        for class_name in class_names:
            row = [class_name]
            total = 0
            for _, stats in stats_list:
                class_dist = stats.class_distribution()
                count = class_dist[class_dist["category_name"] == class_name][
                    "count"
                ].values
                count = int(count[0]) if len(count) > 0 else 0
                row.append(str(count))
                total += count
            row.append(str(total))
            table.add_row(*row)

        logger.info("\nClass Distribution:")
        console.print(table)

    def state_dict(self):
        """Return callback state."""
        return {}

    def load_state_dict(self, state_dict):
        """Load callback state."""
        pass
