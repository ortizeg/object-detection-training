import json
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table


class DatasetStatisticsCallback(L.Callback):
    """
    Callback to compute and visualize dataset statistics at the start of training.
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

        stats = {}
        splits = ["train", "val", "test"]
        class_names = getattr(datamodule, "class_names", [])
        num_classes = len(class_names)

        for split in splits:
            dataset = None
            if split == "train":
                dataset = getattr(datamodule, "train_dataloader", lambda: None)()
                if dataset is None:
                    # Try direct access if dataloader method not ready
                    dataset = getattr(datamodule, "train_dataset", None)
            elif split == "val":
                dataset = getattr(datamodule, "val_dataloader", lambda: None)()
                if dataset is None:
                    dataset = getattr(datamodule, "val_dataset", None)
            elif split == "test":
                # Ensure test dataset is setup as it's usually done later
                try:
                    datamodule.setup("test")
                except Exception as e:
                    logger.warning(f"Failed to setup test dataset: {e}")

                dataset = getattr(datamodule, "test_dataloader", lambda: None)()
                if dataset is None:
                    dataset = getattr(datamodule, "test_dataset", None)

            # Extract the actual dataset from dataloader if needed
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset

            if dataset is not None:
                stats[split] = self._compute_split_stats(dataset, num_classes)
            else:
                logger.warning(f"No {split} dataset found.")

        if not stats:
            logger.warning("No datasets found to compute statistics.")
            return

        # 1. Generate Summary Table
        console = Console()
        summary_table = Table(
            title="Dataset Summary",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        summary_table.add_column("Split", style="cyan")
        summary_table.add_column("Images", justify="right", style="green")
        summary_table.add_column("Detection Boxes", justify="right", style="green")

        total_images = 0
        total_detections = 0
        for split, s in stats.items():
            summary_table.add_row(
                split.capitalize(), str(s["total_images"]), str(s["total_detections"])
            )
            total_images += s["total_images"]
            total_detections += s["total_detections"]

        summary_table.add_section()
        summary_table.add_row(
            "Total", str(total_images), str(total_detections), style="bold"
        )

        logger.info("\nDataset Summary:")
        console.print(summary_table)

        # 2. Generate Class-wise Table
        class_table = Table(
            title="Class Distribution",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        class_table.add_column("Class", style="cyan")
        for split in splits:
            if split in stats:
                class_table.add_column(
                    split.capitalize(), justify="right", style="green"
                )

        for i, name in enumerate(class_names):
            row = [name]
            for split in splits:
                if split in stats:
                    row.append(str(stats[split]["class_counts"][i]))
            class_table.add_row(*row)

        logger.info("\nClass Distribution:")
        console.print(class_table)

        # 3. Save to JSON
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_stats = {
            "summary": {
                "splits": {
                    s.capitalize(): {
                        "images": stats[s]["total_images"],
                        "detections": stats[s]["total_detections"],
                    }
                    for s in stats
                },
                "total": {"images": total_images, "detections": total_detections},
            },
            "class_distribution": {
                name: {
                    s.capitalize(): int(stats[s]["class_counts"][i])
                    for s in stats
                    if s in stats
                }
                for i, name in enumerate(class_names)
            },
        }
        with open(self.output_dir / "dataset_stats.json", "w") as f:
            json.dump(json_stats, f, indent=4)

        # 4. Plot Histograms
        self._plot_statistics(stats, class_names)

        logger.info(f"Dataset statistics exported to {self.output_dir}")

    def _compute_split_stats(self, dataset: Any, num_classes: int) -> Dict[str, Any]:
        """Compute stats for a single dataset split."""
        total_images = len(dataset)
        class_counts = np.zeros(num_classes, dtype=int)
        total_detections = 0

        # We assume dataset has 'coco' attribute or similar if it's CocoDetection
        # or we can iterate if it's small. For large datasets, iteration might be slow.
        # But for statistics at start, it's usually acceptable if done efficiently.

        if hasattr(dataset, "coco"):
            # Efficient COCO-specific stats
            coco = dataset.coco
            for ann_id in coco.getAnnIds():
                ann = coco.loadAnns(ann_id)[0]
                cat_id = ann["category_id"]
                # Map cat_id to 0-indexed if label_map exists in dataset
                if (
                    hasattr(dataset, "prepare")
                    and hasattr(dataset.prepare, "label_map")
                    and dataset.prepare.label_map
                ):
                    label = dataset.prepare.label_map.get(cat_id)
                    if label is not None:
                        class_counts[label] += 1
                        total_detections += 1
                else:
                    # Fallback to category_id - 1 if no map
                    class_counts[cat_id - 1] += 1
                    total_detections += 1
        else:
            # Fallback iteration (might be slow for huge datasets, but necessary)
            logger.info("Iterating dataset for statistics...")
            for i in range(len(dataset)):
                _, target = dataset[i]
                labels = target.get("labels")
                if labels is not None:
                    for label in labels:
                        class_counts[int(label)] += 1
                        total_detections += 1

        return {
            "total_images": total_images,
            "total_detections": total_detections,
            "class_counts": class_counts,
        }

    def _plot_statistics(self, stats: Dict[str, Dict], class_names: List[str]):
        """Plot and save bar graphs for class distribution."""
        num_splits = len(stats)
        fig, axes = plt.subplots(
            num_splits, 1, figsize=(12, 5 * num_splits), squeeze=False
        )

        for i, (split, s) in enumerate(stats.items()):
            ax = axes[i, 0]
            counts = s["class_counts"]
            bars = ax.bar(class_names, counts, color="skyblue", edgecolor="navy")

            ax.set_title(f"Class Distribution - {split.capitalize()}")
            ax.set_ylabel("Number of Detections")
            ax.set_xlabel("Class")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Add counts on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(self.output_dir / "dataset_stats_histogram.png", dpi=300)
        plt.close()

        # Also save a combined side-by-side bar chart
        if num_splits > 1:
            plt.figure(figsize=(14, 8))
            x = np.arange(len(class_names))
            width = 0.8 / num_splits

            for i, (split, s) in enumerate(stats.items()):
                offset = (i - (num_splits - 1) / 2) * width
                plt.bar(x + offset, s["class_counts"], width, label=split.capitalize())

            plt.xlabel("Class")
            plt.ylabel("Number of Detections")
            plt.title("Class Distribution Comparison")
            plt.xticks(x, class_names, rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "dataset_stats_comparison.png", dpi=300)
            plt.close()
