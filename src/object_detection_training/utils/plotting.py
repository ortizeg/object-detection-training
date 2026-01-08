from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    class_name: str,
    save_path: Path,
    ap: Optional[float] = None,
) -> None:
    """
    Plot and save Precision-Recall curve.

    Args:
        recall: Recall values.
        precision: Precision values.
        class_name: Name of the class.
        save_path: Path to save the plot.
        ap: Average Precision value (optional).
    """
    plt.figure(figsize=(10, 8))

    # Plot curve
    plt.plot(
        recall,
        precision,
        label=f"{class_name} (AP: {ap:.3f})" if ap is not None else class_name,
        linewidth=2,
    )

    # Formatting
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve - {class_name}", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_detection_curves_plots(
    curves_data: Dict[str, Any],
    class_names: Optional[List[str]],
    output_dir: Path,
    prefix: str = "",
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save plots for all classes in curves_data.

    Args:
        curves_data: Dictionary containing curve data per class.
        class_names: List of class names.
        output_dir: Directory to save plots.
        prefix: Prefix for filenames.
        metrics: Optional dictionary containing AP metrics to display on plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for c_idx, curve in curves_data.items():
        # Handle class name. c_idx might be int or string depending on how it's stored,
        # usually int from curves.py
        try:
            c_idx_int = int(c_idx)
            name = (
                class_names[c_idx_int]
                if class_names and c_idx_int < len(class_names)
                else f"class_{c_idx}"
            )
        except (ValueError, TypeError):
            name = str(c_idx)

        # Get AP if available
        ap = None
        if metrics:
            # Check for map_per_class
            map_per_class = metrics.get("map_per_class")
            if map_per_class is not None:
                # Assuming map_per_class is a tensor or list aligned with class indices
                if isinstance(map_per_class, torch.Tensor):
                    ap = map_per_class[int(c_idx)].item()
                elif isinstance(map_per_class, list):
                    ap = map_per_class[int(c_idx)]

        # PR Curve
        save_path = output_dir / f"{prefix}pr_curve_{name}.png"
        plot_pr_curve(curve["recall"], curve["precision"], name, save_path, ap=ap)
