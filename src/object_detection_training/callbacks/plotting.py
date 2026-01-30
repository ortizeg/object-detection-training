from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
from loguru import logger


class TrainingHistoryPlotter(L.Callback):
    """
    Callback to plot and update training and validation metrics each epoch.
    Saves and overwrites loss_history.png and map_history.png.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the plotter.

        Args:
            output_dir: Directory where the plots will be saved.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mAP": [],
            "val_mAP50": [],
            "val_mAP75": [],
        }
        self.epochs = []

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Collect metrics and update plots at the end of each training epoch."""
        epoch = trainer.current_epoch
        logger.debug(
            f"TrainingHistoryPlotter.on_train_epoch_end called for epoch {epoch}"
        )
        self.epochs.append(epoch)

        # Retrieve logged metrics
        # Note: self.log in pl_module logs to trainer.callback_metrics
        metrics = trainer.callback_metrics

        # Loss metrics
        train_loss = metrics.get("train/loss_epoch") or metrics.get("train/loss")
        val_loss = metrics.get("val/loss_epoch") or metrics.get("val/loss")

        # mAP metrics
        val_map = metrics.get("val/mAP")
        val_map50 = metrics.get("val/mAP_50")
        val_map75 = metrics.get("val/mAP_75")

        # Update history (handle missing values with None or last value)
        self.history["train_loss"].append(
            train_loss.item()
            if train_loss is not None
            else (
                self.history["train_loss"][-1] if self.history["train_loss"] else None
            )
        )
        self.history["val_loss"].append(
            val_loss.item()
            if val_loss is not None
            else (self.history["val_loss"][-1] if self.history["val_loss"] else None)
        )
        self.history["val_mAP"].append(
            val_map.item()
            if val_map is not None
            else (self.history["val_mAP"][-1] if self.history["val_mAP"] else None)
        )
        self.history["val_mAP50"].append(
            val_map50.item()
            if val_map50 is not None
            else (self.history["val_mAP50"][-1] if self.history["val_mAP50"] else None)
        )
        self.history["val_mAP75"].append(
            val_map75.item()
            if val_map75 is not None
            else (self.history["val_mAP75"][-1] if self.history["val_mAP75"] else None)
        )

        # Log gathered metrics for debugging
        logger.debug(f"Gathered metrics: {list(metrics.keys())}")
        logger.debug(f"History state: {self.history}")

        # Plot and save
        try:
            self._plot_metrics()
        except Exception as e:
            logger.error(f"Failed to plot metrics: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _plot_metrics(self):
        """Draw and save plots, overwriting existing files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Loss Plot
        plt.figure(figsize=(10, 6))
        if any(v is not None for v in self.history["train_loss"]):
            plt.plot(
                self.epochs, self.history["train_loss"], label="Train Loss", marker="o"
            )
        if any(v is not None for v in self.history["val_loss"]):
            plt.plot(
                self.epochs, self.history["val_loss"], label="Val Loss", marker="s"
            )

        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_history.png", dpi=300)
        plt.close()

        # 2. mAP Plot
        plt.figure(figsize=(10, 6))

        metrics_to_plot = {
            "val_mAP": ("Val mAP", "s"),
            "val_mAP50": ("Val mAP50", "^"),
            "val_mAP75": ("Val mAP75", "v"),
        }

        any_plotted = False
        for key, (label, marker) in metrics_to_plot.items():
            if any(v is not None for v in self.history[key]):
                plt.plot(self.epochs, self.history[key], label=label, marker=marker)
                any_plotted = True

        plt.title("Validation mAP")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        if any_plotted:
            plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "map_history.png", dpi=300)
        plt.close()

        logger.info(f"Updated training history plots in {self.output_dir}")
