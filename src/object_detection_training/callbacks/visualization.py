import random
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import numpy as np
import supervision as sv
import torch
import wandb
from loguru import logger
from PIL import Image


class VisualizationCallback(L.Callback):
    """
    Callback to visualize predictions on fixed validation samples.

    Logs images to disk and WandB table row-by-row at each epoch.
    """

    def __init__(
        self,
        num_samples: int = 10,
        output_dir: str = "outputs",
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        """
        Initialize visualization callback.

        Args:
            num_samples: Number of images to visualize.
            output_dir: Directory to save visualized images.
            mean: Normalization mean (RGB).
            std: Normalization std (RGB).
        """
        super().__init__()
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

        self.val_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []

        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def _denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor to numpy array [H, W, 3] (0-255)."""
        # tensor is [3, H, W]
        tensor = tensor.cpu() * self.std + self.mean
        tensor = torch.clamp(tensor, 0, 1)
        array = tensor.permute(1, 2, 0).numpy()
        return (array * 255).astype(np.uint8)

    def _collect_samples(self, dataloader, num: int) -> List[Dict[str, Any]]:
        """Collect random samples from dataloader."""
        samples = []
        dataset = dataloader.dataset

        # Random indices
        total = len(dataset)
        indices = random.sample(range(total), min(num, total))

        # We need to use the collate_fn to properly batch them if we process them?
        # Or just access dataset directly and treat as batch size 1.
        # Models usually expect batch dimension.

        for idx in indices:
            img, target = dataset[idx]
            # dataset[idx] returns (img, target) tuple usually
            # img is Tensor [3, H, W]
            # target is Dict

            samples.append({"image": img, "target": target, "image_id": idx})

        return samples

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Collect validation samples on first epoch."""
        if not self.val_samples:
            logger.info(
                f"Collecting {self.num_samples} validation samples for visualization..."
            )
            # We assume datamodule is available
            if trainer.datamodule and trainer.datamodule.val_dataloader():
                self.val_samples = self._collect_samples(
                    trainer.datamodule.val_dataloader(), self.num_samples
                )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Visualize predictions on validation samples."""
        if not self.val_samples:
            return

        epoch = trainer.current_epoch
        save_dir = self.output_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        log_images_data = []

        device = next(pl_module.parameters()).device
        pl_module.eval()

        with torch.no_grad():
            for sample in self.val_samples:
                img_tensor = sample["image"].to(device)  # [3, H, W]
                # Add batch dim
                batch_imgs = img_tensor.unsqueeze(0)

                # Model forward
                outputs = pl_module(batch_imgs)
                preds = pl_module.get_predictions(outputs)[0]

                # Denormalize image for drawing
                image_np = self._denormalize(img_tensor.cpu())

                # Use Supervision
                detections = sv.Detections(
                    xyxy=preds["boxes"].cpu().numpy(),
                    confidence=preds["scores"].cpu().numpy(),
                    class_id=preds["labels"].cpu().numpy().astype(int),
                )

                # Filter by confidence? Default 0.3 for viz
                # detections = detections[detections.confidence > 0.3]

                # Draw
                # Draw Ground Truth
                # Need to convert target format to sv.Detections
                # Target is dict with 'boxes', 'labels'
                target = sample["target"]
                gt_detections = sv.Detections(
                    xyxy=target["boxes"].numpy(),
                    class_id=target["labels"].numpy().astype(int),
                )

                gt_image = image_np.copy()
                gt_image = self.box_annotator.annotate(
                    scene=gt_image, detections=gt_detections
                )
                gt_image = self.label_annotator.annotate(
                    scene=gt_image, detections=gt_detections
                )

                # Draw Predictions
                pred_image = image_np.copy()
                pred_image = self.box_annotator.annotate(
                    scene=pred_image, detections=detections
                )
                pred_image = self.label_annotator.annotate(
                    scene=pred_image, detections=detections
                )

                # Save Prediction to disk for reference
                img_id = sample["image_id"]
                file_path = save_dir / f"val_img_{img_id}.jpg"
                Image.fromarray(pred_image).save(file_path)

                # Store raw images for table logging
                log_images_data.append((gt_image, pred_image))

        # Log to WandB
        # We assume WandB logger is present
        for logger_inst in trainer.loggers:
            if isinstance(logger_inst, L.pytorch.loggers.WandbLogger):
                # Prepare table data for this epoch
                # Columns: epoch, image_id, ground_truth, prediction
                table_data = []
                for sample, (gt_img, pred_img) in zip(
                    self.val_samples, log_images_data
                ):
                    table_data.append(
                        [
                            epoch,
                            sample["image_id"],
                            wandb.Image(gt_img, caption="Ground Truth"),
                            wandb.Image(pred_img, caption="Prediction"),
                        ]
                    )

                # Log a new table for this epoch
                # W&B UI allows using the step slider to view history
                columns = ["epoch", "image_id", "ground_truth", "prediction"]
                table = wandb.Table(columns=columns, data=table_data)
                logger_inst.experiment.log({"val_predictions": table})

    def on_test_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Collect test samples."""
        if not self.test_samples:
            if trainer.datamodule and trainer.datamodule.test_dataloader():
                self.test_samples = self._collect_samples(
                    trainer.datamodule.test_dataloader(), self.num_samples
                )

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Visualize test samples."""
        if not self.test_samples:
            return

        save_dir = self.output_dir / "test_results"
        save_dir.mkdir(parents=True, exist_ok=True)

        log_images = []
        device = next(pl_module.parameters()).device
        pl_module.eval()

        with torch.no_grad():
            for sample in self.test_samples:
                img_tensor = sample["image"].to(device)
                outputs = pl_module(img_tensor.unsqueeze(0))
                preds = pl_module.get_predictions(outputs)[0]

                image_np = self._denormalize(img_tensor.cpu())

                detections = sv.Detections(
                    xyxy=preds["boxes"].cpu().numpy(),
                    confidence=preds["scores"].cpu().numpy(),
                    class_id=preds["labels"].cpu().numpy().astype(int),
                )

                annotated_image = image_np.copy()
                annotated_image = self.box_annotator.annotate(
                    scene=annotated_image, detections=detections
                )
                annotated_image = self.label_annotator.annotate(
                    scene=annotated_image, detections=detections
                )

                img_id = sample["image_id"]
                Image.fromarray(annotated_image).save(
                    save_dir / f"test_img_{img_id}.jpg"
                )
                log_images.append(
                    wandb.Image(annotated_image, caption=f"Test Image {img_id}")
                )

        for logger_inst in trainer.loggers:
            if isinstance(logger_inst, L.pytorch.loggers.WandbLogger):
                columns = [f"Image_{i}" for i in range(len(log_images))]
                table = wandb.Table(columns=columns)
                table.add_data(*log_images)
                logger_inst.experiment.log({"test_results_table": table})
