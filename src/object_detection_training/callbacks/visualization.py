import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import supervision as sv
import torch
import wandb
from loguru import logger
from PIL import Image

from object_detection_training.utils.boxes import cxcywh_to_xyxy


class VisualizationCallback(L.Callback):
    """
    Callback to visualize predictions on fixed validation samples.

    Logs images to disk and WandB table row-by-row at each epoch.
    """

    def __init__(
        self,
        num_samples: int = 10,
        confidence_threshold: float = 0.3,
        output_dir: str = "outputs",
        mean: List[float] = [123.675, 116.28, 103.53],
        std: List[float] = [58.395, 57.12, 57.375],
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

        self.confidence_threshold = confidence_threshold

    def _denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor to numpy array [H, W, 3] (0-255)."""
        # tensor is [3, H, W]
        tensor = tensor.cpu()

        # Denormalize using std and mean (result is back in 0-255 range)
        tensor = tensor * self.std + self.mean

        array = tensor.permute(1, 2, 0).numpy()
        return np.clip(array, 0, 255).astype(np.uint8)

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
        if trainer.datamodule:
            class_names = getattr(trainer.datamodule, "class_names", "ATTR_MISSING")
            logger.info(
                f"Validation Epoch Start: Datamodule found. Classes: {class_names}"
            )
        else:
            logger.info("Validation Epoch Start: No datamodule found in trainer.")

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
        save_dir = self.output_dir / f"epoch_{epoch:03d}" / "images"
        save_dir.mkdir(parents=True, exist_ok=True)

        device = next(pl_module.parameters()).device
        pl_module.eval()

        with torch.no_grad():
            for sample in self.val_samples:
                img_tensor = sample["image"].to(device)  # [3, H, W]
                # Add batch dim
                batch_imgs = img_tensor.unsqueeze(0)

                # Model forward
                outputs = pl_module(batch_imgs)
                preds = pl_module.get_predictions(outputs, confidence_threshold=0.01)[0]

                # Denormalize image for drawing
                image_np = self._denormalize(img_tensor.cpu())
                img_h, img_w = image_np.shape[:2]

                # Use Supervision
                # Boxes from get_predictions are xyxy but normalized [0, 1] if
                # original_sizes not passed
                pred_boxes = preds["boxes"].cpu().numpy()
                scale = np.array([img_w, img_h, img_w, img_h])
                pred_boxes_abs = pred_boxes * scale

                detections = sv.Detections(
                    xyxy=pred_boxes_abs,
                    confidence=preds["scores"].cpu().numpy(),
                    class_id=preds["labels"].cpu().numpy().astype(int),
                )

                # Debug logging
                num_raw = len(preds["scores"])
                high_conf_preds = preds["scores"] > 0.05
                num_high_conf = high_conf_preds.sum().item()

                if num_raw > 0:
                    max_conf = preds["scores"].max().item()
                    logger.debug(
                        f"Image {sample.get('image_id')}: {num_raw} raw detections. "
                        f"Max confidence = {max_conf:.4f}. "
                        f"{num_high_conf} detections > 0.05"
                    )
                else:
                    logger.debug(
                        f"Image {sample.get('image_id')}: No raw detections found."
                    )

                # Filter by confidence? Default 0.3 for viz
                detections = detections[
                    detections.confidence > self.confidence_threshold
                ]

                # Draw Predictions
                pred_image = image_np.copy()
                pred_image = self.box_annotator.annotate(
                    scene=pred_image, detections=detections
                )

                # Use class names for predictions
                # We prioritize class names from datamodule
                pred_labels_list: Optional[List[str]] = None
                class_names = getattr(trainer.datamodule, "class_names", None)

                if class_names is None:
                    logger.info(f"trainer.datamodule: {trainer.datamodule}")
                    if trainer.datamodule:
                        logger.info(f"datamodule attributes: {dir(trainer.datamodule)}")

                if class_names and detections.class_id is not None:
                    pred_labels_list = []
                    for class_id in detections.class_id:
                        cid_int = int(class_id)
                        name = (
                            class_names[cid_int]
                            if cid_int < len(class_names)
                            else f"class_{cid_int}"
                        )
                        pred_labels_list.append(name)

                # Fallback labels if names not available
                if pred_labels_list is None:
                    # supervision default is to show class_id if labels=None
                    pass

                if pred_labels_list:
                    pred_image = self.label_annotator.annotate(
                        scene=pred_image, detections=detections, labels=pred_labels_list
                    )
                else:
                    pred_image = self.label_annotator.annotate(
                        scene=pred_image, detections=detections
                    )

                # Save Prediction to disk for reference
                img_id = sample["image_id"]
                file_path = save_dir / f"val_img_{img_id}.jpg"
                Image.fromarray(pred_image).save(file_path)

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

        save_dir = self.output_dir / "test_results" / "images"
        save_dir.mkdir(parents=True, exist_ok=True)

        log_images = []
        device = next(pl_module.parameters()).device
        pl_module.eval()

        with torch.no_grad():
            for sample in self.test_samples:
                img_tensor = sample["image"].to(device)
                outputs = pl_module(img_tensor.unsqueeze(0))
                preds = pl_module.get_predictions(outputs, confidence_threshold=0.0)[0]

                image_np = self._denormalize(img_tensor.cpu())
                img_h, img_w = image_np.shape[:2]

                # Scale boxes to absolute pixel coordinates
                pred_boxes = preds["boxes"].cpu().numpy()
                scale = np.array([img_w, img_h, img_w, img_h])
                pred_boxes_abs = pred_boxes * scale

                detections = sv.Detections(
                    xyxy=pred_boxes_abs,
                    confidence=preds["scores"].cpu().numpy(),
                    class_id=preds["labels"].cpu().numpy().astype(int),
                )

                # Filter by confidence for test viz as well (using 0.3 as default)
                detections = detections[
                    detections.confidence > self.confidence_threshold
                ]

                annotated_image = image_np.copy()
                annotated_image = self.box_annotator.annotate(
                    scene=annotated_image, detections=detections
                )

                # Fetch class names
                pred_labels_list = None
                class_names = getattr(trainer.datamodule, "class_names", None)
                if class_names:
                    pred_labels_list = []
                    for cid, conf in zip(detections.class_id, detections.confidence):
                        cid_int = int(cid)
                        if cid_int < len(class_names):
                            name = class_names[cid_int]
                        else:
                            name = f"unknown_{cid_int}"
                        pred_labels_list.append(f"{name} {conf:.2f}")

                if pred_labels_list:
                    annotated_image = self.label_annotator.annotate(
                        scene=annotated_image,
                        detections=detections,
                        labels=pred_labels_list,
                    )
                else:
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

    def _visualize_gt(
        self,
        samples: List[Dict[str, Any]],
        split: str,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Visualize ground truth annotations for a given split."""
        if not samples:
            return

        save_dir = self.output_dir / "ground_truth" / split
        save_dir.mkdir(parents=True, exist_ok=True)

        log_images = []
        # Device for denormalization if needed (though we do it on cpu)
        # Just use cpu for GT viz

        for sample in samples:
            img_tensor = sample["image"]
            target = sample["target"]
            img_id = sample.get("image_id", "unknown")
            if isinstance(img_id, torch.Tensor):
                img_id = img_id.item()

            image_np = self._denormalize(img_tensor)

            # Ground Truth Detections
            # Target boxes are normalized cxcywh [0, 1] from RFDETR transforms
            # We need to convert to absolute xyxy for supervision

            boxes_tensor = (
                target["boxes"].cpu()
                if isinstance(target["boxes"], torch.Tensor)
                else torch.tensor(target["boxes"])
            )
            labels = (
                target["labels"].cpu().numpy().astype(int)
                if isinstance(target["labels"], torch.Tensor)
                else target["labels"].astype(int)
            )

            img_h, img_w = image_np.shape[:2]
            scale_tensor = torch.tensor([img_w, img_h, img_w, img_h])

            # Un-normalize
            boxes_abs = boxes_tensor * scale_tensor

            # cxcywh -> xyxy
            boxes_xyxy = cxcywh_to_xyxy(boxes_abs).numpy()

            detections = sv.Detections(
                xyxy=boxes_xyxy,
                class_id=labels,
            )

            # Use class names if available
            labels_list = None
            if (
                hasattr(trainer.datamodule, "class_names")
                and trainer.datamodule.class_names
            ):
                try:
                    labels_list = [
                        trainer.datamodule.class_names[class_id] for class_id in labels
                    ]
                except IndexError:
                    logger.warning(f"Class ID out of range for class names: {labels}")

            annotated_image = image_np.copy()
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=detections
            )

            if labels_list:
                annotated_image = self.label_annotator.annotate(
                    scene=annotated_image, detections=detections, labels=labels_list
                )
            else:
                annotated_image = self.label_annotator.annotate(
                    scene=annotated_image, detections=detections
                )

            # Save to disk
            Image.fromarray(annotated_image).save(
                save_dir / f"{split}_img_{img_id}.jpg"
            )

            log_images.append(
                wandb.Image(annotated_image, caption=f"{split} Ground Truth {img_id}")
            )

        # Log to WandB
        for logger_inst in trainer.loggers:
            if isinstance(logger_inst, L.pytorch.loggers.WandbLogger):
                columns = [f"Image_{i}" for i in range(len(log_images))]
                table = wandb.Table(columns=columns)
                table.add_data(*log_images)
                logger_inst.experiment.log({f"ground_truth/{split}": table})

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Collect and visualize ground truth samples at the start of training."""
        logger.info("Visualizing ground truth samples...")

        if not trainer.datamodule:
            logger.warning("No datamodule found, skipping ground truth visualization.")
            return

        # We need to manually call setup if it hasn't been called yet,
        # but usually Trainer calls it.
        # However, at on_fit_start, setup might be done?
        # Lightning timeline: setup -> on_fit_start -> ...
        # So dataloaders should be available or creatable.

        splits = []

        # Train
        try:
            if trainer.datamodule.train_dataloader():
                splits.append(("train", trainer.datamodule.train_dataloader()))
        except Exception as e:
            logger.warning(f"Could not get train dataloader for visualization: {e}")

        # Val
        try:
            if trainer.datamodule.val_dataloader():
                splits.append(("val", trainer.datamodule.val_dataloader()))
        except Exception as e:
            logger.warning(f"Could not get val dataloader for visualization: {e}")

        # Test
        try:
            # Some datamodules might not have test_dataloader implemented or set up
            if (
                hasattr(trainer.datamodule, "test_dataloader")
                and trainer.datamodule.test_dataloader()
            ):
                splits.append(("test", trainer.datamodule.test_dataloader()))
        except Exception as e:
            # This is fine, test might not be available
            logger.debug(f"Could not get test dataloader for visualization: {e}")

        for split_name, dataloader in splits:
            logger.info(f"Collecting {split_name} samples for GT visualization...")
            samples = self._collect_samples(dataloader, self.num_samples)
            self._visualize_gt(samples, split_name, trainer, pl_module)

        logger.info("Ground truth visualization complete.")
