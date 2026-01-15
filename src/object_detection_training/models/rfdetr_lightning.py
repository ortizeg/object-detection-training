"""
RFDETR model wrappers for PyTorch Lightning.

This module wraps the rfdetr models to work with the Lightning training framework.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.models import build_criterion_and_postprocessors

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.utils.hydra import register

# Model checkpoint URLs for download
CHECKPOINT_URLS = {
    "nano": (
        "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
    ),
    "small": (
        "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth"
    ),
    "medium": (
        "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth"
    ),
    "large": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
}


def download_checkpoint(url: str, destination: Path) -> Path:
    """Download a checkpoint file if it doesn't exist."""
    import urllib.request

    destination = Path(destination)
    if destination.exists():
        logger.info(f"Checkpoint already exists: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint from {url}")

    try:
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Checkpoint downloaded to {destination}")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise

    return destination


class RFDETRLightningModel(BaseDetectionModel):
    """
    PyTorch Lightning wrapper for RFDETR models.

    Wraps the rfdetr package models to work with Lightning training framework.
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "nano": {"class": RFDETRNano, "checkpoint_key": "nano"},
        "small": {"class": RFDETRSmall, "checkpoint_key": "small"},
        "medium": {"class": RFDETRMedium, "checkpoint_key": "medium"},
        "large": {"class": RFDETRLarge, "checkpoint_key": "large"},
    }

    def __init__(
        self,
        variant: str = "small",
        num_classes: int = 80,
        pretrain_weights: Optional[str] = None,
        learning_rate: float = 2.5e-4,
        lr_encoder: float = 1.5e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.993,
        input_height: int = 512,
        input_width: int = 512,
        download_pretrained: bool = True,
        output_dir: str = "outputs",
    ):
        """
        Initialize RFDETR Lightning model.

        Args:
            variant: Model variant (nano, small, medium, large).
            num_classes: Number of detection classes.
            pretrain_weights: Path to pretrained weights file.
            input_height: Input image height.
            input_width: Input image width.
            learning_rate: Base learning rate.
            lr_encoder: Learning rate for encoder.
            weight_decay: Weight decay.
            warmup_epochs: Number of warmup epochs.
            use_ema: Enable EMA.
            ema_decay: EMA decay factor.
            download_pretrained: Download pretrained weights if not available.
            output_dir: Base directory for outputting results.
        """
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            use_ema=use_ema,
            ema_decay=ema_decay,
            input_height=input_height,
            input_width=input_width,
            output_dir=output_dir,
        )

        self.lr_encoder = lr_encoder
        self.variant = variant
        self.pretrain_weights = pretrain_weights
        self.download_pretrained = download_pretrained

        if variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Choose from {list(self.MODEL_VARIANTS.keys())}"
            )

        # Initialize the RFDETR model
        model_config = self.MODEL_VARIANTS[variant]
        model_class = model_config["class"]

        # Handle pretrained weights
        if pretrain_weights is None and download_pretrained:
            cache_dir = Path.home() / ".cache" / "rfdetr"
            checkpoint_path = cache_dir / f"rf-detr-{variant}.pth"
            if not checkpoint_path.exists():
                url = CHECKPOINT_URLS[model_config["checkpoint_key"]]
                download_checkpoint(url, checkpoint_path)
            pretrain_weights = str(checkpoint_path)

        logging_info = f"Initializing RFDETR {variant} model"
        logger.info(logging_info)
        self.rfdetr_wrapper = model_class(
            pretrain_weights=pretrain_weights,
            num_classes=num_classes,
            resolution=input_height,
        )

        # Register the actual nn.Module so Lightning/Optimizer can see parameters
        # valid chain based on forward usage: self.model.model.model
        # We assign it to self.model which is a standard name
        self.model = self.rfdetr_wrapper.model.model

        # Store internal model reference for export if needed
        # (though self.model is the nn.Module now)
        self._internal_model = self.model

        # Build criterion for training
        logger.info("Building criterion for training...")
        if build_criterion_and_postprocessors:
            # args are in the wrapper
            self.criterion, self.postprocessors = build_criterion_and_postprocessors(
                self.rfdetr_wrapper.model.args
            )
        else:
            self.criterion = None
            self.postprocessors = None

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Custom training step to handle RFDETR weighted losses."""
        images, targets = batch
        # self(images, targets) returns dict with 'loss' and 'train/...' keys
        outputs = self(images, targets)

        # Extract total loss
        total_loss = outputs["loss"]

        # Log total loss
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log other components
        for k, v in outputs.items():
            if k != "loss":
                self.log(k, v, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def forward(
        self, images: torch.Tensor, targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Batch of images [B, C, H, W].
            targets: Optional list of target dictionaries for training.

        Returns:
            Dictionary with losses (training) or predictions (inference).
        """
        if self._export_mode:
            # For ONNX export, return detections directly
            return self._forward_for_export(images)

        if self.training and targets is not None:
            # Training mode - compute losses
            return self._forward_train(images, targets)
        else:
            # Inference mode
            return self._forward_inference(images)

    def _forward_train(
        self, images: torch.Tensor, targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        # self.model is now the LWDETR nn.Module
        outputs = self.model(images)

        if self.criterion is None:
            raise RuntimeError("Criterion not initialized. Cannot compute loss.")

        # Compute losses using criterion
        loss_dict = self.criterion(outputs, targets)

        # Calculate total weighted loss
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # Prepare log dict
        log_data = {"loss": losses}
        log_data.update({f"train/{k}": v for k, v in loss_dict.items()})

        return log_data

    def _forward_inference(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for inference."""
        with torch.no_grad():
            outputs = self.model(images)
        return outputs

    def _forward_for_export(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass for ONNX export."""
        outputs = self.model(images)
        # Return in export format (boxes, scores, labels)
        return outputs

    def get_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        original_sizes: Optional[List[Tuple[int, int]]] = None,
        confidence_threshold: float = 0.0,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert model outputs to prediction format for metrics."""
        predictions = []

        # Extract predictions from outputs
        # This depends on the specific output format of RFDETR
        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")

        if pred_logits is None or pred_boxes is None:
            return predictions

        batch_size = pred_logits.shape[0]
        for b in range(batch_size):
            logits = pred_logits[b]
            boxes = pred_boxes[b]

            # Get scores and labels
            probs = logits.softmax(-1)
            scores, labels = probs[..., :-1].max(-1)

            # Filter by score threshold
            keep = scores > confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Convert boxes from cxcywh to xyxy if needed
            if boxes.numel() > 0:
                # Assuming boxes are in normalized cxcywh format
                cx, cy, w, h = boxes.unbind(-1)
                boxes = torch.stack(
                    [
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2,
                    ],
                    dim=-1,
                )

                # Rescale if original sizes provided
                if original_sizes is not None and b < len(original_sizes):
                    orig_h, orig_w = original_sizes[b]
                    boxes = boxes * torch.tensor(
                        [orig_w, orig_h, orig_w, orig_h],
                        device=boxes.device,
                        dtype=boxes.dtype,
                    )

            predictions.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
            )

        return predictions

    def export_onnx(
        self,
        output_path: str,
        input_height: int = 640,
        input_width: int = 640,
        opset_version: int = 17,
        simplify: bool = True,
        dynamic_axes: Optional[Dict] = None,
    ) -> str:
        """Export using RFDETR's built-in export method."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting RFDETR model to ONNX: {output_path}")

        # Use RFDETR's built-in export
        # Note: We need to ensure correct model usage for export
        self.rfdetr_wrapper.export(
            output_dir=str(output_path.parent),
            simplify=simplify,
            opset_version=opset_version,
        )

        return str(output_path)

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Separate parameters for encoder and decoder
        encoder_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name or "encoder" in name:
                encoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": self.learning_rate},
            {"params": encoder_params, "lr": self.lr_encoder},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler with warmup and cosine annealing
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs

            # After warmup, use cosine annealing
            max_epochs = getattr(self.trainer, "max_epochs", 100)
            if max_epochs <= self.warmup_epochs:
                return 1.0

            progress = (epoch - self.warmup_epochs) / (max_epochs - self.warmup_epochs)
            # Decay to 0.01 of the original learning rate
            return 0.01 + (1 - 0.01) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# Register model variants with Hydra
@register(group="model")
class RFDETRNanoModel(RFDETRLightningModel):
    """RFDETR Nano model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="nano", **kwargs)


@register(group="model")
class RFDETRSmallModel(RFDETRLightningModel):
    """RFDETR Small model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="small", **kwargs)


@register(group="model")
class RFDETRMediumModel(RFDETRLightningModel):
    """RFDETR Medium model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="medium", **kwargs)


@register(group="model")
class RFDETRLargeModel(RFDETRLightningModel):
    """RFDETR Large model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="large", **kwargs)
