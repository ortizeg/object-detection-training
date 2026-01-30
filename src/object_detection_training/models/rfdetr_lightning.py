"""
RFDETR model wrappers for PyTorch Lightning.

This module wraps the rfdetr models to work with the Lightning training framework.
"""

import os
from pathlib import Path

import omegaconf
import torch
from loguru import logger
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.models import build_criterion_and_postprocessors
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.utils.boxes import cxcywh_to_xyxy
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
        pretrain_weights: str | None = None,
        learning_rate: float = 2.5e-4,
        lr_encoder: float = 1.5e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,  # Match rfdetr TrainConfig default
        input_height: int = 512,
        input_width: int = 512,
        lr_vit_layer_decay: float = 0.8,
        lr_component_decay: float = 0.7,
        out_feature_indexes: list[int] = [3, 6, 9, 12],
        download_pretrained: bool = True,
        output_dir: str = "outputs",
        image_mean: list[float] = [123.675, 116.28, 103.53],
        image_std: list[float] = [58.395, 57.12, 57.375],
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
            download_pretrained: Download pretrained weights if not available.
            output_dir: Base directory for outputting results.
        """
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            input_height=input_height,
            input_width=input_width,
            output_dir=output_dir,
        )

        self.image_mean = image_mean
        self.image_std = image_std

        self.lr_encoder = lr_encoder
        self.variant = variant
        self.pretrain_weights = pretrain_weights
        self.download_pretrained = download_pretrained
        self.lr_vit_layer_decay = lr_vit_layer_decay
        self.lr_component_decay = lr_component_decay
        self.out_feature_indexes = out_feature_indexes

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

        # Ensure the model head matches the target number of classes.
        # rfdetr's Model.__init__ grows the head to match checkpoints (e.g. 91 for COCO)
        # but the native training script shrinks it back before training.
        current_out_features = self.rfdetr_wrapper.model.model.class_embed.weight.shape[
            0
        ]
        if current_out_features != num_classes + 1:
            logger.info(
                "Reinitializing detection head from {} to {} classes",
                current_out_features,
                num_classes + 1,
            )
            self.rfdetr_wrapper.model.reinitialize_detection_head(num_classes + 1)

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
        # Individual loss components for logging (scalars only)
        total_loss = outputs["loss"]

        # Log total loss
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log other components (ensure they are scalars)
        for k, v in outputs.items():
            if k not in ["loss", ""] and isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(k, v, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def forward(
        self, images: torch.Tensor, targets: list[dict] | None = None
    ) -> dict[str, torch.Tensor]:
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

        if targets is not None and not self._export_mode:
            # Training/Validation mode - compute losses
            return self._forward_train(images, targets)
        else:
            # Inference mode
            return self._forward_inference(images)

    def _forward_train(
        self, images: torch.Tensor, targets: list[dict]
    ) -> dict[str, torch.Tensor]:
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
        prefix = "train" if self.training else "val"
        log_data.update({f"{prefix}/{k}": v for k, v in loss_dict.items()})

        # Add raw model outputs for potential metric calculation (validation_step)
        log_data.update(outputs)

        return log_data

    def _forward_inference(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for inference."""
        with torch.no_grad():
            outputs = self.model(images)
        return outputs

    def _forward_for_export(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass for ONNX export."""
        outputs = self.model(images)
        # Return in export format (boxes, scores, labels)
        return outputs

    def get_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        original_sizes: list[tuple[int, int]] | None = None,
        confidence_threshold: float = 0.0,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert model outputs to prediction format for metrics.

        Uses sigmoid activation to match rfdetr package's PostProcess.
        RFDETR uses focal loss which is multi-label, so sigmoid is correct.
        """
        predictions = []

        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")

        if pred_logits is None or pred_boxes is None:
            return predictions

        batch_size = pred_logits.shape[0]
        for b in range(batch_size):
            # pred_logits has num_classes + 1 channels if explicitly reinitialized
            # or if it's following DETR convention where the last class is background.
            # We slice it to only look at foreground classes (0 to num_classes - 1).
            logits = pred_logits[b, :, : self.num_classes]
            boxes = pred_boxes[b]  # [num_queries, 4]

            # Use sigmoid activation (matches rfdetr PostProcess)
            # NOT softmax - rfdetr uses focal loss which is multi-label
            probs = logits.sigmoid()

            # Get max probability and corresponding label per query
            scores, labels = probs.max(dim=-1)

            # Filter by score threshold
            keep = scores > confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Convert boxes from cxcywh to xyxy
            if boxes.numel() > 0:
                boxes = cxcywh_to_xyxy(boxes)

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
        dynamic_axes: dict | None = None,
    ) -> str:
        """Export using RFDETR's built-in export method."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.serialization.add_safe_globals(
            [
                omegaconf.listconfig.ListConfig,
                omegaconf.dictconfig.DictConfig,
                omegaconf.base.ContainerMetadata,
                omegaconf.base.Metadata,
                omegaconf.nodes.AnyNode,
            ]
        )

        logger.info(f"Exporting RFDETR model to ONNX: {output_path}")

        # Forcing legacy ONNX exporter as the new Dynamo exporter (torch.export)
        # has issues with RF-DETR's architectural complexities (like unallocated
        # tensors)
        os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = "1"

        # Explicit monkeypatch for torch.onnx.export to enforce dynamo=False
        # this ensures that even if environment variables are ignored, the export
        # will fall back to the legacy TorchScript-based path.
        original_export = torch.onnx.export

        def monkeypatched_export(*args, **kwargs):
            kwargs["dynamo"] = False
            return original_export(*args, **kwargs)

        # Replace temporarily
        torch.onnx.export = monkeypatched_export

        # Use RFDETR's built-in export
        self.rfdetr_wrapper.export(
            output_dir=str(output_path.parent),
            simplify=simplify,
            opset_version=opset_version,
        )

        # Restore original export
        torch.onnx.export = original_export

        return str(output_path)

    def configure_optimizers(self):
        """Configure optimizers and schedulers matching rfdetr package."""
        # Parameters values are now passed via Hydra config
        lr_vit_layer_decay = self.lr_vit_layer_decay
        lr_component_decay = self.lr_component_decay
        num_layers = self.out_feature_indexes[-1] + 1

        param_dicts = []

        # We need to map our local param names to what get_dinov2_lr_decay_rate expects
        # Our model is at self.model (which is internal model)
        # Parameters look like: transformer.decoder..., etc.
        # But wait, our self.model is self.rfdetr_wrapper.model.model
        # Let's check the prefixes.

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # Default values
            lr = self.learning_rate
            weight_decay = self.weight_decay

            # Apply LLRD and selective WD if it's backbone
            # Backbone parameters in self.model start with "backbone.0"
            if n.startswith("backbone.0.encoder"):
                # Matches Backbone.get_named_param_lr_pairs logic
                layer_id = num_layers + 1
                if "embeddings" in n:
                    layer_id = 0
                elif ".layer." in n and ".residual." not in n:
                    # e.g. backbone.0.encoder.encoder.layer.5...
                    try:
                        layer_id = int(n[n.find(".layer.") :].split(".")[2]) + 1
                    except (IndexError, ValueError):
                        pass

                lr_decay = lr_vit_layer_decay ** (num_layers + 1 - layer_id)
                lr = self.lr_encoder * lr_decay * (lr_component_decay**2)

            # Selective weight decay: 0 for bias, norm, pos_embed, etc.
            if any(
                k in n
                for k in ["bias", "norm", "gamma", "pos_embed", "rel_pos", "embeddings"]
            ):
                weight_decay = 0.0

            param_dicts.append(
                {
                    "params": [p],
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        optimizer = torch.optim.AdamW(param_dicts)

        # Learning rate scheduler: Linear Warmup + Cosine Annealing
        max_epochs = self.trainer.max_epochs if self.trainer else 300
        warmup_epochs = max(
            1, self.warmup_epochs
        )  # Ensure at least 1 epoch for LinearLR
        cosine_epochs = max(1, max_epochs - warmup_epochs)

        # Linear warmup from 0.1% to 100% of base LR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Cosine annealing decay after warmup
        # eta_min set to 5% of base LR to prevent training stall at end
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.learning_rate * 0.05,
        )

        # Combine: warmup first, then cosine decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# Register model variants with Hydra
@register(name="RFDETRNano")
class RFDETRNanoModel(RFDETRLightningModel):
    """RFDETR Nano model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="nano", **kwargs)


@register(name="RFDETRSmall")
class RFDETRSmallModel(RFDETRLightningModel):
    """RFDETR Small model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="small", **kwargs)


@register(name="RFDETRMedium")
class RFDETRMediumModel(RFDETRLightningModel):
    """RFDETR Medium model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="medium", **kwargs)


@register(name="RFDETRLarge")
class RFDETRLargeModel(RFDETRLightningModel):
    """RFDETR Large model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="large", **kwargs)
