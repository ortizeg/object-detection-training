"""
RFDETR model wrappers for PyTorch Lightning.

This module wraps the rfdetr models to work with the Lightning training framework.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import omegaconf
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.models.rfdetr import (
    build_criterion_and_postprocessors,
)
from object_detection_training.utils.boxes import cxcywh_to_xyxy
from object_detection_training.utils.download import download_checkpoint
from object_detection_training.utils.hydra import register

# Add argparse.Namespace to safe globals for torch.load in PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

# Model weight handling logic is now configuration-driven.


# Model weight handling logic is now configuration-driven.


@register(name="rfdetr")
class RFDETRLightningModel(BaseDetectionModel):
    """
    PyTorch Lightning wrapper for RFDETR models.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        postprocessors: Optional[nn.Module] = None,
        num_classes: int = 80,
        pretrain_weights: Optional[str] = None,
        weights_url: Optional[str] = None,
        download_pretrained: bool = False,
        learning_rate: float = 2.5e-4,
        lr_encoder: float = 1.5e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,
        input_height: int = 512,
        input_width: int = 512,
        lr_vit_layer_decay: float = 0.8,
        lr_component_decay: float = 0.7,
        out_feature_indexes: List[int] = [3, 6, 9, 12],
        output_dir: str = "outputs",
        image_mean: List[float] = [123.675, 116.28, 103.53],
        image_std: List[float] = [58.395, 57.12, 57.375],
        **kwargs: Any,
    ):
        """
        Initialize RFDETR Lightning model.

        Args:
            model: The RFDETR model instance.
            criterion: The loss criterion.
            postprocessors: Post-processors for inference.
            num_classes: Number of detection classes.
            pretrain_weights: Path to pretrained weights file.
            input_height: Input image height.
            input_width: Input image width.
            learning_rate: Base learning rate.
            lr_encoder: Learning rate for encoder.
            weight_decay: Weight decay.
            warmup_epochs: Number of warmup epochs.
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
        self.save_hyperparameters(
            ignore=["model", "criterion", "postprocessors", "kwargs"]
        )

        self.download_pretrained = download_pretrained
        self.pretrain_weights = pretrain_weights
        self.weights_url = weights_url

        if download_pretrained and not pretrain_weights:
            if not weights_url:
                logger.warning(
                    "download_pretrained=True but no weights_url specified. Skipping download."
                )
            else:
                # Use a generic name if variant is not available
                filename = Path(weights_url).name
                dest = Path(output_dir) / "checkpoints" / filename
                try:
                    self.pretrain_weights = str(download_checkpoint(weights_url, dest))
                except Exception as e:
                    logger.error(f"Failed to download pretrained weights: {e}")

        self.image_mean = image_mean
        self.image_std = image_std

        self.lr_encoder = lr_encoder
        self.lr_vit_layer_decay = lr_vit_layer_decay
        self.lr_component_decay = lr_component_decay
        self.out_feature_indexes = out_feature_indexes

        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors

        # Handle weight loading if provided
        if self.pretrain_weights:
            logger.info(f"Loading weights from {self.pretrain_weights}")
            checkpoint = torch.load(self.pretrain_weights, map_location="cpu")
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            # Robust loading: filter out parameters with mismatched shapes
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        logger.warning(
                            f"Skipping parameter {k} due to shape mismatch: "
                            f"checkpoint {v.shape} vs model {model_state_dict[k].shape}"
                        )
                else:
                    logger.debug(f"Skipping parameter {k} not in model")

            self.model.load_state_dict(filtered_state_dict, strict=False)

        # Ensure the model head matches the target number of classes.
        current_out_features = self.model.class_embed.weight.shape[0]
        if current_out_features != num_classes:
            logger.info(
                "Reinitializing detection head from {} to {} classes",
                current_out_features,
                num_classes,
            )
            # Check if model has reinitialize_detection_head method
            if hasattr(self.model, "reinitialize_detection_head"):
                self.model.reinitialize_detection_head(num_classes)
            else:
                # Manual reinitialization if needed
                hidden_dim = self.model.class_embed.in_features
                self.model.class_embed = nn.Linear(hidden_dim, num_classes)
                nn.init.constant_(self.model.class_embed.bias, -4.6)  # focal loss init

        # If criterion is still None, try building it if we have args (unlikely in direct pass)
        if self.criterion is None and hasattr(self, "args"):
            logger.info("Building criterion for training...")
            self.criterion, self.postprocessors = build_criterion_and_postprocessors(
                self.args
            )

        self.save_hyperparameters(ignore=["model", "criterion", "postprocessors"])

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

        if targets is not None and not self._export_mode:
            # Training/Validation mode - compute losses
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
        prefix = "train" if self.training else "val"
        log_data.update({f"{prefix}/{k}": v for k, v in loss_dict.items()})

        # Add raw model outputs for potential metric calculation (validation_step)
        log_data.update(outputs)

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
        dynamic_axes: Optional[Dict] = None,
    ) -> str:
        """Export using custom export logic."""

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

        # Forcing legacy ONNX exporter
        os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = "1"

        # Explicit monkeypatch for torch.onnx.export to enforce dynamo=False
        original_export = torch.onnx.export

        def monkeypatched_export(*args, **kwargs):
            kwargs["dynamo"] = False
            return original_export(*args, **kwargs)

        # Replace temporarily
        torch.onnx.export = monkeypatched_export

        # For RF-DETR export, we use the model's export method if it exists
        if hasattr(self.model, "export"):
            self.model.export()

        # Dummy input
        dummy_input = torch.randn(1, 3, input_height, input_width).to(self.device)

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["pred_boxes", "pred_logits"],
            dynamic_axes=dynamic_axes or {"input": {0: "batch"}},
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
