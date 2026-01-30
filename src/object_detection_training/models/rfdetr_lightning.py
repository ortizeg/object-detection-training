"""
RFDETR model wrappers for PyTorch Lightning.

This module wraps the rfdetr models to work with the Lightning training framework.
Uses local model architecture code instead of the rfdetr PyPI package.
All architecture parameters are configured via Hydra YAML configs.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import omegaconf
import torch
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.models.rfdetr.lwdetr import (
    build_criterion_and_postprocessors,
)
from object_detection_training.models.rfdetr.model_factory import (
    HOSTED_MODELS,
    Model,
)
from object_detection_training.utils.boxes import cxcywh_to_xyxy
from object_detection_training.utils.hydra import register


def _download_checkpoint(url: str, destination: Path) -> Path:
    """Download a checkpoint file if it doesn't exist."""
    import urllib.request

    destination = Path(destination)
    if destination.exists():
        logger.info(f"Checkpoint already exists: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint from {url}")

    try:
        urllib.request.urlretrieve(url, destination)  # noqa: S310
        logger.info(f"Checkpoint downloaded to {destination}")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise

    return destination


class RFDETRLightningModel(BaseDetectionModel):
    """
    PyTorch Lightning wrapper for RFDETR models.

    All architecture parameters are passed directly from Hydra YAML configs.
    No Pydantic config classes are used at runtime.
    """

    def __init__(
        self,
        # --- Base training params (from base.yaml) ---
        num_classes: int = 80,
        learning_rate: float = 2.5e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,
        input_height: int = 512,
        input_width: int = 512,
        output_dir: str = "outputs",
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        # --- RFDETR training params (from rfdetr_base.yaml) ---
        lr_encoder: float = 1.5e-4,
        lr_vit_layer_decay: float = 0.8,
        lr_component_decay: float = 0.7,
        download_pretrained: bool = True,
        pretrain_weights: str | None = None,
        # --- Scheduler params ---
        warmup_start_factor: float = 1e-3,
        cosine_eta_min_factor: float = 0.05,
        # --- Architecture params (from rfdetr_base.yaml + variant overrides) ---
        encoder: str = "dinov2_windowed_small",
        hidden_dim: int = 256,
        dec_layers: int = 3,
        patch_size: int = 16,
        num_windows: int = 2,
        sa_nheads: int = 8,
        ca_nheads: int = 16,
        dec_n_points: int = 2,
        two_stage: bool = True,
        bbox_reparam: bool = True,
        lite_refpoint_refine: bool = True,
        layer_norm: bool = True,
        amp: bool = True,
        group_detr: int = 13,
        gradient_checkpointing: bool = False,
        positional_encoding_size: int = 32,
        ia_bce_loss: bool = True,
        cls_loss_coef: float = 1.0,
        segmentation_head: bool = False,
        mask_downsample_ratio: int = 4,
        projector_scale: list[str] | None = None,
        num_queries: int = 300,
        num_select: int = 300,
        resolution: int | None = None,
        out_feature_indexes: list[int] | None = None,
        dim_feedforward: int = 2048,
        decoder_norm: str = "LN",
        vit_encoder_num_layers: int = 12,
        position_embedding: str = "sine",
        # --- Loss / Matcher params ---
        set_cost_class: int = 2,
        set_cost_bbox: int = 5,
        set_cost_giou: int = 2,
        bbox_loss_coef: int = 5,
        giou_loss_coef: int = 2,
        focal_alpha: float = 0.25,
        aux_loss: bool = True,
        # --- Checkpoint identification ---
        checkpoint_name: str | None = None,
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            input_height=input_height,
            input_width=input_width,
            output_dir=output_dir,
        )

        self.image_mean = (
            image_mean if image_mean is not None else [123.675, 116.28, 103.53]
        )
        self.image_std = image_std if image_std is not None else [58.395, 57.12, 57.375]

        # Training params
        self.lr_encoder = lr_encoder
        self.lr_vit_layer_decay = lr_vit_layer_decay
        self.lr_component_decay = lr_component_decay
        self.download_pretrained = download_pretrained
        self.pretrain_weights = pretrain_weights

        # Scheduler params (previously hardcoded)
        self.warmup_start_factor = warmup_start_factor
        self.cosine_eta_min_factor = cosine_eta_min_factor

        # Resolve defaults
        resolved_out_feature_indexes = (
            list(out_feature_indexes)
            if out_feature_indexes is not None
            else [3, 6, 9, 12]
        )
        self.out_feature_indexes = resolved_out_feature_indexes

        resolved_resolution = resolution if resolution is not None else input_height
        resolved_projector_scale = (
            list(projector_scale) if projector_scale is not None else ["P4"]
        )

        # Handle pretrained weights download
        if pretrain_weights is None and download_pretrained and checkpoint_name:
            cache_dir = Path.home() / ".cache" / "rfdetr"
            checkpoint_path = cache_dir / checkpoint_name
            if not checkpoint_path.exists() and checkpoint_name in HOSTED_MODELS:
                url = HOSTED_MODELS[checkpoint_name]
                _download_checkpoint(url, checkpoint_path)
            if checkpoint_path.exists():
                pretrain_weights = str(checkpoint_path)

        logger.info(
            "Initializing RFDETR model (encoder={}, hidden_dim={}, dec_layers={}, "
            "resolution={})",
            encoder,
            hidden_dim,
            dec_layers,
            resolved_resolution,
        )

        # Auto-detect device for Model construction
        # (Model.__init__ moves weights to device; Lightning manages placement after)
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        # Build Model directly from params (no Pydantic config intermediary)
        model_params = {
            "device": _device,
            "encoder": encoder,
            "hidden_dim": hidden_dim,
            "dec_layers": dec_layers,
            "patch_size": patch_size,
            "num_windows": num_windows,
            "sa_nheads": sa_nheads,
            "ca_nheads": ca_nheads,
            "dec_n_points": dec_n_points,
            "two_stage": two_stage,
            "bbox_reparam": bbox_reparam,
            "lite_refpoint_refine": lite_refpoint_refine,
            "layer_norm": layer_norm,
            "amp": amp,
            "group_detr": group_detr,
            "gradient_checkpointing": gradient_checkpointing,
            "positional_encoding_size": positional_encoding_size,
            "ia_bce_loss": ia_bce_loss,
            "cls_loss_coef": cls_loss_coef,
            "segmentation_head": segmentation_head,
            "mask_downsample_ratio": mask_downsample_ratio,
            "projector_scale": resolved_projector_scale,
            "num_queries": num_queries,
            "num_select": num_select,
            "resolution": resolved_resolution,
            "out_feature_indexes": resolved_out_feature_indexes,
            "dim_feedforward": dim_feedforward,
            "decoder_norm": decoder_norm,
            "vit_encoder_num_layers": vit_encoder_num_layers,
            "position_embedding": position_embedding,
            "set_cost_class": set_cost_class,
            "set_cost_bbox": set_cost_bbox,
            "set_cost_giou": set_cost_giou,
            "bbox_loss_coef": bbox_loss_coef,
            "giou_loss_coef": giou_loss_coef,
            "focal_alpha": focal_alpha,
            "aux_loss": aux_loss,
            "num_classes": num_classes,
            "pretrain_weights": pretrain_weights,
        }
        self._rfdetr_model = Model(**model_params)

        # Ensure the model head matches the target number of classes.
        current_out_features = self._rfdetr_model.model.class_embed.weight.shape[0]
        if current_out_features != num_classes + 1:
            logger.info(
                "Reinitializing detection head from {} to {} classes",
                current_out_features,
                num_classes + 1,
            )
            self._rfdetr_model.reinitialize_detection_head(num_classes + 1)

        # Register the actual nn.Module so Lightning/Optimizer can see parameters
        self.model = self._rfdetr_model.model

        # Store internal model reference for export if needed
        self._internal_model = self.model

        # Build criterion for training
        logger.info("Building criterion for training...")
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(
            self._rfdetr_model.args
        )

        self.save_hyperparameters()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Custom training step to handle RFDETR weighted losses."""
        images, targets = batch

        # self(images, targets) returns dict with 'loss' and 'train/...' keys
        outputs = self(images, targets)
        # Individual loss components for logging (scalars only)
        total_loss: torch.Tensor = outputs["loss"]

        # Log total loss
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log other components (ensure they are scalars)
        for k, v in outputs.items():
            if k not in ["loss", ""] and isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(k, v, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def forward(
        self, images: torch.Tensor, targets: list[dict[str, Any]] | None = None
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
        self, images: torch.Tensor, targets: list[dict[str, Any]]
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
            loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
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
            outputs: dict[str, torch.Tensor] = self.model(images)
        return outputs

    def _forward_for_export(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for ONNX export."""
        outputs: dict[str, torch.Tensor] = self.model(images)
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
        predictions: list[dict[str, torch.Tensor]] = []

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
        dynamic_axes: dict[str, Any] | None = None,
    ) -> str:
        """Export model to ONNX format."""
        from copy import deepcopy

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.serialization.add_safe_globals(
            [
                omegaconf.listconfig.ListConfig,
                omegaconf.dictconfig.DictConfig,
                omegaconf.base.ContainerMetadata,
                omegaconf.base.Metadata,
                omegaconf.nodes.AnyNode,
            ]
        )

        logger.info(f"Exporting RFDETR model to ONNX: {out_path}")

        # Forcing legacy ONNX exporter as the new Dynamo exporter (torch.export)
        # has issues with RF-DETR's architectural complexities (like unallocated
        # tensors)
        os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = "1"

        # Explicit monkeypatch for torch.onnx.export to enforce dynamo=False
        # this ensures that even if environment variables are ignored, the export
        # will fall back to the legacy TorchScript-based path.
        original_export = torch.onnx.export

        def monkeypatched_export(*args: Any, **kwargs: Any) -> Any:
            kwargs["dynamo"] = False
            return original_export(*args, **kwargs)

        # Replace temporarily
        torch.onnx.export = monkeypatched_export

        device = self._rfdetr_model.device
        model = deepcopy(self.model.cpu())
        model.to(device)
        model.eval()
        model.export()

        resolution = self._rfdetr_model.resolution
        dummy_input = torch.randn(1, 3, resolution, resolution, device=device)

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(out_path),
                input_names=["input"],
                output_names=["dets", "labels"],
                opset_version=opset_version,
            )

        # Restore original export
        torch.onnx.export = original_export
        self.model.to(device)

        return str(out_path)

    def configure_optimizers(self) -> Any:
        """Configure optimizers and schedulers matching rfdetr package."""
        lr_vit_layer_decay = self.lr_vit_layer_decay
        lr_component_decay = self.lr_component_decay
        num_layers = self.out_feature_indexes[-1] + 1

        param_dicts = []

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
                    with contextlib.suppress(IndexError, ValueError):
                        layer_id = int(n[n.find(".layer.") :].split(".")[2]) + 1

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
        max_epochs: int = (self.trainer.max_epochs or 300) if self.trainer else 300
        warmup_epochs = max(
            1, self.warmup_epochs
        )  # Ensure at least 1 epoch for LinearLR
        cosine_epochs = max(1, max_epochs - warmup_epochs)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.learning_rate * self.cosine_eta_min_factor,
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

    _checkpoint_name = "rf-detr-nano.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="RFDETRSmall")
class RFDETRSmallModel(RFDETRLightningModel):
    """RFDETR Small model for Hydra instantiation."""

    _checkpoint_name = "rf-detr-small.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="RFDETRMedium")
class RFDETRMediumModel(RFDETRLightningModel):
    """RFDETR Medium model for Hydra instantiation."""

    _checkpoint_name = "rf-detr-medium.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="RFDETRLarge")
class RFDETRLargeModel(RFDETRLightningModel):
    """RFDETR Large model for Hydra instantiation."""

    _checkpoint_name = "rf-detr-large.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)
