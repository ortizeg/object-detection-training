"""
YOLOX model wrapper for PyTorch Lightning.

This module provides Lightning-compatible wrappers for YOLOX models.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.models.yolox.yolox import YOLOX
from object_detection_training.utils.boxes import cxcywh_to_xyxy
from object_detection_training.utils.download import download_checkpoint
from object_detection_training.utils.hydra import register

# YOLOX checkpoint URLs from official releases
# Model configurations and URLs are now configuration-driven.


# Model configurations and URLs are now configuration-driven.


@register(name="yolox")
class YOLOXLightningModel(BaseDetectionModel):
    """
    PyTorch Lightning wrapper for YOLOX models.

    Wraps YOLOX with Lightning training framework.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        num_classes: int = 80,
        pretrain_weights: Optional[str] = None,
        weights_url: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 5,
        download_pretrained: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.9998,
        input_height: int = 640,
        input_width: int = 640,
        output_dir: str = "outputs",
        image_mean: List[float] = [0.0, 0.0, 0.0],
        image_std: List[float] = [1.0, 1.0, 1.0],
        **kwargs: Any,
    ):
        """
        Initialize YOLOX Lightning model.

        Args:
            model: The YOLOX model instance.
            backbone: The YOLOX backbone instance (optional if model provided).
            head: The YOLOX head instance (optional if model provided).
            num_classes: Number of detection classes.
            pretrain_weights: Path to pretrained weights file.
            weights_url: URL to download pretrained weights.
            learning_rate: Base learning rate.
            weight_decay: Weight decay.
            warmup_epochs: Number of warmup epochs.
            download_pretrained: Download pretrained weights if not available.
            input_height: Input image height.
            input_width: Input image width.
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
        self.save_hyperparameters(ignore=["model", "backbone", "head", "kwargs"])

        self.image_mean = image_mean
        self.image_std = image_std

        self.pretrain_weights = pretrain_weights
        self.weights_url = weights_url
        self.download_pretrained = download_pretrained
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.input_height = input_height
        self.input_width = input_width

        if download_pretrained and not pretrain_weights:
            if not weights_url:
                logger.warning(
                    "download_pretrained=True but no weights_url specified. Skipping download."
                )
            else:
                filename = Path(weights_url).name
                dest = Path(output_dir) / "checkpoints" / filename
                try:
                    self.pretrain_weights = str(download_checkpoint(weights_url, dest))
                except Exception as e:
                    logger.error(f"Failed to download pretrained weights: {e}")

        # Build YOLOX model
        logger.info("Initializing YOLOX model")

        if model is not None:
            self.model = model
        else:
            logger.info("Initializing YOLOX model with provided backbone and head")
            self.model = YOLOX(backbone=backbone, head=head)

        # Initialize BatchNorm with official YOLOX settings
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        # Initialize biases BEFORE loading weights
        self.model.head.initialize_biases(prior_prob=1e-2)

        # Load pretrained weights (after download if applicable)
        if self.pretrain_weights:
            self._load_weights(self.pretrain_weights)

        # Reinitialize classification biases AFTER loading weights
        # This is crucial when num_classes differs from pretrained (cls_preds skipped)
        self._reinitialize_cls_biases()

        self.save_hyperparameters(ignore=["model", "backbone", "head"])

    def _reinitialize_cls_biases(self):
        """Reinitialize classification prediction biases after weight loading.

        When loading pretrained weights with different num_classes, the cls_preds
        weights are skipped, leaving random initialization. We reinitialize the
        biases with proper prior probability for stable training.
        """
        prior_prob = 1e-2
        bias_init = -math.log((1 - prior_prob) / prior_prob)

        for conv in self.model.head.cls_preds:
            nn.init.constant_(conv.bias, bias_init)

        logger.info(f"Reinitialized cls_preds biases with prior_prob={prior_prob}")

    def _load_weights(self, checkpoint_path: str):
        """Load weights from checkpoint."""
        logger.info(f"Loading weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            # Create new state dict with mapping
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}

            # Match parameters
            matched = []
            unmatched = []
            class_mismatch = []

            for k, v in state_dict.items():
                if k in model_state_dict:
                    # Check for class dimension mismatch in head
                    if "cls_preds" in k and v.shape[0] != self.num_classes:
                        class_mismatch.append(k)
                        continue

                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                        matched.append(k)
                    else:
                        unmatched.append(
                            f"{k} (shape mismatch: {v.shape} vs "
                            f"{model_state_dict[k].shape})"
                        )
                else:
                    unmatched.append(k)

            # Log summary
            logger.info("Checkpoint match summary:")
            logger.info(f"  Matched: {len(matched)} / {len(model_state_dict)}")
            if class_mismatch:
                logger.info(f"  Class mismatch (skipped): {len(class_mismatch)}")
            if unmatched:
                logger.debug(f"  Unmatched: {unmatched[:10]}...")

            self.model.load_state_dict(filtered_state_dict, strict=False)
            logger.info("Weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        # self(images, targets) returns dict from YOLOX.forward
        outputs = self(images, targets)

        loss = outputs["total_loss"]
        iou_loss = outputs["iou_loss"]
        obj_loss = outputs["conf_loss"]
        cls_loss = outputs["cls_loss"]
        l1_loss = outputs["l1_loss"]
        num_fg = outputs["num_fg"]

        # Log losses
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/iou_loss", iou_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/obj_loss", obj_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log("train/l1_loss", l1_loss, on_step=True, on_epoch=True, prog_bar=False)

        # Log num_fg
        if isinstance(num_fg, torch.Tensor):
            num_fg_val = num_fg.float().mean()
        else:
            num_fg_val = float(num_fg)
        self.log("train/num_fg", num_fg_val, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def forward(
        self, images: torch.Tensor, targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Handle NestedTensor from rfdetr collation
        if hasattr(images, "tensors"):
            images = images.tensors

        if self._export_mode:
            return self.model(images)

        if targets is not None:
            # Un-normalize targets for YOLOX loss calculation
            # targets is a list of dicts with 'boxes' and 'labels'
            # boxes are in [0, 1] cxcywh format
            # Use actual image shape for accurate un-normalization
            # (esp. for multi-scale)
            img_h, img_w = images.shape[2:]
            unnormalized_targets = []
            for t in targets:
                new_t = t.copy()
                if "boxes" in new_t and new_t["boxes"].numel() > 0:
                    boxes = new_t["boxes"].clone()
                    boxes[:, [0, 2]] *= img_w
                    boxes[:, [1, 3]] *= img_h
                    new_t["boxes"] = boxes
                unnormalized_targets.append(new_t)

            outputs = self.model(images, unnormalized_targets)
            # Add images shape to outputs for post-processing resolution retrieval
            if "image_shape" not in outputs:
                outputs["image_shape"] = images.shape[2:]
            return outputs
        else:
            with torch.no_grad():
                outputs = self.model(images)
            # Add images shape to outputs for post-processing resolution retrieval
            return {"predictions": outputs, "image_shape": images.shape[2:]}

    def get_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        original_sizes: Optional[List[Tuple[int, int]]] = None,
        confidence_threshold: float = 0.1,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert model outputs to prediction format."""
        predictions = []

        pred = outputs.get("predictions")
        if pred is None:
            return predictions

        # YOLOX outputs: [batch, num_anchors, 5 + num_classes]
        # Format: [x, y, w, h, obj_conf, cls_conf...]
        batch_size = pred.shape[0]

        for b in range(batch_size):
            box_preds = pred[b]  # [num_anchors, 5 + num_classes]

            # Get scores and labels first
            obj_conf = box_preds[:, 4]
            cls_conf = box_preds[:, 5:]
            scores, labels = cls_conf.max(dim=1)
            scores = scores * obj_conf

            # Filter by threshold early to speed up NMS
            keep = scores > confidence_threshold
            box_preds = box_preds[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(scores) == 0:
                predictions.append(
                    {
                        "boxes": torch.zeros((0, 4), device=scores.device),
                        "scores": scores,
                        "labels": labels,
                    }
                )
                continue

            # Get boxes (cxcywh -> xyxy)
            # Important: unbind to avoid in-place corruption
            boxes = cxcywh_to_xyxy(box_preds[:, :4])

            # NMS (Required for YOLOX post-processing)
            from torchvision.ops import batched_nms

            keep_indices = batched_nms(boxes, scores, labels, iou_threshold=0.45)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

            # Normalize boxes to [0, 1] or scale to original_sizes
            # Use actual runtime resolution (image_shape) for normalization
            img_h, img_w = outputs.get(
                "image_shape", (self.input_height, self.input_width)
            )
            if original_sizes is not None and len(original_sizes) > b:
                orig_h, orig_w = original_sizes[b]
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] / img_w) * orig_w
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] / img_h) * orig_h
            else:
                # Normalize boxes to [0, 1] relative to current resolution
                boxes[:, [0, 2]] /= img_w
                boxes[:, [1, 3]] /= img_h

            predictions.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
            )

        return predictions

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Optimizer and scheduler configuration.
        """
        # Separate parameters into groups
        # (apply weight decay to weights, not biases/norm)
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "bn" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        # Configurable optimizer parameters
        momentum = 0.9
        nesterov = True

        optimizer = torch.optim.SGD(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            momentum=momentum,
            nesterov=nesterov,
        )

        # Scheduler with warmup + cosine annealing
        if self.trainer and hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback for testing or when trainer is not yet fully initialized
            total_steps = 10000

        # Avoid 0 total steps
        total_steps = max(1, total_steps)

        # Warmup scheduler
        warmup_epochs = self.warmup_epochs
        max_epochs = self.trainer.max_epochs if self.trainer else 100

        # Calculate warmup steps
        warmup_steps = int(total_steps * (warmup_epochs / max(1, max_epochs)))
        # Ensure warmup doesn't take more than half of training
        warmup_steps = min(warmup_steps, total_steps // 2)
        warmup_steps = max(1, warmup_steps)

        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps
        )

        # Main scheduler (Cosine Annealing) - starts after warmup
        main_steps = max(1, total_steps - warmup_steps)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_steps, eta_min=self.learning_rate * 0.05
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
