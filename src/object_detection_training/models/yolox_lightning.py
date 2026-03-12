"""
YOLOX model wrapper for PyTorch Lightning.

This module provides Lightning-compatible wrappers for YOLOX models.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.models.yolox import YOLOPAFPN, YOLOX, YOLOXHead
from object_detection_training.types import (
    DetectionBatch,
    DetectionTarget,
    ModelOutputs,
    OptimizerConfig,
)
from object_detection_training.utils.boxes import cxcywh_to_xyxy
from object_detection_training.utils.hydra import register

# YOLOX checkpoint URLs from official releases, keyed by checkpoint filename
YOLOX_CHECKPOINT_URLS = {
    "yolox_nano.pth": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_nano.pth"
    ),
    "yolox_tiny.pth": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_tiny.pth"
    ),
    "yolox_s.pth": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_s.pth"
    ),
    "yolox_m.pth": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_m.pth"
    ),
    "yolox_l.pth": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_l.pth"
    ),
    "yolox_x.pth": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_x.pth"
    ),
}

# Original YOLOX model configurations (kept as documentation reference).
# Architecture params are now passed explicitly from Hydra YAML configs.
# YOLOX_CONFIGS = {
#     "nano": {"depth": 0.33, "width": 0.25, "depthwise": True},
#     "tiny": {"depth": 0.33, "width": 0.375, "depthwise": False},
#     "s":    {"depth": 0.33, "width": 0.50,  "depthwise": False},
#     "m":    {"depth": 0.67, "width": 0.75,  "depthwise": False},
#     "l":    {"depth": 1.0,  "width": 1.0,   "depthwise": False},
#     "x":    {"depth": 1.33, "width": 1.25,  "depthwise": False},
# }


def download_checkpoint(url: str, destination: Path) -> Path:
    """Download a checkpoint file if it doesn't exist.

    Uses torch.hub.download_url_to_file which handles redirects,
    shows progress, and is proven to work across environments
    (local, Docker, GCP).

    Args:
        url: URL to download from.
        destination: Local path to save the file.

    Raises:
        RuntimeError: If download fails for any reason.
    """
    destination = Path(destination)
    if destination.exists():
        logger.info(f"Checkpoint already cached: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint from {url}")

    try:
        torch.hub.download_url_to_file(url, str(destination), progress=True)
    except Exception as e:
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        raise RuntimeError(
            f"Failed to download pretrained weights from {url}: {e}"
        ) from e

    if not destination.exists():
        raise RuntimeError(
            f"Download appeared to succeed but file not found at {destination}"
        )

    size_mb = destination.stat().st_size / (1024 * 1024)
    logger.info(f"Checkpoint downloaded to {destination} ({size_mb:.1f} MB)")
    return destination


class YOLOXLightningModel(BaseDetectionModel):
    """
    PyTorch Lightning wrapper for YOLOX models.

    Wraps YOLOX with Lightning training framework.
    """

    def __init__(
        self,
        num_classes: int = 80,
        pretrain_weights: str | None = None,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 5,
        download_pretrained: bool = True,
        input_height: int = 640,
        input_width: int = 640,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        output_dir: str = "outputs",
        freeze_backbone_epochs: int = 0,
        no_aug_epochs: int = 15,
        l1_loss_epoch: int = 0,
        iou_loss_type: str = "iou",
        depth: float = 0.33,
        width: float = 0.50,
        depthwise: bool = False,
        in_channels: list[int] | None = None,
        checkpoint_name: str = "yolox_s.pth",
    ):
        """
        Initialize YOLOX Lightning model.

        Args:
            num_classes: Number of detection classes.
            pretrain_weights: Path to pretrained weights file.
            learning_rate: Base learning rate.
            weight_decay: Weight decay.
            warmup_epochs: Number of warmup epochs.
            download_pretrained: Download pretrained weights if not available.
            input_height: Input image height.
            input_width: Input image width.
            output_dir: Base directory for outputting results.
            freeze_backbone_epochs: Freeze backbone for this many initial epochs
                during fine-tuning. 0 disables freezing.
            no_aug_epochs: Number of final epochs without mosaic/mixup augmentation.
                At epoch (max_epochs - no_aug_epochs), mosaic/mixup are disabled
                and L1 regression loss is enabled. Official YOLOX uses 15.
                Set to 0 to disable.
            iou_loss_type: IoU loss variant for box regression ('iou' or 'giou').
                GIoU provides better gradients for non-overlapping boxes.
            depth: Network depth multiplier.
            width: Network width multiplier.
            depthwise: Use depthwise-separable convolutions.
            in_channels: Feature map channel sizes for FPN/head.
            checkpoint_name: Filename for pretrained checkpoint lookup and caching.
        """
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            input_height=input_height,
            input_width=input_width,
            image_mean=image_mean,
            image_std=image_std,
            output_dir=output_dir,
        )

        self.pretrain_weights = pretrain_weights
        self.download_pretrained = download_pretrained
        self.input_height = input_height
        self.input_width = input_width
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.no_aug_epochs = no_aug_epochs
        self.l1_loss_epoch = l1_loss_epoch  # backward compat fallback
        self.checkpoint_name = checkpoint_name

        if in_channels is None:
            in_channels = [256, 512, 1024]

        # Build YOLOX model
        logger.info(
            f"Initializing YOLOX model "
            f"(depth={depth}, width={width}, depthwise={depthwise})"
        )

        backbone = YOLOPAFPN(  # type: ignore[no-untyped-call]
            depth=depth,
            width=width,
            in_channels=in_channels,
            depthwise=depthwise,
        )
        head = YOLOXHead(  # type: ignore[no-untyped-call]
            num_classes=num_classes,
            width=width,
            in_channels=in_channels,
            depthwise=depthwise,
        )

        self.model = YOLOX(backbone=backbone, head=head)  # type: ignore[no-untyped-call]

        # Override IoU loss type if requested
        if iou_loss_type != "iou":
            from object_detection_training.models.yolox.yolo_head import IOUloss

            self.model.head.iou_loss = IOUloss(
                reduction="none", loss_type=iou_loss_type
            )
            logger.info(f"Using {iou_loss_type} loss for box regression")

        # Initialize BatchNorm with official YOLOX settings
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        # Initialize biases BEFORE loading weights
        self.model.head.initialize_biases(prior_prob=1e-2)

        # Load pretrained weights
        if pretrain_weights:
            self._load_weights(pretrain_weights)
        elif download_pretrained:
            self._download_and_load_weights()

        # Reinitialize classification biases AFTER loading weights
        # This is crucial when num_classes differs from pretrained (cls_preds skipped)
        self._reinitialize_cls_biases()

        # Freeze backbone for initial fine-tuning epochs if requested
        if self.freeze_backbone_epochs > 0:
            self._freeze_backbone()

        self.save_hyperparameters()

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters (PAFPN backbone) for fine-tuning.

        When fine-tuning from pretrained weights, freezing the backbone initially
        allows the head to adapt to new classes before backbone features are modified.
        """
        for param in self.model.backbone.backbone.parameters():
            param.requires_grad = False
        n_frozen = sum(
            1 for p in self.model.backbone.backbone.parameters() if not p.requires_grad
        )
        logger.info(
            f"Froze {n_frozen} backbone parameters "
            f"for {self.freeze_backbone_epochs} epochs"
        )

    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.model.backbone.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze backbone parameters")

    def on_train_epoch_start(self) -> None:
        """Handle epoch-based training schedule changes.

        Implements official YOLOX no-augmentation refinement period:
        at epoch (max_epochs - no_aug_epochs), mosaic/mixup are disabled
        and L1 regression loss is enabled for final convergence.
        """
        if (
            self.freeze_backbone_epochs > 0
            and self.current_epoch == self.freeze_backbone_epochs
        ):
            self._unfreeze_backbone()

        # No-augmentation period: disable mosaic/mixup + enable L1 loss
        if self.no_aug_epochs > 0 and self.trainer:
            max_epochs = self.trainer.max_epochs or 300
            no_aug_start = max_epochs - self.no_aug_epochs
            if self.current_epoch == no_aug_start:
                # Enable L1 loss for refined box regression
                if not self.model.head.use_l1:
                    self.model.head.use_l1 = True
                    logger.info(
                        f"No-aug period: enabled L1 loss at epoch {self.current_epoch}"
                    )

                # Disable mosaic/mixup on the training dataloader's dataset
                train_dl = self.trainer.train_dataloader
                if train_dl is not None:
                    dataset = train_dl.dataset
                    if hasattr(dataset, "enabled"):
                        dataset.enabled = False
                        logger.info(
                            f"No-aug period: disabled mosaic/mixup "
                            f"at epoch {self.current_epoch}"
                        )

        # Legacy fallback: explicit l1_loss_epoch (for backward compat)
        elif (
            self.l1_loss_epoch > 0
            and self.current_epoch == self.l1_loss_epoch
            and not self.model.head.use_l1
        ):
            self.model.head.use_l1 = True
            logger.info(f"Enabled L1 regression loss at epoch {self.current_epoch}")

    def _reinitialize_cls_biases(self) -> None:
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

    def _download_and_load_weights(self) -> None:
        """Download and load pretrained weights.

        Raises:
            RuntimeError: If the checkpoint_name has no URL, or if
                download/loading fails. Training must not proceed
                without pretrained weights when they were requested.
        """
        if self.checkpoint_name not in YOLOX_CHECKPOINT_URLS:
            raise RuntimeError(
                f"download_pretrained=True but no checkpoint URL for "
                f"checkpoint_name '{self.checkpoint_name}'. Available: "
                f"{list(YOLOX_CHECKPOINT_URLS.keys())}"
            )

        cache_dir = Path.home() / ".cache" / "yolox"
        checkpoint_path = cache_dir / self.checkpoint_name

        url = YOLOX_CHECKPOINT_URLS[self.checkpoint_name]
        download_checkpoint(url, checkpoint_path)
        self._load_weights(str(checkpoint_path))

    def _load_weights(self, checkpoint_path: str) -> None:
        """Load weights from checkpoint.

        Raises:
            RuntimeError: If the checkpoint cannot be loaded or contains
                no matching parameters.
        """
        logger.info(f"Loading weights from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

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
        logger.info(f"Checkpoint match summary for {self.checkpoint_name}:")
        logger.info(f"  Matched: {len(matched)} / {len(model_state_dict)}")
        if class_mismatch:
            logger.info(f"  Class mismatch (skipped): {len(class_mismatch)}")
        if unmatched:
            logger.debug(f"  Unmatched: {unmatched[:10]}...")

        if not filtered_state_dict:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_path} has 0 matching parameters. "
                f"Checkpoint keys: {len(state_dict)}, "
                f"Model keys: {len(model_state_dict)}"
            )

        self.model.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Weights loaded successfully")

    def training_step(self, batch: DetectionBatch, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        # self(images, targets) returns dict from YOLOX.forward
        outputs = self(images, targets)

        batch_size = (
            images.tensors.shape[0] if hasattr(images, "tensors") else images.shape[0]
        )

        loss = outputs["total_loss"]
        iou_loss = outputs["iou_loss"]
        obj_loss = outputs["conf_loss"]
        cls_loss = outputs["cls_loss"]
        l1_loss = outputs["l1_loss"]
        num_fg = outputs["num_fg"]

        # Log losses (sync_dist=True for correct epoch-level DDP aggregation)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/iou_loss",
            iou_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/obj_loss",
            obj_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/cls_loss",
            cls_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/l1_loss",
            l1_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log num_fg
        num_fg_val: float | torch.Tensor
        if isinstance(num_fg, torch.Tensor):
            num_fg_val = num_fg.float().mean()
        else:
            num_fg_val = float(num_fg)
        self.log(
            "train/num_fg",
            num_fg_val,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return torch.as_tensor(loss)

    def forward(
        self, images: torch.Tensor, targets: list[DetectionTarget] | None = None
    ) -> ModelOutputs:
        """Forward pass.

        Targets are expected in pixel CXCYWH format (from YOLOX transforms
        which apply ConvertBoundingBoxFormat). They are passed directly to the
        YOLOX head — no format conversion needed here.
        """
        # Handle NestedTensor from rfdetr collation
        if hasattr(images, "tensors"):
            images = images.tensors

        # YOLOX pretrained weights were trained with BGR input (OpenCV convention).
        # Our data pipeline uses PIL which produces RGB. Swap R↔B channels so that
        # pretrained features are used correctly from epoch 1.
        images = images[:, [2, 1, 0], :, :]

        if self._export_mode:
            result: dict[str, torch.Tensor] = self.model(images)
            return result

        if targets is not None:
            # Transforms output pixel CXCYWH — pass directly to YOLOX head
            outputs: ModelOutputs = self.model(images, targets)
            if "image_shape" not in outputs:
                outputs["image_shape"] = images.shape[2:]
            return outputs
        else:
            with torch.no_grad():
                raw_outputs = self.model(images)
            return {"predictions": raw_outputs, "image_shape": images.shape[2:]}

    def get_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        original_sizes: list[tuple[int, int]] | None = None,
        confidence_threshold: float = 0.1,
        nms_iou_threshold: float = 0.45,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert model outputs to prediction format.

        Returns boxes in normalized [0,1] XYXY coordinates when
        ``original_sizes`` is not provided (matching the base class contract).
        When ``original_sizes`` is provided, boxes are in absolute pixel XYXY.
        """
        predictions: list[dict[str, torch.Tensor]] = []

        pred = outputs.get("predictions")
        if pred is None:
            return predictions

        # YOLOX outputs: [batch, num_anchors, 5 + num_classes]
        # Format: [cx, cy, w, h, obj_conf, cls_conf...] in pixel coords
        batch_size = pred.shape[0]
        img_shape = outputs.get("image_shape", (self.input_height, self.input_width))
        img_h, img_w = img_shape[0], img_shape[1]

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

            # pixel cxcywh → pixel xyxy
            boxes = cxcywh_to_xyxy(box_preds[:, :4])

            # NMS
            from torchvision.ops import batched_nms

            keep_indices = batched_nms(
                boxes, scores, labels, iou_threshold=nms_iou_threshold
            )
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

            if original_sizes is not None:
                # Scale to requested original image dimensions
                orig_h, orig_w = original_sizes[b]
                scale = torch.tensor(
                    [orig_w / img_w, orig_h / img_h, orig_w / img_w, orig_h / img_h],
                    device=boxes.device,
                    dtype=boxes.dtype,
                )
                boxes = boxes * scale
            elif boxes.numel() > 0:
                # Normalize to [0,1] XYXY — standard output format
                norm_scale = torch.tensor(
                    [img_w, img_h, img_w, img_h],
                    device=boxes.device,
                    dtype=boxes.dtype,
                )
                boxes = boxes / norm_scale

            predictions.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
            )

        return predictions

    def validation_step(self, batch: DetectionBatch, batch_idx: int) -> None:
        """Validation step for YOLOX.

        Targets are in pixel CXCYWH (from YOLOX transforms). Predictions are
        normalized [0,1] XYXY from get_predictions(). Targets are converted to
        [0,1] XYXY to match for the supervision MeanAveragePrecision metric.
        """
        images, targets = batch
        outputs = self(images, targets)

        batch_size = (
            images.tensors.shape[0] if hasattr(images, "tensors") else images.shape[0]
        )

        # --- Log losses (same as base) ---
        loss_components = {
            k: v for k, v in outputs.items() if "loss" in k.lower() and v.numel() == 1
        }
        if "total_loss" in outputs:
            total_loss = outputs["total_loss"]
            loss_components.pop("total_loss", None)
        elif "loss" in outputs:
            total_loss = outputs["loss"]
            loss_components.pop("loss", None)
        else:
            total_loss = (
                sum(loss_components.values()) if loss_components else torch.tensor(0.0)
            )
        self.log(
            "val/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        for name, value in loss_components.items():
            log_name = name if name.startswith("val/") else f"val/{name}"
            self.log(
                log_name,
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        # Predictions are already normalized [0,1] XYXY from get_predictions()
        preds = self.get_predictions(outputs, confidence_threshold=0.0)

        # Convert targets from pixel CXCYWH → normalized [0,1] XYXY for metrics
        img_h, img_w = outputs.get("image_shape", (self.input_height, self.input_width))
        norm_targets = []
        for t in targets:
            boxes = t["boxes"].clone()
            if boxes.numel() > 0:
                boxes = cxcywh_to_xyxy(boxes)
                boxes[:, [0, 2]] /= img_w
                boxes[:, [1, 3]] /= img_h
            norm_targets.append({"boxes": boxes, "labels": t["labels"]})

        sv_preds, sv_targets = self._to_sv_detections(preds, norm_targets)
        self.val_map.update(sv_preds, sv_targets)

        # Store for curve computation (CPU, normalized [0,1] xyxy)
        self.val_preds_storage.extend(
            [{k: v.cpu() for k, v in p.items()} for p in preds]
        )
        self.val_targets_storage.extend(
            [{k: v.cpu() for k, v in t.items()} for t in norm_targets]
        )

    def configure_optimizers(self) -> OptimizerConfig:
        """
        Configure optimizer and learning rate scheduler.

        Matches official YOLOX parameter group structure:
        - pg0: BatchNorm weights (no weight decay)
        - pg1: Other weights (with weight decay)
        - pg2: Biases (no weight decay)

        Returns:
            Optimizer and scheduler configuration.
        """
        pg0: list[nn.Parameter] = []  # BN weights - no decay
        pg1: list[nn.Parameter] = []  # Other weights - with decay
        pg2: list[nn.Parameter] = []  # Biases - no decay

        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        momentum = 0.9
        nesterov = True

        optimizer = torch.optim.SGD(
            pg0,
            lr=self.learning_rate,
            momentum=momentum,
            nesterov=nesterov,
        )
        optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})
        optimizer.add_param_group({"params": pg2})

        # Scheduler with warmup + cosine annealing
        total_steps: int
        if self.trainer and hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            total_steps = 10000

        total_steps = max(1, total_steps)

        warmup_epochs = self.warmup_epochs
        max_epochs: int = (self.trainer.max_epochs or 100) if self.trainer else 100

        # Calculate warmup steps
        warmup_steps = int(total_steps * (warmup_epochs / max(1, max_epochs)))
        warmup_steps = min(warmup_steps, total_steps // 2)
        warmup_steps = max(1, warmup_steps)

        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps
        )

        # Main scheduler (Cosine Annealing)
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

    def export_onnx(
        self,
        output_path: str,
        input_height: int = 640,
        input_width: int = 640,
        opset_version: int = 17,
        simplify: bool = True,
        dynamic_axes: dict[str, Any] | None = None,
    ) -> str:
        """Export YOLOX to ONNX with a single output tensor.

        YOLOX outputs ``[batch, num_anchors, 5 + num_classes]`` where
        columns are ``[cx, cy, w, h, obj_conf, cls_0, ...]`` in pixel
        coordinates.  This differs from the base class which assumes
        three separate output tensors.
        """
        import onnx

        self.set_export_mode(True)
        device = next(self.parameters()).device

        input_shape = (1, 3, input_height, input_width)
        dummy_input = torch.randn(*input_shape, device=device)

        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting YOLOX to ONNX: {out_path}")
        torch.onnx.export(
            self,
            (dummy_input,),
            str(out_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        if simplify:
            try:
                import onnxsim

                model = onnx.load(str(out_path))
                model_simp, check = onnxsim.simplify(model)
                if check:
                    onnx.save(model_simp, str(out_path))
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX simplification failed, using original model")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        logger.info(f"YOLOX ONNX export complete: {out_path}")
        return str(out_path)


# Register model variants with Hydra
@register(name="YOLOXNano")
class YOLOXNanoModel(YOLOXLightningModel):
    """YOLOX Nano model for Hydra instantiation."""

    _checkpoint_name = "yolox_nano.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="YOLOXTiny")
class YOLOXTinyModel(YOLOXLightningModel):
    """YOLOX Tiny model for Hydra instantiation."""

    _checkpoint_name = "yolox_tiny.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="YOLOXS")
class YOLOXSModel(YOLOXLightningModel):
    """YOLOX Small model for Hydra instantiation."""

    _checkpoint_name = "yolox_s.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="YOLOXM")
class YOLOXMModel(YOLOXLightningModel):
    """YOLOX Medium model for Hydra instantiation."""

    _checkpoint_name = "yolox_m.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="YOLOXL")
class YOLOXLModel(YOLOXLightningModel):
    """YOLOX Large model for Hydra instantiation."""

    _checkpoint_name = "yolox_l.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)


@register(name="YOLOXX")
class YOLOXXModel(YOLOXLightningModel):
    """YOLOX X-Large model for Hydra instantiation."""

    _checkpoint_name = "yolox_x.pth"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        super().__init__(**kwargs)
