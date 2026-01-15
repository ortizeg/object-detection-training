"""
YOLOX model wrapper for PyTorch Lightning.

This module provides Lightning-compatible wrappers for YOLOX models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from object_detection_training.models.base import BaseDetectionModel
from object_detection_training.utils.hydra import register
from object_detection_training.yolox import YOLOPAFPN, YOLOX, YOLOXHead

# YOLOX checkpoint URLs from official releases
YOLOX_CHECKPOINT_URLS = {
    "yolox_nano": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_nano.pth"
    ),
    "yolox_tiny": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_tiny.pth"
    ),
    "yolox_s": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_s.pth"
    ),
    "yolox_m": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_m.pth"
    ),
    "yolox_l": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_l.pth"
    ),
    "yolox_x": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_x.pth"
    ),
}

# YOLOX model configurations (depth, width)
YOLOX_CONFIGS = {
    "nano": {"depth": 0.33, "width": 0.25, "depthwise": True},
    "tiny": {"depth": 0.33, "width": 0.375, "depthwise": False},
    "s": {"depth": 0.33, "width": 0.50, "depthwise": False},
    "m": {"depth": 0.67, "width": 0.75, "depthwise": False},
    "l": {"depth": 1.0, "width": 1.0, "depthwise": False},
    "x": {"depth": 1.33, "width": 1.25, "depthwise": False},
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


class YOLOXLightningModel(BaseDetectionModel):
    """
    PyTorch Lightning wrapper for YOLOX models.

    Wraps YOLOX with Lightning training framework.
    """

    def __init__(
        self,
        variant: str = "s",
        num_classes: int = 80,
        pretrain_weights: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 5,
        use_ema: bool = True,
        ema_decay: float = 0.9998,
        download_pretrained: bool = True,
        input_height: int = 640,
        input_width: int = 640,
        output_dir: str = "outputs",
    ):
        """
        Initialize YOLOX Lightning model.

        Args:
            variant: Model variant (nano, tiny, s, m, l, x).
            num_classes: Number of detection classes.
            pretrain_weights: Path to pretrained weights file.
            learning_rate: Base learning rate.
            weight_decay: Weight decay.
            warmup_epochs: Number of warmup epochs.
            use_ema: Enable EMA.
            ema_decay: EMA decay factor.
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
            use_ema=use_ema,
            ema_decay=ema_decay,
            input_height=input_height,
            input_width=input_width,
            output_dir=output_dir,
        )

        self.variant = variant
        self.pretrain_weights = pretrain_weights
        self.download_pretrained = download_pretrained
        self.input_height = input_height
        self.input_width = input_width

        if variant not in YOLOX_CONFIGS:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from {list(YOLOX_CONFIGS.keys())}"
            )

        config = YOLOX_CONFIGS[variant]
        depth = config["depth"]
        width = config["width"]
        depthwise = config["depthwise"]

        # Build YOLOX model
        logger.info(f"Initializing YOLOX {variant} model")

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(
            depth=depth,
            width=width,
            in_channels=in_channels,
            depthwise=depthwise,
        )
        head = YOLOXHead(
            num_classes=num_classes,
            width=width,
            in_channels=in_channels,
            depthwise=depthwise,
        )

        self.model = YOLOX(backbone=backbone, head=head)

        # Load pretrained weights
        if pretrain_weights:
            self._load_weights(pretrain_weights)
        elif download_pretrained:
            self._download_and_load_weights()

        self.save_hyperparameters()

    def _download_and_load_weights(self):
        """Download and load pretrained weights."""
        if self.variant not in YOLOX_CHECKPOINT_URLS:
            logger.warning(f"No pretrained weights available for {self.variant}")
            return

        cache_dir = Path.home() / ".cache" / "yolox"
        checkpoint_path = cache_dir / f"yolox_{self.variant}.pth"

        if not checkpoint_path.exists():
            url = YOLOX_CHECKPOINT_URLS[self.variant]
            download_checkpoint(url, checkpoint_path)

        self._load_weights(str(checkpoint_path))

    def _load_weights(self, checkpoint_path: str):
        """Load weights from checkpoint."""
        logger.info(f"Loading weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

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
            logger.info(f"Checkpoint match summary for {self.variant}:")
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
        self.log(
            "train/num_fg",
            num_fg.float() if isinstance(num_fg, torch.Tensor) else float(num_fg),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

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

        if self.training and targets is not None:
            outputs = self.model(images, targets)
            return outputs
        else:
            with torch.no_grad():
                outputs = self.model(images)
            return {"predictions": outputs}

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

            # Get boxes (cxcywh -> xyxy)
            boxes = box_preds[:, :4].clone()
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes[:, 0] = cx - w / 2
            boxes[:, 1] = cy - h / 2
            boxes[:, 2] = cx + w / 2
            boxes[:, 3] = cy + h / 2

            # Get scores and labels
            obj_conf = box_preds[:, 4]
            cls_conf = box_preds[:, 5:]
            scores, labels = cls_conf.max(dim=1)
            scores = scores * obj_conf

            # Filter by threshold
            keep = scores > confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Normalize boxes to [0, 1] relative to input resolution
            # YOLOX predictions are in absolute pixels for the input_height/width
            boxes[:, [0, 2]] /= self.input_width
            boxes[:, [1, 3]] /= self.input_height

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


# Register model variants with Hydra
@register(group="model")
class YOLOXNanoModel(YOLOXLightningModel):
    """YOLOX Nano model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="nano", **kwargs)


@register(group="model")
class YOLOXTinyModel(YOLOXLightningModel):
    """YOLOX Tiny model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="tiny", **kwargs)


@register(group="model")
class YOLOXSModel(YOLOXLightningModel):
    """YOLOX Small model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="s", **kwargs)


@register(group="model")
class YOLOXMModel(YOLOXLightningModel):
    """YOLOX Medium model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="m", **kwargs)


@register(group="model")
class YOLOXLModel(YOLOXLightningModel):
    """YOLOX Large model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="l", **kwargs)


@register(group="model")
class YOLOXXModel(YOLOXLightningModel):
    """YOLOX X-Large model for Hydra instantiation."""

    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(variant="x", **kwargs)
