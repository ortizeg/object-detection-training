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
from object_detection_training.utils.boxes import cxcywh_to_xyxy
from object_detection_training.utils.hydra import register

# YOLOX checkpoint URLs from official releases
YOLOX_CHECKPOINT_URLS = {
    "nano": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_nano.pth"
    ),
    "tiny": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_tiny.pth"
    ),
    "s": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_s.pth"
    ),
    "m": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_m.pth"
    ),
    "l": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
        "yolox_l.pth"
    ),
    "x": (
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
        variant: str = "s",
        num_classes: int = 80,
        pretrain_weights: str | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 5,
        download_pretrained: bool = True,
        input_height: int = 640,
        input_width: int = 640,
        output_dir: str = "outputs",
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        freeze_backbone_epochs: int = 0,
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
            download_pretrained: Download pretrained weights if not available.
            input_height: Input image height.
            input_width: Input image width.
            output_dir: Base directory for outputting results.
            freeze_backbone_epochs: Freeze backbone for this many initial epochs
                during fine-tuning. 0 disables freezing.
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

        self.image_mean = image_mean if image_mean is not None else [0.0, 0.0, 0.0]
        self.image_std = image_std if image_std is not None else [1.0, 1.0, 1.0]

        self.variant = variant
        self.pretrain_weights = pretrain_weights
        self.download_pretrained = download_pretrained
        self.input_height = input_height
        self.input_width = input_width
        self.freeze_backbone_epochs = freeze_backbone_epochs

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
        """Unfreeze backbone after freeze_backbone_epochs."""
        if (
            self.freeze_backbone_epochs > 0
            and self.current_epoch == self.freeze_backbone_epochs
        ):
            self._unfreeze_backbone()

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
            RuntimeError: If the variant has no checkpoint URL, or if
                download/loading fails. Training must not proceed
                without pretrained weights when they were requested.
        """
        if self.variant not in YOLOX_CHECKPOINT_URLS:
            raise RuntimeError(
                f"download_pretrained=True but no checkpoint URL for "
                f"variant '{self.variant}'. Available: "
                f"{list(YOLOX_CHECKPOINT_URLS.keys())}"
            )

        cache_dir = Path.home() / ".cache" / "yolox"
        checkpoint_path = cache_dir / f"yolox_{self.variant}.pth"

        url = YOLOX_CHECKPOINT_URLS[self.variant]
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
        logger.info(f"Checkpoint match summary for {self.variant}:")
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

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
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
        num_fg_val: float | torch.Tensor
        if isinstance(num_fg, torch.Tensor):
            num_fg_val = num_fg.float().mean()
        else:
            num_fg_val = float(num_fg)
        self.log("train/num_fg", num_fg_val, on_step=True, on_epoch=True, prog_bar=True)

        return torch.as_tensor(loss)

    def forward(
        self, images: torch.Tensor, targets: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Forward pass."""
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

            outputs: dict[str, Any] = self.model(images, unnormalized_targets)
            # Add images shape to outputs for post-processing resolution retrieval
            if "image_shape" not in outputs:
                outputs["image_shape"] = images.shape[2:]
            return outputs
        else:
            with torch.no_grad():
                raw_outputs = self.model(images)
            # Add images shape to outputs for post-processing resolution retrieval
            return {"predictions": raw_outputs, "image_shape": images.shape[2:]}

    def get_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        original_sizes: list[tuple[int, int]] | None = None,
        confidence_threshold: float = 0.1,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert model outputs to prediction format."""
        predictions: list[dict[str, torch.Tensor]] = []

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

    def configure_optimizers(self) -> Any:
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


# Register model variants with Hydra
@register(name="YOLOXNano")
class YOLOXNanoModel(YOLOXLightningModel):
    """YOLOX Nano model for Hydra instantiation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="nano", **kwargs)


@register(name="YOLOXTiny")
class YOLOXTinyModel(YOLOXLightningModel):
    """YOLOX Tiny model for Hydra instantiation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="tiny", **kwargs)


@register(name="YOLOXS")
class YOLOXSModel(YOLOXLightningModel):
    """YOLOX Small model for Hydra instantiation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="s", **kwargs)


@register(name="YOLOXM")
class YOLOXMModel(YOLOXLightningModel):
    """YOLOX Medium model for Hydra instantiation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="m", **kwargs)


@register(name="YOLOXL")
class YOLOXLModel(YOLOXLightningModel):
    """YOLOX Large model for Hydra instantiation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="l", **kwargs)


@register(name="YOLOXX")
class YOLOXXModel(YOLOXLightningModel):
    """YOLOX X-Large model for Hydra instantiation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="x", **kwargs)
