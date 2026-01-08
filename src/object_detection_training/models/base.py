"""
Base detection model abstraction using PyTorch Lightning.
"""

import json
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
from loguru import logger
from torchmetrics.detection import MeanAveragePrecision


class BaseDetectionModel(L.LightningModule):
    """
    Abstract base class for object detection models.

    Provides a common interface for training, validation, and testing
    detection models with PyTorch Lightning.
    """

    def __init__(
        self,
        num_classes: int = 80,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_encoder: Optional[float] = None,
        warmup_epochs: int = 5,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ):
        """
        Initialize the base detection model.

        Args:
            num_classes: Number of detection classes.
            learning_rate: Base learning rate.
            weight_decay: Weight decay for optimizer.
            lr_encoder: Learning rate for encoder/backbone (if different).
            warmup_epochs: Number of warmup epochs.
            use_ema: Whether to use EMA for model weights.
            ema_decay: Decay factor for EMA.
        """
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_encoder = lr_encoder or learning_rate
        self.warmup_epochs = warmup_epochs
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        self._export_mode = False

        # Metrics
        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True,
        )
        self.test_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True,
        )

        self.save_hyperparameters()

    @abstractmethod
    def forward(
        self, images: torch.Tensor, targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            images: Batch of images [B, C, H, W].
            targets: Optional list of target dictionaries for training.

        Returns:
            Dictionary containing model outputs (losses during training,
            predictions during inference).
        """
        pass

    @abstractmethod
    def get_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert model outputs to prediction format.

        Args:
            outputs: Raw model outputs.
            original_sizes: Original image sizes for rescaling boxes.

        Returns:
            List of prediction dictionaries with 'boxes', 'scores', 'labels'.
        """
        pass

    def set_export_mode(self, export: bool = True) -> None:
        """
        Set model to export mode for ONNX export.

        Args:
            export: Whether to enable export mode.
        """
        self._export_mode = export
        self.eval()
        if export:
            logger.info("Model set to export mode")

    def export_onnx(
        self,
        output_path: str,
        input_height: int = 640,
        input_width: int = 640,
        opset_version: int = 17,
        simplify: bool = True,
        dynamic_axes: Optional[Dict] = None,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model.
            input_height: Input image height.
            input_width: Input image width.
            opset_version: ONNX opset version.
            simplify: Whether to simplify the ONNX model.
            dynamic_axes: Dynamic axes for input/output.

        Returns:
            Path to exported ONNX model.
        """
        import onnx

        self.set_export_mode(True)
        device = next(self.parameters()).device

        input_shape = (1, 3, input_height, input_width)
        dummy_input = torch.randn(*input_shape, device=device)

        if dynamic_axes is None:
            dynamic_axes = {
                "images": {0: "batch_size"},
                "boxes": {0: "batch_size"},
                "scores": {0: "batch_size"},
                "labels": {0: "batch_size"},
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to ONNX: {output_path}")
        torch.onnx.export(
            self,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["images"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes=dynamic_axes,
        )

        if simplify:
            try:
                import onnxsim

                model = onnx.load(str(output_path))
                model_simp, check = onnxsim.simplify(model)
                if check:
                    onnx.save(model_simp, str(output_path))
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX simplification failed, using original model")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        self.set_export_mode(False)
        return str(output_path)

    def compute_model_stats(
        self,
        input_height: int = 640,
        input_width: int = 640,
    ) -> Dict[str, Any]:
        """
        Compute model statistics (parameters, FLOPs, size, inference speed).

        Args:
            input_height: Input image height for FLOPs computation.
            input_width: Input image width for FLOPs computation.

        Returns:
            Dictionary with model statistics.
        """
        device = next(self.parameters()).device
        input_shape = (1, 3, input_height, input_width)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Compute FLOPs
        try:
            from fvcore.nn import FlopCountAnalysis

            dummy_input = torch.randn(*input_shape, device=device)
            flops = FlopCountAnalysis(self, dummy_input)
            total_flops = flops.total()
        except Exception as e:
            logger.warning(f"Failed to compute FLOPs: {e}")
            total_flops = 0

        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        # Measure inference speed
        dummy_input = torch.randn(*input_shape, device=device)
        was_training = self.training
        self.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self(dummy_input)

        # Measure
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) / num_runs * 1000

        stats = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": total_flops,
            "model_size_mb": round(model_size_mb, 2),
            "inference_time_ms": round(inference_time_ms, 2),
            "fps": round(1000 / inference_time_ms, 2) if inference_time_ms > 0 else 0,
            "input_shape": list(input_shape),
        }

        logger.info(f"Model stats: {json.dumps(stats, indent=2)}")
        self.train(was_training)
        return stats

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        outputs = self(images, targets)

        # Sum all losses
        losses = {k: v for k, v in outputs.items() if "loss" in k.lower()}
        total_loss = sum(losses.values())

        # Log losses
        for name, value in losses.items():
            self.log(
                f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=False
            )
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def on_validation_epoch_start(self) -> None:
        """Initialize validation storage."""
        self.val_preds_storage = []
        self.val_targets_storage = []

    def validation_step(self, batch, batch_idx) -> None:
        """Validation step."""
        images, targets = batch
        outputs = self(images)

        # Convert to predictions format
        preds = self.get_predictions(outputs)

        # Convert targets to metrics format
        formatted_targets = []
        for target in targets:
            formatted_targets.append(
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                }
            )

        self.val_map.update(preds, formatted_targets)

        # Store for curve computation
        # Move to CPU to save GPU memory
        self.val_preds_storage.extend(
            [{k: v.cpu() for k, v in p.items()} for p in preds]
        )
        self.val_targets_storage.extend(
            [{k: v.cpu() for k, v in t.items()} for t in formatted_targets]
        )

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics at epoch end."""
        metrics = self.val_map.compute()

        # Log main metrics
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"], prog_bar=True)
        self.log("val/mAP_75", metrics["map_75"])

        # Log per-class mAP if available
        class_names = getattr(self.trainer.datamodule, "class_names", None)
        if "map_per_class" in metrics and metrics["map_per_class"].numel() > 0:
            for i, ap in enumerate(metrics["map_per_class"]):
                if not torch.isnan(ap):
                    name = f"class_{i}"
                    if class_names and i < len(class_names):
                        name = class_names[i]
                    self.log(f"val/mAP_{name}", ap)

        self.val_map.reset()

        # Compute and log curves (PR, F1)
        # Only log if we have a WandB logger and it's strictly the main process
        # to avoid duplicates
        # But logging is usually handled by rank_0.
        try:
            import wandb

            from object_detection_training.metrics.curves import (
                compute_detection_curves,
            )

            curves = compute_detection_curves(
                self.val_preds_storage, self.val_targets_storage, self.num_classes
            )

            for logger_inst in self.trainer.loggers:
                if isinstance(logger_inst, L.pytorch.loggers.WandbLogger):
                    # Log curves for each class
                    for c_idx, curve_data in curves.items():
                        c_name = (
                            class_names[c_idx]
                            if class_names and c_idx < len(class_names)
                            else f"class_{c_idx}"
                        )

                        # Subsample for plotting if too large (WandB has limits)
                        step = max(1, len(curve_data["scores"]) // 1000)
                        indices = range(0, len(curve_data["scores"]), step)

                        # Data for Custom Chart: Score, Precision, Recall, F1
                        data = []
                        for i in indices:
                            data.append(
                                [
                                    curve_data["scores"][i],
                                    curve_data["precision"][i],
                                    curve_data["recall"][i],
                                    curve_data["f1"][i],
                                ]
                            )

                        # Create a WandB Table
                        table = wandb.Table(
                            data=data,
                            columns=["confidence", "precision", "recall", "f1"],
                        )

                        # Log Precision-Recall Curve
                        # Using raw table allows custom Vega plots, but `wandb.plot`
                        # is easier. Interactive threshold plot: F1 vs Confidence
                        logger_inst.experiment.log(
                            {
                                f"val/curves/{c_name}_pr": wandb.plot.line(
                                    table,
                                    "recall",
                                    "precision",
                                    title=f"PR Curve - {c_name}",
                                ),
                                f"val/curves/{c_name}_f1_conf": wandb.plot.line(
                                    table,
                                    "confidence",
                                    "f1",
                                    title=f"F1 vs Confidence - {c_name}",
                                ),
                            }
                        )
        except ImportError:
            pass  # Skip if dependencies missing
        except Exception as e:
            logger.warning(f"Failed to log curves: {e}")

        # Clear storage
        self.val_preds_storage = []
        self.val_targets_storage = []

    def on_test_epoch_start(self) -> None:
        self.test_preds_storage = []
        self.test_targets_storage = []

    def test_step(self, batch, batch_idx) -> None:
        """Test step."""
        images, targets = batch
        outputs = self(images)

        preds = self.get_predictions(outputs)

        formatted_targets = []
        for target in targets:
            formatted_targets.append(
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                }
            )

        self.test_map.update(preds, formatted_targets)

        # Store for curve computation
        self.test_preds_storage.extend(
            [{k: v.cpu() for k, v in p.items()} for p in preds]
        )
        self.test_targets_storage.extend(
            [{k: v.cpu() for k, v in t.items()} for t in formatted_targets]
        )

    def on_test_epoch_end(self) -> None:
        """Compute test metrics at epoch end."""
        metrics = self.test_map.compute()

        self.log("test/mAP", metrics["map"])
        self.log("test/mAP_50", metrics["map_50"])
        self.log("test/mAP_75", metrics["map_75"])

        class_names = getattr(self.trainer.datamodule, "class_names", None)
        if "map_per_class" in metrics and metrics["map_per_class"].numel() > 0:
            for i, ap in enumerate(metrics["map_per_class"]):
                if not torch.isnan(ap):
                    name = f"class_{i}"
                    if class_names and i < len(class_names):
                        name = class_names[i]
                    self.log(f"test/mAP_{name}", ap)

        self.test_map.reset()

        # Log curves
        try:
            import wandb

            from object_detection_training.metrics.curves import (
                compute_detection_curves,
            )

            curves = compute_detection_curves(
                self.test_preds_storage, self.test_targets_storage, self.num_classes
            )

            for logger_inst in self.trainer.loggers:
                if isinstance(logger_inst, L.pytorch.loggers.WandbLogger):
                    for c_idx, curve_data in curves.items():
                        c_name = (
                            class_names[c_idx]
                            if class_names and c_idx < len(class_names)
                            else f"class_{c_idx}"
                        )

                        step = max(1, len(curve_data["scores"]) // 1000)
                        indices = range(0, len(curve_data["scores"]), step)
                        data = []
                        for i in indices:
                            data.append(
                                [
                                    curve_data["scores"][i],
                                    curve_data["precision"][i],
                                    curve_data["recall"][i],
                                    curve_data["f1"][i],
                                ]
                            )

                        table = wandb.Table(
                            data=data,
                            columns=["confidence", "precision", "recall", "f1"],
                        )

                        logger_inst.experiment.log(
                            {
                                f"test/curves/{c_name}_pr": wandb.plot.line(
                                    table,
                                    "recall",
                                    "precision",
                                    title=f"PR Curve - {c_name}",
                                ),
                                f"test/curves/{c_name}_f1_conf": wandb.plot.line(
                                    table,
                                    "confidence",
                                    "f1",
                                    title=f"F1 vs Confidence - {c_name}",
                                ),
                            }
                        )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to log curves: {e}")

        self.test_preds_storage = []
        self.test_targets_storage = []

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

        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
