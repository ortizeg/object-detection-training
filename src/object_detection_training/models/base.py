"""
Base detection model abstraction using PyTorch Lightning.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import lightning as L
import supervision as sv
import torch
from loguru import logger
from supervision.metrics import MeanAveragePrecision

from object_detection_training.metrics.curves import compute_detection_curves
from object_detection_training.utils.boxes import cxcywh_to_xyxy
from object_detection_training.utils.plotting import save_detection_curves_plots


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
        warmup_epochs: int = 5,
        input_height: int = 576,
        input_width: int = 576,
        output_dir: str = "outputs",
    ):
        """
        Initialize the base detection model.

        Args:
            num_classes: Number of detection classes.
            learning_rate: Base learning rate.
            weight_decay: Weight decay for optimizer.
            warmup_epochs: Number of warmup epochs.
            output_dir: Base directory for outputting results (metrics/images).
        """
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.input_height = input_height
        self.input_width = input_width
        self.output_dir = Path(output_dir)

        self._export_mode = False

        # Metrics
        self.val_map = MeanAveragePrecision()
        self.test_map = MeanAveragePrecision()

        self.save_hyperparameters()

    @abstractmethod
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, images: torch.Tensor, targets: list[dict] | None = None
    ) -> dict[str, torch.Tensor]:
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
        outputs: dict[str, torch.Tensor],
        original_sizes: list[tuple[int, int]] | None = None,
        confidence_threshold: float = 0.0,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Convert model outputs to prediction format.

        Args:
            outputs: Raw model outputs.
            original_sizes: Original image sizes for rescaling boxes.
            confidence_threshold: Threshold for filtering predictions.

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
        dynamic_axes: dict | None = None,
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

        try:
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
                        logger.warning(
                            "ONNX simplification failed, using original model"
                        )
                except Exception as e:
                    logger.warning(f"ONNX simplification failed: {e}")
                    logger.warning("Using original ONNX model")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            import traceback

            logger.error(traceback.format_exc())
            if output_path.exists():
                output_path.unlink()

        self.set_export_mode(False)
        return str(output_path)

    def compute_model_stats(
        self,
        input_height: int = 640,
        input_width: int = 640,
    ) -> dict[str, Any]:
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
            # Targeting self.model instead of self to avoid Lightning wrapper overhead
            # and potential recursion issues with some FLOP counters.
            model_to_analyze = getattr(self, "model", self)

            # Set model to eval mode and try FLOP analysis
            was_training = model_to_analyze.training
            model_to_analyze.eval()
            try:
                flops = FlopCountAnalysis(model_to_analyze, dummy_input)
                # Suppress warnings about untraced operations
                flops.unsupported_ops_warnings(False)
                flops.uncalled_modules_warnings(False)
                flops.tracer_warnings("none")
                total_flops = flops.total()
            except Exception:
                # Fallback: analyze backbone only if full model tracing fails
                # This is common for detection heads that use non-traceable ops
                # (e.g., grid computations, argmax for labels)
                backbone = getattr(model_to_analyze, "backbone", None)
                if backbone is not None:
                    try:
                        flops = FlopCountAnalysis(backbone, dummy_input)
                        flops.unsupported_ops_warnings(False)
                        flops.uncalled_modules_warnings(False)
                        flops.tracer_warnings("none")
                        total_flops = flops.total()
                        logger.debug(
                            "FLOP count based on backbone only (head tracing failed)"
                        )
                    except Exception as e2:
                        logger.warning(f"Failed to compute FLOPs (backbone): {e2}")
                        total_flops = 0
                else:
                    total_flops = 0
            finally:
                # Restore original training state
                model_to_analyze.train(was_training)
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

        self.train(was_training)
        return stats

    def _to_sv_detections(
        self,
        preds: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[list[sv.Detections], list[sv.Detections]]:
        """
        Convert framework format to supervision Detections.

        Args:
            preds: List of prediction dictionaries.
            targets: List of target dictionaries.

        Returns:
            Tuple of (list of sv.Detections, list of sv.Detections).
        """
        sv_preds = []
        for p in preds:
            sv_preds.append(
                sv.Detections(
                    xyxy=p["boxes"].cpu().numpy(),
                    confidence=p["scores"].float().cpu().numpy(),
                    class_id=p["labels"].cpu().numpy().astype(int),
                )
            )

        sv_targets = []
        for t in targets:
            sv_targets.append(
                sv.Detections(
                    xyxy=t["boxes"].cpu().numpy(),
                    class_id=t["labels"].cpu().numpy().astype(int),
                )
            )

        return sv_preds, sv_targets

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        outputs = self(images, targets)

        # Individual loss components for logging (scalars only)
        loss_components = {
            k: v for k, v in outputs.items() if "loss" in k.lower() and v.numel() == 1
        }

        # Determine total loss, avoiding double counting 'loss' or 'total_loss'
        if "loss" in outputs:
            total_loss = outputs["loss"]
            loss_components.pop("loss", None)
        elif "total_loss" in outputs:
            total_loss = outputs["total_loss"]
            loss_components.pop("total_loss", None)
        else:
            total_loss = (
                sum(loss_components.values()) if loss_components else torch.tensor(0.0)
            )

        # Log losses
        for name, value in loss_components.items():
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
        outputs = self(images, targets)

        # Extract individual loss components for detailed logging
        loss_components = {
            k: v for k, v in outputs.items() if "loss" in k.lower() and v.numel() == 1
        }

        # Determine total loss, avoiding double counting
        if "loss" in outputs:
            total_loss = outputs["loss"]
            loss_components.pop("loss", None)
        elif "total_loss" in outputs:
            total_loss = outputs["total_loss"]
            loss_components.pop("total_loss", None)
        else:
            total_loss = (
                sum(loss_components.values()) if loss_components else torch.tensor(0.0)
            )

        if total_loss is not None:
            self.log(
                "val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
            )
            for name, value in loss_components.items():
                if not name.startswith("val/"):
                    name = f"val/{name}"
                self.log(name, value, on_step=False, on_epoch=True)

        # Convert to predictions format
        preds = self.get_predictions(outputs, confidence_threshold=0.0)

        # Convert targets to metrics format
        formatted_targets = []
        for target in targets:
            # Targets are normalized cxcywh, metric expects xyxy
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = cxcywh_to_xyxy(boxes)

            formatted_targets.append(
                {
                    "boxes": boxes,
                    "labels": target["labels"],
                }
            )

        sv_preds, sv_targets = self._to_sv_detections(preds, formatted_targets)
        self.val_map.update(sv_preds, sv_targets)

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
        result = self.val_map.compute()
        logger.info(
            f"Validation mAP@50: {result.map50:.4f}, mAP@50:95: {result.map50_95:.4f}"
        )

        # Log main metrics
        self.log("val/mAP", result.map50_95, prog_bar=True)
        self.log("val/mAP_50", result.map50, prog_bar=True)
        self.log("val/mAP_75", result.map75)

        # Log per-class mAP if available
        class_names = getattr(self.trainer.datamodule, "class_names", None)

        if class_names and result.ap_per_class is not None:
            # Create a mapping from class_id to AP (average across IoU thresholds)
            class_ap_map = {
                class_id: float(result.ap_per_class[i].mean())
                for i, class_id in enumerate(result.matched_classes)
            }
            for class_id, name in enumerate(class_names):
                ap = class_ap_map.get(class_id, 0.0)
                self.log(f"val/mAP_{name}", ap)
        elif result.ap_per_class is not None:
            for i, class_id in enumerate(result.matched_classes):
                name = f"class_{class_id}"
                ap = float(result.ap_per_class[i].mean())
                self.log(f"val/mAP_{name}", ap)

        self.val_map.reset()

        # Compute and log curves (PR, F1)
        curves = compute_detection_curves(
            self.val_preds_storage, self.val_targets_storage, self.num_classes
        )

        epoch_dir = self.output_dir / "val_results" / f"epoch_{self.current_epoch:03d}"
        metrics_dir = epoch_dir / "metrics"

        # Build metrics dict for compatibility with plotting curves
        metrics = {
            "map": result.map50_95,
            "map_50": result.map50,
            "map_75": result.map75,
            "map_per_class": result.ap_per_class,
        }

        save_detection_curves_plots(curves, class_names, metrics_dir, metrics=metrics)

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

        preds = self.get_predictions(outputs, confidence_threshold=0.0)

        formatted_targets = []
        for target in targets:
            # Targets are normalized cxcywh, metric expects xyxy
            boxes = target["boxes"]
            if boxes.numel() > 0:
                # cxcywh -> xyxy
                boxes = cxcywh_to_xyxy(boxes)

            formatted_targets.append(
                {
                    "boxes": boxes,
                    "labels": target["labels"],
                }
            )

        sv_preds, sv_targets = self._to_sv_detections(preds, formatted_targets)
        self.test_map.update(sv_preds, sv_targets)

        # Store for curve computation
        self.test_preds_storage.extend(
            [{k: v.cpu() for k, v in p.items()} for p in preds]
        )
        self.test_targets_storage.extend(
            [{k: v.cpu() for k, v in t.items()} for t in formatted_targets]
        )

    def on_test_epoch_end(self) -> None:
        """Compute test metrics at epoch end."""
        result = self.test_map.compute()

        # Build metrics dict for compatibility with plotting curves
        metrics = {
            "map": result.map50_95,
            "map_50": result.map50,
            "map_75": result.map75,
            "map_per_class": result.ap_per_class,
        }

        self.log("test/mAP", result.map50_95)
        self.log("test/mAP_50", result.map50)
        self.log("test/mAP_75", result.map75)

        class_names = getattr(self.trainer.datamodule, "class_names", None)

        if class_names and result.ap_per_class is not None:
            # Create a mapping from class_id to AP (average across IoU thresholds)
            class_ap_map = {
                class_id: float(result.ap_per_class[i].mean())
                for i, class_id in enumerate(result.matched_classes)
            }
            for class_id, name in enumerate(class_names):
                ap = class_ap_map.get(class_id, 0.0)
                self.log(f"test/mAP_{name}", ap)
        elif result.ap_per_class is not None:
            for i, class_id in enumerate(result.matched_classes):
                name = f"class_{class_id}"
                ap = float(result.ap_per_class[i].mean())
                self.log(f"test/mAP_{name}", ap)

        self.test_map.reset()

        # Log curves
        curves = compute_detection_curves(
            self.test_preds_storage, self.test_targets_storage, self.num_classes
        )

        metrics_dir = self.output_dir / "test_results" / "metrics"
        save_detection_curves_plots(curves, class_names, metrics_dir, metrics=metrics)

        self.test_preds_storage = []
        self.test_targets_storage = []
