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

from object_detection_training.metrics.curves import compute_detection_curves
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
        use_ema: bool = True,
        ema_decay: float = 0.9999,
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
            use_ema: Whether to use EMA for model weights.
            ema_decay: Decay factor for EMA.
            output_dir: Base directory for outputting results (metrics/images).
        """
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.input_height = input_height
        self.input_width = input_width
        self.output_dir = Path(output_dir)

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
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        raise NotImplementedError

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
        confidence_threshold: float = 0.0,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert model outputs to prediction format.

        Args:
            outputs: Raw model outputs.
            original_sizes: Original image sizes for rescaling boxes.

        Returns:
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
        preds = self.get_predictions(outputs, confidence_threshold=0.0)

        # Convert targets to metrics format
        formatted_targets = []
        for target in targets:
            # Targets are normalized cxcywh, metric expects xyxy
            boxes = target["boxes"]
            if boxes.numel() > 0:
                # cxcywh -> xyxy
                cx, cy, w, h = boxes.unbind(-1)
                x1 = cx - 0.5 * w
                y1 = cy - 0.5 * h
                x2 = cx + 0.5 * w
                y2 = cy + 0.5 * h
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            formatted_targets.append(
                {
                    "boxes": boxes,
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
        classes_tensor = metrics.get("classes")
        map_per_class = metrics.get("map_per_class")

        if class_names and map_per_class is not None:
            for class_id, name in enumerate(class_names):
                ap = torch.tensor(-1.0, device=self.device)
                if classes_tensor is not None:
                    # Find where this class ID appears in the classes tensor
                    matches = (classes_tensor == class_id).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        ap = map_per_class[matches[0].item()]

                self.log(f"val/mAP_{name}", ap)
        elif map_per_class is not None:
            # Fallback to whatever metrics returned
            for i, ap in enumerate(map_per_class):
                class_id = i
                if classes_tensor is not None and i < len(classes_tensor):
                    class_id = int(classes_tensor[i].item())

                name = f"class_{class_id}"
                self.log(f"val/mAP_{name}", ap)

        self.val_map.reset()

        # Compute and log curves (PR, F1)
        curves = compute_detection_curves(
            self.val_preds_storage, self.val_targets_storage, self.num_classes
        )

        epoch_dir = self.output_dir / f"epoch_{self.current_epoch:03d}"
        metrics_dir = epoch_dir / "metrics"

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
                cx, cy, w, h = boxes.unbind(-1)
                x1 = cx - 0.5 * w
                y1 = cy - 0.5 * h
                x2 = cx + 0.5 * w
                y2 = cy + 0.5 * h
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            formatted_targets.append(
                {
                    "boxes": boxes,
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
        classes_tensor = metrics.get("classes")
        map_per_class = metrics.get("map_per_class")

        if class_names and map_per_class is not None:
            for class_id, name in enumerate(class_names):
                ap = torch.tensor(-1.0, device=self.device)
                if classes_tensor is not None:
                    matches = (classes_tensor == class_id).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        ap = map_per_class[matches[0].item()]
                self.log(f"test/mAP_{name}", ap)
        elif map_per_class is not None:
            for i, ap in enumerate(map_per_class):
                class_id = i
                if classes_tensor is not None and i < len(classes_tensor):
                    class_id = int(classes_tensor[i].item())
                name = f"class_{class_id}"
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
