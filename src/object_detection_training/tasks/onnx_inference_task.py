"""
ONNX Inference Task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field

from object_detection_training.tasks.base_task import BaseTask
from object_detection_training.utils.hydra import register

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

_POST_PROCESSOR_REGISTRY: dict[str, type] = {}


def _init_post_processor_registry() -> dict[str, type]:
    """Lazily populate the registry to avoid circular imports."""
    if not _POST_PROCESSOR_REGISTRY:
        from object_detection_training.inference.postprocess import (
            RFDETRPostProcessor,
            YOLOXPostProcessor,
        )

        _POST_PROCESSOR_REGISTRY["yolox"] = YOLOXPostProcessor
        _POST_PROCESSOR_REGISTRY["rfdetr"] = RFDETRPostProcessor
    return _POST_PROCESSOR_REGISTRY


@register(group="task")
class ONNXInferenceTask(BaseTask):
    """Run ONNX inference on a directory of images.

    Produces per-image ``DetectionAnnotation`` JSON files with
    normalised bounding boxes and an embedded label map.
    """

    name: str = Field(default="inference_onnx", description="Task name")

    # Model
    model_path: Path = Field(description="Path to the .onnx model file")

    # Data
    image_dir: Path = Field(description="Directory containing images")

    # Label map
    label_map: dict[int, str] = Field(
        description="Mapping from class index to label name"
    )

    # Post-processing
    post_processor_type: str = Field(
        default="rfdetr",
        description="Post-processor type: 'yolox' or 'rfdetr'",
    )
    confidence_threshold: float = Field(
        default=0.25, description="Minimum detection confidence"
    )
    nms_iou_threshold: float = Field(
        default=0.45, description="NMS IoU threshold (YOLOX only)"
    )

    # Input dimensions
    input_height: int = Field(default=640, description="Model input height")
    input_width: int = Field(default=640, description="Model input width")
    batch_size: int = Field(default=1, description="Batch size for inference")

    def run(self) -> dict[str, str | None]:
        """Run inference on all images and write annotation files.

        Returns:
            Dict with ``output_dir`` and ``num_images`` processed.
        """
        from object_detection_training.inference.annotation import (
            DetectionAnnotationWriter,
        )
        from object_detection_training.inference.image import ImageLoader
        from object_detection_training.inference.inferencer import (
            ONNXInferencer,
        )
        from object_detection_training.inference.models import (
            DetectionAnnotation,
        )

        # Resolve output directory
        if self.output_dir is None:
            self.output_dir = Path("inference_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build post-processor
        registry = _init_post_processor_registry()
        pp_cls = registry.get(self.post_processor_type)
        if pp_cls is None:
            msg = (
                f"Unknown post_processor_type '{self.post_processor_type}'. "
                f"Available: {sorted(registry.keys())}"
            )
            raise ValueError(msg)

        pp_kwargs: dict[str, Any] = {
            "label_map": self.label_map,
            "confidence_threshold": self.confidence_threshold,
        }
        if self.post_processor_type == "yolox":
            pp_kwargs["nms_iou_threshold"] = self.nms_iou_threshold

        post_processor = pp_cls(**pp_kwargs)

        # Build inferencer
        inferencer = ONNXInferencer(
            model_path=self.model_path,
            post_processor=post_processor,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
        )

        # Discover images
        image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_paths:
            logger.warning(f"No images found in {self.image_dir}")
            return {"output_dir": str(self.output_dir), "num_images": "0"}

        logger.info(f"Found {len(image_paths)} images in {self.image_dir}")

        # Run inference and write annotations
        writer = DetectionAnnotationWriter(self.output_dir)
        for img_path in image_paths:
            loader = ImageLoader(img_path)
            image = loader.read()
            detections = inferencer.predict(
                image,
                image_width=loader.width,
                image_height=loader.height,
            )
            annotation = DetectionAnnotation(
                image_filename=loader.filename,
                image_width=loader.width,
                image_height=loader.height,
                label_map=self.label_map,
                detections=detections,
            )
            writer.write(annotation)
            logger.debug(f"{loader.filename}: {len(detections)} detections")

        logger.info(
            f"Inference complete. {len(image_paths)} images -> {self.output_dir}"
        )
        return {
            "output_dir": str(self.output_dir),
            "num_images": str(len(image_paths)),
        }
