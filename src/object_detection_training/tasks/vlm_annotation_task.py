"""VLM Annotation Task using Gemini."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydantic import Field

from object_detection_training.tasks.base_task import BaseTask
from object_detection_training.utils.hydra import register

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})


@register(group="task")
class VLMAnnotationTask(BaseTask):
    """Run VLM inference on a directory of images to generate annotations.

    Uses Gemini (or other VLMs) to detect objects and saves the results
    as DetectionAnnotation JSON files.
    """

    name: str = Field(default="vlm_annotation", description="Task name")

    # Data
    image_dir: Path = Field(description="Directory containing images")

    # Model Configuration
    model_name: str = Field(
        default="gemini-1.5-pro", description="Name of the VLM model to use"
    )
    classes: list[str] = Field(description="List of class names to detect")
    prompt_template: str | None = Field(
        default=None, description="Optional custom prompt template"
    )

    def run(self) -> dict[str, str | None]:
        """Run inference on all images and write annotation files.

        Returns:
            Dict with ``output_dir`` and ``num_images`` processed.
        """
        from object_detection_training.inference.gemini_inferencer import (
            GeminiInferencer,
        )
        from object_detection_training.io.annotation import (
            DetectionAnnotationWriter,
        )
        from object_detection_training.io.image import ImageLoader
        from object_detection_training.schemas.annotation import (
            AnnotationInfo,
            DetectionAnnotation,
        )

        # Resolve output directory
        if self.output_dir is None:
            self.output_dir = Path("vlm_annotations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize label map from classes list
        label_map = dict(enumerate(self.classes))

        # Initialize Inferencer
        logger.info(f"Initializing GeminiInferencer with model: {self.model_name}")
        inferencer = GeminiInferencer(
            model_name=self.model_name,
            classes=self.classes,
            prompt_template=self.prompt_template,
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
            try:
                loader = ImageLoader(img_path)
                image = loader.read()

                detections = inferencer.predict(
                    image,
                    image_width=loader.width,
                    image_height=loader.height,
                )

                annotation = DetectionAnnotation(
                    filename=loader.filename,
                    categories=label_map,
                    info=AnnotationInfo(
                        annotations_source=self.model_name,
                        image_width=loader.width,
                        image_height=loader.height,
                    ),
                    annotations=detections,
                )

                writer.write(annotation)
                logger.debug(f"{loader.filename}: {len(detections)} detections")

            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {e}")
                continue

        logger.info(
            f"VLM Inference complete. {len(image_paths)} images processed -> "
            f"{self.output_dir}"
        )
        return {
            "output_dir": str(self.output_dir),
            "num_images": str(len(image_paths)),
        }
