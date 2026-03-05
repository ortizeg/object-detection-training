"""Visualization utilities using supervision."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from object_detection_training.schemas.annotation import DetectionAnnotation


def draw_detections(
    image: npt.NDArray[np.uint8],
    annotation: DetectionAnnotation,
) -> npt.NDArray[np.uint8]:
    """Draw detections on an image using supervision.

    Args:
        image: Image as a numpy array (BGR).
        annotation: DetectionAnnotation object containing detections.

    Returns:
        Image with detections drawn.
    """
    if not annotation.annotations:
        return image

    # Convert our Detection objects to supervision Detections
    xyxy: list[list[float]] = []
    confidence: list[float] = []
    class_id: list[int] = []

    img_h, img_w = image.shape[:2]

    for det in annotation.annotations:
        # Our bboxes are normalized x, y, w, h
        x1 = det.bbox.x * img_w
        y1 = det.bbox.y * img_h
        x2 = (det.bbox.x + det.bbox.w) * img_w
        y2 = (det.bbox.y + det.bbox.h) * img_h

        xyxy.append([x1, y1, x2, y2])
        confidence.append(det.confidence)
        class_id.append(det.class_id)

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id, dtype=int),
    )

    # Setup annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Create labels
    labels = [
        f"{annotation.categories.get(cls_id, str(cls_id))} {conf:.2f}"
        for cls_id, conf in zip(
            detections.class_id,  # type: ignore[arg-type]
            detections.confidence,  # type: ignore[arg-type]
            strict=True,
        )
    ]

    # Annotate
    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=detections,
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels,
    )

    return annotated_image  # type: ignore[no-any-return]


def save_annotated_image(
    image: npt.NDArray[np.uint8],
    annotation: DetectionAnnotation,
    output_path: Path,
) -> None:
    """Draw detections and save the image.

    Args:
        image: Image as a numpy array (BGR).
        annotation: DetectionAnnotation object.
        output_path: Path to save the annotated image.
    """
    annotated_image = draw_detections(image, annotation)
    cv2.imwrite(str(output_path), annotated_image)


def show_detections_interactive(
    image: npt.NDArray[np.uint8],
    annotation: DetectionAnnotation,
    window_name: str = "Debug Detections",
) -> int:
    """Display annotated image in a window and wait for a keypress.

    Args:
        image: Image as a numpy array (BGR).
        annotation: DetectionAnnotation object containing detections.
        window_name: Name for the OpenCV window.

    Returns:
        The key code pressed by the user (ord('q') = 113 to quit).
    """
    annotated_image = draw_detections(image, annotation)

    # Resize for display if image is very large
    max_display = 1280
    h, w = annotated_image.shape[:2]
    if max(h, w) > max_display:
        scale = max_display / max(h, w)
        annotated_image = cv2.resize(  # type: ignore[assignment]
            annotated_image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    cv2.imshow(window_name, annotated_image)
    key: int = cv2.waitKey(0) & 0xFF
    return key
