"""Debug script: run OmDet-Turbo on one image and save annotated result."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from object_detection_training.inference.omdet_turbo_inferencer import (
    OmDetTurboInferencer,
)
from object_detection_training.schemas.detection import Detection

# COCO-aligned classes (same as eval task builders)
CLASSES = ["person", "sports ball", "referee", "basketball hoop", "jersey number"]

# Eval label map for display
DISPLAY_NAMES = {
    0: "player",
    1: "ball",
    2: "referee",
    3: "rim",
    4: "number",
}

# COCO name -> eval ID (for remapping)
NAME_TO_EVAL_ID = {
    "person": 0,
    "sports ball": 1,
    "referee": 2,
    "basketball hoop": 3,
    "jersey number": 4,
}


def filter_area_outliers(
    detections: list[Detection], max_area_fraction: float = 0.05
) -> list[Detection]:
    """Remove boxes exceeding max_area_fraction of image area."""
    filtered = [d for d in detections if d.bbox.w * d.bbox.h <= max_area_fraction]
    print(
        f"Area filter: {len(detections)} -> {len(filtered)} "
        f"(removed {len(detections) - len(filtered)}, max={max_area_fraction:.0%})"
    )
    return filtered


def run_debug(image_path: str, output_path: str = "omdet_debug.png") -> None:
    """Run OmDet-Turbo on a single image and save annotated output."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image: {Path(image_path).name} ({w}x{h})")

    print("Loading OmDet-Turbo...")
    inferencer = OmDetTurboInferencer(
        model_name="omlab/omdet-turbo-swin-tiny-hf",
        classes=CLASSES,
        box_threshold=0.20,
        device="cpu",
    )

    print("Running inference...")
    detections = inferencer.predict(img, image_width=w, image_height=h)
    print(f"Raw detections: {len(detections)}")

    # Remap to eval IDs
    remapped: list[Detection] = []
    for det in detections:
        class_name = CLASSES[det.class_id]
        eval_id = NAME_TO_EVAL_ID.get(class_name)
        if eval_id is None:
            continue
        remapped.append(
            Detection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=eval_id,
            )
        )

    # Apply area filter
    filtered = filter_area_outliers(remapped)

    # Convert to supervision Detections
    xyxy = []
    confs = []
    class_ids = []
    labels = []

    for det in filtered:
        x1 = det.bbox.x * w
        y1 = det.bbox.y * h
        x2 = (det.bbox.x + det.bbox.w) * w
        y2 = (det.bbox.y + det.bbox.h) * h

        xyxy.append([x1, y1, x2, y2])
        confs.append(det.confidence)
        class_ids.append(det.class_id)
        labels.append(f"{DISPLAY_NAMES[det.class_id]} {det.confidence:.2f}")

    print(f"Final detections: {len(xyxy)}")
    for label in labels:
        print(f"  - {label}")

    if xyxy:
        sv_dets = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )

        annotated = img.copy()
        annotated = sv.BoxAnnotator(thickness=2).annotate(
            scene=annotated, detections=sv_dets
        )
        annotated = sv.LabelAnnotator(text_scale=0.5, text_padding=4).annotate(
            scene=annotated, detections=sv_dets, labels=labels
        )
    else:
        annotated = img.copy()
        print("No detections!")

    cv2.imwrite(output_path, annotated)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_omdet_turbo.py <image_path> [output_path]")
        sys.exit(1)

    out = sys.argv[2] if len(sys.argv) > 2 else "omdet_debug.png"
    run_debug(sys.argv[1], out)
