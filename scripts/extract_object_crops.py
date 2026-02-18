"""Extract masked object crops from a COCO detection dataset using SAM.

Analyzes class distribution, crops detections with padding, runs SAM to
generate precise masks, and stores crops organized by category.

Usage:
    python scripts/extract_object_crops.py \
        --dataset-path /path/to/train/ \
        --output-dir /path/to/crops/

Each crop is saved as a ``{annotation_id}.png`` + ``{annotation_id}.json`` pair
inside a per-category subfolder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from object_detection_training.data.coco_detection_dataset import (
    COCODetectionDataset,
)
from object_detection_training.data.dataset_stats import DatasetStatistics
from object_detection_training.schemas.crop_manifest import (
    CropMetadata,
    RLEMask,
    encode_rle,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract masked object crops from a COCO detection dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Root path to the dataset split (must contain COCO annotations).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save extracted crops.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Categories to extract (default: all).",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.1,
        help="Padding ratio around bounding boxes (default: 0.1).",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="facebook/sam-vit-base",
        help="HuggingFace SAM model ID (default: facebook/sam-vit-base).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run SAM on (default: auto-detect).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Minimum bbox area to consider (default: 100.0).",
    )
    return parser.parse_args(argv)


def _auto_device() -> str:
    """Select the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pad_and_clamp_bbox(
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    img_w: int,
    img_h: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Pad a bbox and clamp to image bounds. Returns (x1, y1, x2, y2)."""
    pad_w = bbox_w * padding_ratio
    pad_h = bbox_h * padding_ratio

    x1 = max(0, int(bbox_x - pad_w))
    y1 = max(0, int(bbox_y - pad_h))
    x2 = min(img_w, int(bbox_x + bbox_w + pad_w))
    y2 = min(img_h, int(bbox_y + bbox_h + pad_h))

    return x1, y1, x2, y2


def main(argv: list[str] | None = None) -> None:
    """Main entry point for crop extraction."""
    args = _parse_args(argv)

    device = args.device or _auto_device()
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = COCODetectionDataset(root_path=args.dataset_path, split="train")
    stats = DatasetStatistics(dataset)
    dist = stats.class_distribution()
    logger.info(f"Class distribution:\n{dist.to_string()}")

    # Load SAM
    logger.info(f"Loading SAM model: {args.sam_model}")
    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(args.sam_model)
    model = SamModel.from_pretrained(args.sam_model).to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build category filter
    categories_dict = dataset._categories
    if args.categories:
        target_cat_ids = {
            cid for cid, name in categories_dict.items() if name in args.categories
        }
    else:
        target_cat_ids = set(categories_dict.keys())

    # Process annotations
    ann_df = dataset.annotations_df
    total_extracted = 0
    skipped_small = 0

    for _, row in ann_df.iterrows():
        cat_id = int(row["category_id"])
        if cat_id not in target_cat_ids:
            continue

        bbox_w = float(row["bbox_w"])
        bbox_h = float(row["bbox_h"])
        if bbox_w * bbox_h < args.min_area:
            skipped_small += 1
            continue

        cat_name = categories_dict[cat_id]
        ann_id = int(row["annotation_id"])
        image_id = int(row["image_id"])

        # Load image
        img_path = dataset.get_image_path(image_id)
        if img_path is None or not img_path.exists():
            logger.warning(f"Image not found for image_id={image_id}")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # Padded crop
        bbox_x = float(row["bbox_x"])
        bbox_y = float(row["bbox_y"])
        x1, y1, x2, y2 = _pad_and_clamp_bbox(
            bbox_x, bbox_y, bbox_w, bbox_h, img_w, img_h, args.padding_ratio
        )

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        crop_img = img.crop((x1, y1, x2, y2))

        # SAM box prompt (bbox coords relative to crop)
        box_in_crop = [
            bbox_x - x1,
            bbox_y - y1,
            bbox_x - x1 + bbox_w,
            bbox_y - y1 + bbox_h,
        ]

        # Run SAM
        with torch.no_grad():
            inputs = processor(
                images=crop_img,
                input_boxes=[
                    [
                        [
                            box_in_crop[0],
                            box_in_crop[1],
                            box_in_crop[2],
                            box_in_crop[3],
                        ]
                    ]
                ],
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )

        # Take best mask (highest IoU score)
        mask_tensor = masks[0][0]  # (num_masks, H, W)
        scores = outputs.iou_scores[0][0]  # (num_masks,)
        best_idx = scores.argmax().item()
        binary_mask = mask_tensor[best_idx].numpy().astype(bool)

        # Encode and save
        rle = encode_rle(binary_mask)
        img_info = dataset.get_image_info(image_id)
        source_filename = str(img_info["file_name"]) if img_info is not None else ""

        metadata = CropMetadata(
            filename=f"{ann_id}.png",
            mask=RLEMask(counts=rle.counts, height=rle.height, width=rle.width),
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            source_image=source_filename,
            annotation_id=ann_id,
            category_name=cat_name,
        )

        cat_dir = output_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        crop_img.save(cat_dir / f"{ann_id}.png")
        metadata.save_json(cat_dir / f"{ann_id}.json")
        total_extracted += 1

    logger.info(
        f"Extraction complete: {total_extracted} crops extracted, "
        f"{skipped_small} skipped (below min area {args.min_area})"
    )


if __name__ == "__main__":
    main()
