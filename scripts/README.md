# Scripts

Standalone utility scripts for dataset preparation and debugging. These scripts are **not** part of the installed package — they are run directly with `pixi run python scripts/<name>.py`.

## extract_object_crops.py

Extract masked object crops from a COCO detection dataset using SAM (Segment Anything Model). Produces per-category directories of cropped images with RLE-encoded masks, ready for use with `ObjectInserter`.

### Prerequisites

The script uses SAM1 via HuggingFace `transformers` (already included in project dependencies). The model weights are downloaded automatically on first run.

### Usage

```bash
# Extract all categories
pixi run python scripts/extract_object_crops.py \
    --dataset-path /path/to/train/ \
    --output-dir /path/to/crops/

# Extract specific categories only
pixi run python scripts/extract_object_crops.py \
    --dataset-path /path/to/train/ \
    --output-dir /path/to/crops/ \
    --categories ball rim

# Custom SAM model and padding
pixi run python scripts/extract_object_crops.py \
    --dataset-path /path/to/train/ \
    --output-dir /path/to/crops/ \
    --sam-model facebook/sam-vit-large \
    --padding-ratio 0.15 \
    --min-area 200
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | *(required)* | Root path to dataset split (must contain COCO annotations) |
| `--output-dir` | *(required)* | Directory to save extracted crops |
| `--categories` | all | Space-separated list of categories to extract |
| `--padding-ratio` | `0.1` | Fraction of bbox dimension to add as padding around each crop |
| `--sam-model` | `facebook/sam-vit-base` | HuggingFace SAM model ID |
| `--device` | auto | Device for SAM inference (`cuda`, `mps`, or `cpu`) |
| `--min-area` | `100.0` | Minimum bbox area (in pixels) to include |

### Output Structure

```
output-dir/
  ball/
    12345.png       # cropped image (padded)
    12345.json      # CropMetadata with RLE mask, source info
    12346.png
    12346.json
  player/
    67890.png
    67890.json
```

Each `.json` file is a self-contained `CropMetadata` record with the RLE-encoded binary mask, original bounding box coordinates, source image filename, and category name.

### Pipeline

1. Loads the COCO dataset and prints class distribution
2. For each annotation (filtered by `--categories` and `--min-area`):
   - Pads the bounding box by `--padding-ratio` and clamps to image bounds
   - Crops the region from the source image
   - Runs SAM with a box prompt to generate a precise segmentation mask
   - Selects the highest-confidence mask prediction
   - Saves the crop as PNG and the RLE-encoded mask as JSON
3. Prints extraction summary

## overfit_test_yolox.py

Overfit test for YOLOX model on a small number of samples to verify training convergence.

## verify_yolox_weights.py

Verify YOLOX pretrained weights load correctly and produce expected output shapes.
