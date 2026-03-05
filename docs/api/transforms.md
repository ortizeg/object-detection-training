# Transforms

Custom `torchvision.transforms.v2.Transform` subclasses for object detection training pipelines.

## ObjectInserter

`ObjectInserter` is a copy-paste augmentation transform that addresses class imbalance by inserting masked object crops of underrepresented classes into training images.

### Workflow

1. **Extract crops** using `scripts/extract_object_crops.py` (runs SAM for precise masks)
2. **Configure the transform** in your training pipeline with a `crops_dir` pointing to the extraction output
3. During training, `ObjectInserter` randomly selects crops, applies affine transforms, and pastes them into images at positions that avoid occluding existing detections

### Usage

```python
from torchvision.transforms import v2
from object_detection_training.transforms import ObjectInserter

pipeline = v2.Compose([
    ObjectInserter(
        crops_dir="/path/to/extracted_crops/",
        category_to_label={"ball": 0, "player": 1},
        p=0.5,                    # 50% chance per image
        max_objects_per_image=2,
        iou_threshold=0.3,        # max overlap with existing boxes
        max_crop_ratio=0.4,       # cap crop size to 40% of image
        category_configs={
            "ball": {"rotation_range": (-180, 180)},
            "player": {"rotation_range": (-10, 10)},
        },
    ),
    v2.RandomHorizontalFlip(p=0.5),
    # ... rest of pipeline
])

# Use in training loop — expects (PIL.Image, target_dict)
out_image, out_target = pipeline(image, target)
```

### Design: Lazy Loading

Crops are **not** loaded into memory at init time. `ObjectInserter` builds a lightweight file index (paths only) during `__init__`, then reads the crop image and decodes the RLE mask from disk on demand during `forward()`. This keeps memory usage constant regardless of how many crops are available.

### Per-Category Configuration

Different object categories may need different augmentation parameters. Use `category_configs` to override `rotation_range` and `scale_range` per category:

- **Ball**: symmetric object, can rotate freely `(-180, 180)`
- **Player**: upright object, limited rotation `(-10, 10)`

## API Reference

::: object_detection_training.transforms
