from __future__ import annotations

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from
https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import pycocotools.mask as coco_mask
import torch
import torch.utils.data
import torchvision

import object_detection_training.models.rfdetr.transforms as T
from object_detection_training.models.rfdetr.collate import collate_fn

__all__ = [
    "CocoDetection",
    "make_coco_transforms",
    "make_coco_transforms_square_div_64",
    "collate_fn",
    "ConvertCoco",
]


def compute_multi_scale_scales(
    resolution, expanded_scales=False, patch_size=16, num_windows=4
):
    # round to the nearest multiple of 4*patch_size to enable both patching
    # and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = (
        [-3, -2, -1, 0, 1, 2, 3, 4]
        if not expanded_scales
        else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    )
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [
        scale for scale in proposed_scales if scale >= patch_size * num_windows * 2
    ]  # ensure minimum image size
    return proposed_scales


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except Exception:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        include_masks=False,
        label_map: dict | None = None,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertCoco(include_masks=include_masks, label_map=label_map)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco:
    def __init__(self, include_masks=False, label_map: dict | None = None):
        self.include_masks = include_masks
        self.label_map = label_map

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        if self.label_map:
            classes = [self.label_map[c] for c in classes]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key
        # when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and "segmentation" in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(
    image_set,
    input_height,
    input_width,
    multi_scale=False,
    expanded_scales=False,
    skip_random_resize=False,
    patch_size=16,
    num_windows=4,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
):
    # Always use PILToTensor (keeps 0-255 range)
    normalize = T.Compose([T.PILToTensor(), T.Normalize(mean, std)])

    scales = [input_height]
    if multi_scale:
        scales = compute_multi_scale_scales(
            input_height, expanded_scales, patch_size, num_windows
        )
        if skip_random_resize:
            scales = [scales[-1]]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set in ["val", "test", "val_speed"]:
        return T.Compose(
            [
                (
                    T.SquareResize([input_height])
                    if image_set == "val_speed"
                    else T.RandomResize([input_height], max_size=1333)
                ),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_square_div_64(
    image_set,
    input_height,
    input_width,
    multi_scale=False,
    expanded_scales=False,
    skip_random_resize=False,
    patch_size=16,
    num_windows=4,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
):
    """ """
    # Always use PILToTensor (keeps 0-255 range)
    normalize = T.Compose([T.PILToTensor(), T.Normalize(mean, std)])

    scales = [input_height]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(
            input_height, expanded_scales, patch_size, num_windows
        )
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.SquareResize(scales),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.SquareResize(scales),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.SquareResize([input_height]),
                normalize,
            ]
        )
    if image_set == "test":
        return T.Compose(
            [
                T.SquareResize([input_height]),
                normalize,
            ]
        )
    if image_set == "val_speed":
        return T.Compose(
            [
                T.SquareResize([input_height]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")
