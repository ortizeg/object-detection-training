"""Torchvision v2 transform pipeline for object detection training.

Custom ``v2.Transform`` subclasses that complement the standard torchvision v2
transforms.  All classes leverage the v2 dispatch mechanism so they
automatically handle ``tv_tensors.BoundingBoxes``, ``tv_tensors.Image``, and
plain ``torch.Tensor`` inputs.

The transforms are designed to be fully parameterizable via Hydra YAML configs.
"""

from object_detection_training.transforms.conversion import (
    NormalizeBoxCoords,
    ToFloat32Tensor,
)
from object_detection_training.transforms.multi_scale_resize import (
    MultiScaleRandomResize,
    MultiScaleResize,
    compute_multi_scale_scales,
)
from object_detection_training.transforms.random_size_crop import RandomSizeCrop

__all__ = [
    "MultiScaleRandomResize",
    "MultiScaleResize",
    "NormalizeBoxCoords",
    "RandomSizeCrop",
    "ToFloat32Tensor",
    "compute_multi_scale_scales",
]
