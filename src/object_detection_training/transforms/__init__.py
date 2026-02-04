"""Torchvision v2 transform pipeline for object detection training.

Custom ``v2.Transform`` subclasses that complement the standard torchvision v2
transforms.  All classes leverage the v2 dispatch mechanism so they
automatically handle ``tv_tensors.BoundingBoxes``, ``tv_tensors.Image``, and
plain ``torch.Tensor`` inputs.

The transforms are designed to be fully parameterizable via Hydra YAML configs.
"""

from object_detection_training.transforms.custom import (
    MultiScaleRandomResize,
    MultiScaleResize,
    NormalizeBoxCoords,
    RandomSizeCrop,
    ToFloat32Tensor,
)

__all__ = [
    "MultiScaleRandomResize",
    "MultiScaleResize",
    "NormalizeBoxCoords",
    "RandomSizeCrop",
    "ToFloat32Tensor",
]
