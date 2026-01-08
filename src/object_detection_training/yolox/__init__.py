#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
#
# Original source: https://github.com/Megvii-BaseDetection/YOLOX
# This code is copied for use in this framework.
"""YOLOX model module."""

from .darknet import CSPDarknet
from .network_blocks import (
    BaseConv,
    Bottleneck,
    CSPLayer,
    DWConv,
    Focus,
    ResLayer,
    SPPBottleneck,
)
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX

__all__ = [
    # Main model
    "YOLOX",
    # Components
    "CSPDarknet",
    "YOLOPAFPN",
    "YOLOXHead",
    # Blocks
    "BaseConv",
    "Bottleneck",
    "CSPLayer",
    "DWConv",
    "Focus",
    "ResLayer",
    "SPPBottleneck",
]
