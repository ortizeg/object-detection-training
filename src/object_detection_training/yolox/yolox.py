#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
#
# Original source: https://github.com/Megvii-BaseDetection/YOLOX
# This code is copied with minimal modifications for use in this framework.
"""YOLOX main model class."""

import torch.nn as nn

from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module.

    The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            # Import here to avoid circular import
            from .yolo_head import YOLOXHead

            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if targets is not None:
            # Head's forward checks self.training to decide whether to compute losses
            # During validation (model.eval()), we need to temporarily set head to
            # train mode to compute losses for metric computation
            was_training = self.head.training
            self.head.train()
            try:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
            finally:
                # Restore original training state
                if not was_training:
                    self.head.eval()

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            # If targets are provided but we're in eval mode (e.g. validation),
            # we also want predictions for metric computation
            if not self.training:
                # Get predictions in eval mode
                outputs["predictions"] = self.head(fpn_outs)
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
