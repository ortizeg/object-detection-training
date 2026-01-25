#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
#
# Original source: https://github.com/Megvii-BaseDetection/YOLOX
# This code is copied with minimal modifications for use in this framework.
"""YOLOX detection head with decoupled head structure."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_blocks import BaseConv, DWConv


def meshgrid(*tensors):
    """Meshgrid function compatible with older PyTorch versions."""
    return torch.meshgrid(*tensors, indexing="ij")


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate IoU between bboxes."""
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).to(tl.dtype).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUloss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).to(tl.dtype).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOXHead(nn.Module):
    """YOLOX detection head with decoupled classification and regression."""

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            num_classes: Number of classes.
            width: Width multiplier.
            strides: Stride for each feature level.
            in_channels: Input channels for each level.
            act: Activation type.
            depthwise: Whether to use depthwise convs.
        """
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                )
            )
            self.cls_preds.append(
                nn.Conv2d(int(256 * width), self.num_classes, 1, 1, 0)
            )
            self.reg_preds.append(nn.Conv2d(int(256 * width), 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(int(256 * width), 1, 1, 1, 0))

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].dtype
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid(torch.arange(hsize), torch.arange(wsize))
            grid = (
                torch.stack((xv, yv), 2)
                .view(1, 1, hsize, wsize, 2)
                .to(device=output.device, dtype=dtype)
            )
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid(torch.arange(hsize), torch.arange(wsize))
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).to(device=outputs.device, dtype=dtype)
        strides = torch.cat(strides, dim=1).to(device=outputs.device, dtype=dtype)

        outputs = torch.cat(
            [
                (outputs[..., 0:2] + grids) * strides,
                torch.exp(outputs[..., 2:4]) * strides,
                outputs[..., 4:],
            ],
            dim=-1,
        )
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors, n_cls]

        # Calculate standard numer of anchors
        total_num_anchors = outputs.shape[1]

        # Initialize losses
        cls_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        iou_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        obj_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        l1_loss = torch.zeros(1, device=outputs.device, dtype=dtype)

        # Flatten strides and shifts for assignment
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors]
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, n_anchors]

        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = 0
            if labels is not None and len(labels) > batch_idx:
                label = labels[batch_idx]
                if isinstance(label, dict):
                    gt_bboxes_per_image = label.get("boxes", torch.zeros(0, 4))
                    gt_classes = label.get("labels", torch.zeros(0))
                else:
                    gt_bboxes_per_image = torch.zeros(0, 4)
                    gt_classes = torch.zeros(0)

                # Check for empty targets
                if gt_bboxes_per_image.numel() == 0:
                    num_gt = 0
                else:
                    num_gt = gt_bboxes_per_image.shape[0]
            else:
                gt_bboxes_per_image = torch.zeros(0, 4)
                gt_classes = torch.zeros(0)

            if num_gt == 0:
                # No ground truth, all obj targets are 0
                tgt_obj = torch.zeros(
                    total_num_anchors, 1, device=outputs.device, dtype=dtype
                )
                obj_loss += self.bcewithlog_loss(obj_preds[batch_idx], tgt_obj).sum()
                continue

            num_gts += num_gt

            # --- SimOTA Assignment ---
            try:
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    batch_idx,
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bbox_preds[batch_idx],
                    cls_preds[batch_idx],
                    obj_preds[batch_idx],
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
            except Exception:
                # Fallback on assignment error
                tgt_obj = torch.zeros(
                    total_num_anchors, 1, device=outputs.device, dtype=dtype
                )
                obj_loss += self.bcewithlog_loss(obj_preds[batch_idx], tgt_obj).sum()
                continue

            num_fg += num_fg_img

            if num_fg_img > 0:
                # Classification Loss
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)

                cls_loss += self.bcewithlog_loss(
                    cls_preds[batch_idx][fg_mask], cls_target
                ).sum()

                # Regression Loss (IOU)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                iou_loss += self.iou_loss(
                    bbox_preds[batch_idx][fg_mask], reg_target
                ).sum()

                # L1 Loss
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        reg_target,
                        expanded_strides[0][fg_mask],
                        x_shifts[0][fg_mask],
                        y_shifts[0][fg_mask],
                    )
                    l1_loss += self.l1_loss(
                        origin_preds[batch_idx][fg_mask], l1_target
                    ).sum()

            # Objectness Loss
            tgt_obj = torch.zeros(
                total_num_anchors, 1, device=outputs.device, dtype=dtype
            )
            if num_fg_img > 0:
                tgt_obj[fg_mask] = 1.0

            obj_loss += self.bcewithlog_loss(obj_preds[batch_idx], tgt_obj).sum()

        # Normalize losses
        num_fg = max(num_fg, 1)
        cls_loss = cls_loss / num_fg
        iou_loss = iou_loss / num_fg
        obj_loss = obj_loss / num_fg
        l1_loss = l1_loss / num_fg

        # Loss weights (defaults)
        reg_weight = 5.0
        loss = reg_weight * iou_loss + obj_loss + cls_loss + l1_loss

        return (
            loss,
            reg_weight * iou_loss,
            obj_loss,
            cls_loss,
            l1_loss,
            num_fg,
        )

    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bbox_preds,
        cls_preds,
        obj_preds,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        # Place GTs to appropriate device
        gt_bboxes_per_image = gt_bboxes_per_image.to(bbox_preds.device)
        gt_classes = gt_classes.to(bbox_preds.device)

        # 1. Filter anchors: select candidates within GT boxes
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bbox_preds = bbox_preds[fg_mask]
        cls_preds = cls_preds[fg_mask]
        obj_preds = obj_preds[fg_mask]
        num_in_boxes_anchor = bbox_preds.shape[0]

        if num_in_boxes_anchor == 0:
            # Fallback if no anchor inside GT
            return (
                torch.zeros(0, device=gt_classes.device),
                torch.zeros(
                    total_num_anchors, dtype=torch.bool, device=gt_classes.device
                ),
                torch.zeros(0, device=gt_classes.device),
                torch.zeros(0, device=gt_classes.device),
                0,
            )

        # 2. Compute Cost Matrix
        # Pairwise IOU
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bbox_preds, xyxy=False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # Pairwise Classification Ccost
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)

        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~is_in_boxes_and_center)
        )

        # 3. Dynamic K Matching
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask,  # Captured fg_mask from dynamic_k_matching call
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # b_l = x_centers_per_image - gt_bboxes_per_image_l
        # b_r = gt_bboxes_per_image_r - x_centers_per_image
        # b_t = y_centers_per_image - gt_bboxes_per_image_t
        # b_b = gt_bboxes_per_image_b - y_centers_per_image
        # bbox_deltas = torch.stack([b_l, b_b, b_r, b_t], 2)

        # is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        # is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # Center sampling (matches official YOLOX)
        center_radius = 1.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_b, c_r, c_t], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # In boxes AND in centers (simplified to match official YOLOX)
        # Use center-based filtering only for anchor selection
        is_in_boxes_anchor = is_in_centers_all

        is_in_boxes_and_center = is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            # If an anchor matched to multiple GTs, pick the one with min cost
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask_new = fg_mask.clone()
        # Update full mask
        # fg_mask has length total_anchors, only True where we pre-selected
        # We need to map back to full indices
        fg_idxs = torch.nonzero(fg_mask, as_tuple=True)[0]
        fg_mask_new[fg_idxs[~fg_mask_inboxes]] = False
        fg_mask = fg_mask_new

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask,
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target
