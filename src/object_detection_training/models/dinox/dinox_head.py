"""DINOX detection head with optional Distribution Focal Loss regression.

A fresh nn.Module implementation (not a subclass of YOLOXHead) that follows
the same decoupled head pattern. When use_dfl=False, the architecture is
functionally equivalent to YOLOXHead. When use_dfl=True, regression outputs
are 4*(reg_max+1) distribution logits converted to continuous LTRB via
DFLModule.

SimOTA assignment logic is self-contained (copied from YOLOXHead) to avoid
depending on third-party internal methods.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from object_detection_training.models.yolox.network_blocks import BaseConv, DWConv

from .dfl import DFLModule, distribution_focal_loss
from .hungarian import HungarianAssigner
from .mal import mal_weight, matchability_score
from .tal import TaskAlignedAssigner

# ---------------------------------------------------------------------------
# Utility helpers (copied from yolo_head.py to keep self-contained)
# ---------------------------------------------------------------------------


def _meshgrid(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Meshgrid compatible with ij indexing."""
    return torch.meshgrid(*tensors, indexing="ij")


def _bboxes_iou(
    bboxes_a: torch.Tensor,
    bboxes_b: torch.Tensor,
    xyxy: bool = True,
) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes."""
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError("Boxes must have 4 columns")

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


# ---------------------------------------------------------------------------
# IoU loss (same as YOLOXHead)
# ---------------------------------------------------------------------------


class _IOULoss(nn.Module):
    """IoU loss for bounding box regression (cxcywh format)."""

    def __init__(
        self,
        reduction: str = "none",
        loss_type: str = "iou",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] != target.shape[0]:
            msg = (
                f"pred and target batch size mismatch: "
                f"{pred.shape[0]} vs {target.shape[0]}"
            )
            raise ValueError(msg)
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2),
            (target[:, :2] - target[:, 2:] / 2),
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2),
            (target[:, :2] + target[:, 2:] / 2),
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).to(tl.dtype).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2),
                (target[:, :2] - target[:, 2:] / 2),
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2),
                (target[:, :2] + target[:, 2:] / 2),
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            loss = 1 - iou**2

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return torch.as_tensor(loss)


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------


def _ltrb_to_cxcywh(
    ltrb: torch.Tensor,
    grid: torch.Tensor,
    stride: torch.Tensor | float,
) -> torch.Tensor:
    """Convert DFL LTRB distances to pixel cxcywh boxes.

    Args:
        ltrb: LTRB distances in stride units, shape [N, 4] (left, top, right, bottom).
        grid: Anchor grid positions (x, y) in grid units, shape [N, 2].
        stride: Stride value(s), scalar or [N, 1].

    Returns:
        Boxes in cxcywh pixel format, shape [N, 4].
    """
    # Grid is in grid-cell units; anchor center in pixels = (grid + 0.5) * stride
    if isinstance(stride, (int, float)):
        anchor_x = (grid[:, 0] + 0.5) * stride
        anchor_y = (grid[:, 1] + 0.5) * stride
        left = ltrb[:, 0] * stride
        top = ltrb[:, 1] * stride
        right = ltrb[:, 2] * stride
        bottom = ltrb[:, 3] * stride
    else:
        s = stride.squeeze(-1) if stride.dim() > 1 else stride
        anchor_x = (grid[:, 0] + 0.5) * s
        anchor_y = (grid[:, 1] + 0.5) * s
        left = ltrb[:, 0] * s
        top = ltrb[:, 1] * s
        right = ltrb[:, 2] * s
        bottom = ltrb[:, 3] * s

    cx = anchor_x - left + (left + right) / 2
    cy = anchor_y - top + (top + bottom) / 2
    w = left + right
    h = top + bottom

    return torch.stack([cx, cy, w, h], dim=-1)


def _cxcywh_to_ltrb_target(
    gt_cxcywh: torch.Tensor,
    grid: torch.Tensor,
    stride: torch.Tensor | float,
    reg_max: int,
) -> torch.Tensor:
    """Convert GT cxcywh boxes to DFL LTRB targets in stride units.

    Args:
        gt_cxcywh: GT boxes in cxcywh pixel format, shape [N, 4].
        grid: Anchor grid positions (x, y) in grid units, shape [N, 2].
        stride: Stride value(s), scalar or [N, 1].
        reg_max: Maximum reg value for clamping.

    Returns:
        LTRB targets in stride units, clamped to [0, reg_max - 0.01], shape [N, 4].
    """
    if isinstance(stride, (int, float)):
        anchor_x = (grid[:, 0] + 0.5) * stride
        anchor_y = (grid[:, 1] + 0.5) * stride
    else:
        s = stride.squeeze(-1) if stride.dim() > 1 else stride
        anchor_x = (grid[:, 0] + 0.5) * s
        anchor_y = (grid[:, 1] + 0.5) * s

    cx, cy, w, h = gt_cxcywh[:, 0], gt_cxcywh[:, 1], gt_cxcywh[:, 2], gt_cxcywh[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    if isinstance(stride, (int, float)):
        left = (anchor_x - x1) / stride
        top = (anchor_y - y1) / stride
        right = (x2 - anchor_x) / stride
        bottom = (y2 - anchor_y) / stride
    else:
        s = stride.squeeze(-1) if stride.dim() > 1 else stride
        left = (anchor_x - x1) / s
        top = (anchor_y - y1) / s
        right = (x2 - anchor_x) / s
        bottom = (y2 - anchor_y) / s

    ltrb = torch.stack([left, top, right, bottom], dim=-1)
    return ltrb.clamp(min=0, max=reg_max - 0.01)


# ---------------------------------------------------------------------------
# DINOXHead
# ---------------------------------------------------------------------------


class DINOXHead(nn.Module):
    """DINOX detection head with optional DFL regression.

    Follows the same decoupled head pattern as YOLOXHead. When use_dfl=False,
    the architecture and output format are equivalent to YOLOXHead.

    Args:
        num_classes: Number of object classes.
        width: Channel width multiplier (0.75 for M-size).
        strides: Feature map strides for each FPN level.
        in_channels: Input channel counts for each FPN level.
        act: Activation function name.
        depthwise: Use depthwise separable convolutions.
        use_dfl: Enable DFL regression.
        reg_max: Max bin index for DFL distributions.
        dfl_loss_weight: Weight for DFL loss term.
        iou_loss_type: IoU loss variant ("iou" or "giou").
        use_soft_labels: Enable soft label assignment targets (IoU^gamma weighting).
        soft_label_gamma: Gamma exponent for soft label quality weighting.
        use_log_iou_cost: Enable -log(IoU) regression cost in SimOTA assignment.
        use_mal: Enable Matchability-Aware Loss weighting on classification loss.
        mal_gamma: Gamma for MAL matchability score balance.
        assigner_type: Label assignment strategy ("simota" or "tal").
        tal_topk: Top-k candidates per GT for TAL.
        tal_alpha: Classification exponent for TAL alignment metric.
        tal_beta: IoU exponent for TAL alignment metric.
        use_dual_head: Enable O2O dual head for NMS-free inference.
        lambda_o2o: Weight for O2O loss contribution to combined loss.
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        strides: list[int] | None = None,
        in_channels: list[int] | None = None,
        act: str = "silu",
        depthwise: bool = False,
        use_dfl: bool = False,
        reg_max: int = 16,
        dfl_loss_weight: float = 0.25,
        iou_loss_type: str = "iou",
        use_soft_labels: bool = False,
        soft_label_gamma: float = 2.0,
        use_log_iou_cost: bool = False,
        use_mal: bool = False,
        mal_gamma: float = 1.5,
        assigner_type: str = "simota",
        tal_topk: int = 13,
        tal_alpha: float = 1.0,
        tal_beta: float = 6.0,
        use_dual_head: bool = False,
        lambda_o2o: float = 1.0,
    ) -> None:
        super().__init__()

        if strides is None:
            strides = [8, 16, 32]
        if in_channels is None:
            in_channels = [256, 512, 1024]

        self.num_classes = num_classes
        self.use_dfl = use_dfl
        self.use_soft_labels = use_soft_labels
        self.soft_label_gamma = soft_label_gamma
        self.use_log_iou_cost = use_log_iou_cost
        self.reg_max = reg_max
        self.dfl_loss_weight = dfl_loss_weight
        self.strides = strides
        self.use_mal = use_mal
        self.mal_gamma = mal_gamma
        self.assigner_type = assigner_type
        self.use_dual_head = use_dual_head
        self.lambda_o2o = lambda_o2o

        # Number of regression output channels
        reg_channels = 4 * (reg_max + 1) if use_dfl else 4

        # Build per-level layers (same pattern as YOLOXHead)
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        conv_block: type[DWConv] | type[BaseConv] = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(  # type: ignore[no-untyped-call]
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    conv_block(int(256 * width), int(256 * width), 3, 1, act=act),
                    conv_block(int(256 * width), int(256 * width), 3, 1, act=act),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    conv_block(int(256 * width), int(256 * width), 3, 1, act=act),
                    conv_block(int(256 * width), int(256 * width), 3, 1, act=act),
                )
            )
            self.cls_preds.append(
                nn.Conv2d(int(256 * width), self.num_classes, 1, 1, 0)
            )
            self.reg_preds.append(nn.Conv2d(int(256 * width), reg_channels, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(int(256 * width), 1, 1, 1, 0))

        # DFL module for distribution-to-point conversion
        if use_dfl:
            self.dfl = DFLModule(reg_max)

        # TAL assigner (instantiated only when selected)
        if assigner_type == "tal":
            self._tal_assigner = TaskAlignedAssigner(
                topk=tal_topk,
                alpha=tal_alpha,
                beta=tal_beta,
                num_classes=num_classes,
            )

        # O2O dual head: separate prediction layers, shared conv stacks
        if use_dual_head:
            self.cls_preds_o2o = nn.ModuleList()
            self.reg_preds_o2o = nn.ModuleList()
            for _i in range(len(in_channels)):
                self.cls_preds_o2o.append(
                    nn.Conv2d(int(256 * width), self.num_classes, 1, 1, 0)
                )
                self.reg_preds_o2o.append(
                    nn.Conv2d(int(256 * width), reg_channels, 1, 1, 0)
                )
            # Initialize O2O weights by copying from O2M for better convergence
            for i in range(len(in_channels)):
                self.cls_preds_o2o[i].load_state_dict(self.cls_preds[i].state_dict())
                self.reg_preds_o2o[i].load_state_dict(self.reg_preds[i].state_dict())
            self._hungarian_assigner = HungarianAssigner(
                alpha=tal_alpha,
                beta=tal_beta,
                num_classes=num_classes,
            )

        # Loss functions
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = _IOULoss(reduction="none", loss_type=iou_loss_type)

        # Grid cache (training path — _get_output_and_grid)
        self.grids: list[torch.Tensor] = [torch.zeros(1)] * len(in_channels)
        # DFL anchor center cache (training path — _get_output_and_grid)
        # Keyed by FPN level k → (anchor_x, anchor_y) in pixel coords
        self._anchor_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Grid/stride cache (inference path — _decode_outputs)
        # Keyed by (hsize, wsize, stride) → (grid, strides) tensors
        self._decode_grid_cache: dict[
            tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        logger.debug(
            f"DINOXHead: num_classes={num_classes}, use_dfl={use_dfl}, "
            f"reg_max={reg_max}, reg_channels={reg_channels}"
        )

    def initialize_biases(self, prior_prob: float) -> None:
        """Initialize classification and objectness biases."""
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(bias_val)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(bias_val)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(
        self,
        xin: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            float,
        ]
    ):
        """Forward pass.

        Uses ``targets is not None`` (not self.training) to branch between
        training and inference, ensuring clean ONNX tracing.

        Args:
            xin: FPN feature maps, one per stride level.
            targets: Training targets (list of dicts with "boxes", "labels").
                When provided, compute and return losses.
            labels: Alias for targets (compatibility with YOLOX call convention).

        Returns:
            Inference: [B, N, 5+num_classes] decoded predictions.
            Training: Tuple (total_loss, iou_loss, conf_loss, cls_loss,
                l1_loss, num_fg).
        """
        # Allow labels as alias for targets (YOLOX compat)
        if targets is None and labels is not None:
            targets = labels

        if targets is not None:
            return self._forward_train(xin, targets)
        else:
            return self._forward_inference(xin)

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------

    def _forward_inference(self, xin: list[torch.Tensor]) -> torch.Tensor:
        """Inference forward: decode predictions to [B, N, 5+C]."""
        outputs: list[torch.Tensor] = []
        hw_sizes: list[tuple[int, int]] = []

        for k, (cls_conv, reg_conv, _stride_val, x) in enumerate(
            zip(
                self.cls_convs,
                self.reg_convs,
                self.strides,
                xin,
                strict=True,
            )
        ):
            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            reg_feat = reg_conv(x)

            if self.use_dual_head:
                # O2O inference: use O2O prediction layers, constant-1 objectness
                cls_output = self.cls_preds_o2o[k](cls_feat)
                reg_output = self.reg_preds_o2o[k](reg_feat)
                obj_output = torch.ones(
                    reg_output.shape[0],
                    1,
                    reg_output.shape[2],
                    reg_output.shape[3],
                    device=reg_output.device,
                    dtype=reg_output.dtype,
                )
            else:
                cls_output = self.cls_preds[k](cls_feat)
                reg_output = self.reg_preds[k](reg_feat)
                obj_output = self.obj_preds[k](reg_feat).sigmoid()

            # [B, C, H, W] -> concat [reg, obj, cls_sigmoid]
            output = torch.cat([reg_output, obj_output, cls_output.sigmoid()], 1)
            hw_sizes.append((output.shape[-2], output.shape[-1]))
            outputs.append(output)

        # Flatten spatial dims and concat across FPN levels
        flat = torch.cat([o.flatten(start_dim=2) for o in outputs], dim=2).permute(
            0, 2, 1
        )  # [B, N, C]

        return self._decode_outputs(flat, hw_sizes, dtype=xin[0].dtype)

    def _decode_outputs(
        self,
        outputs: torch.Tensor,
        hw_sizes: list[tuple[int, int]],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decode raw flattened outputs to [B, N, 5+C].

        For non-DFL: standard exp-based decode (same as YOLOX).
        For DFL: DFLModule integral + LTRB-to-CXCYWH conversion.
        """
        grids: list[torch.Tensor] = []
        strides_list: list[torch.Tensor] = []

        for (hsize, wsize), stride_val in zip(hw_sizes, self.strides, strict=True):
            cache_key = (hsize, wsize, stride_val)
            if cache_key in self._decode_grid_cache:
                grid, stride_t = self._decode_grid_cache[cache_key]
            else:
                yv, xv = _meshgrid(torch.arange(hsize), torch.arange(wsize))
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                stride_t = torch.full((*grid.shape[:2], 1), stride_val)
                self._decode_grid_cache[cache_key] = (grid, stride_t)
            grids.append(grid)
            strides_list.append(stride_t)

        grids_cat = torch.cat(grids, dim=1).to(device=outputs.device, dtype=dtype)
        strides_cat = torch.cat(strides_list, dim=1).to(
            device=outputs.device, dtype=dtype
        )

        if self.use_dfl:
            reg_channels = 4 * (self.reg_max + 1)
            reg_raw = outputs[..., :reg_channels]
            obj_cls = outputs[..., reg_channels:]

            # DFL integral: [B, N, 4*(reg_max+1)] -> [B, N, 4] LTRB in stride units
            ltrb = self.dfl(reg_raw)

            # LTRB to CXCYWH in pixel coords
            # grids_cat: [1, N, 2], strides_cat: [1, N, 1]
            batch_size = outputs.shape[0]
            # Expand grid/stride for batch
            g = grids_cat.expand(batch_size, -1, -1)
            s = strides_cat.expand(batch_size, -1, -1)

            # anchor center in pixels
            anchor_x = (g[..., 0] + 0.5) * s[..., 0]
            anchor_y = (g[..., 1] + 0.5) * s[..., 0]

            left = ltrb[..., 0] * s[..., 0]
            top = ltrb[..., 1] * s[..., 0]
            right = ltrb[..., 2] * s[..., 0]
            bottom = ltrb[..., 3] * s[..., 0]

            cx = anchor_x - left + (left + right) / 2
            cy = anchor_y - top + (top + bottom) / 2
            w = left + right
            h = top + bottom

            decoded = torch.cat(
                [
                    cx.unsqueeze(-1),
                    cy.unsqueeze(-1),
                    w.unsqueeze(-1),
                    h.unsqueeze(-1),
                    obj_cls,
                ],
                dim=-1,
            )
        else:
            decoded = torch.cat(
                [
                    (outputs[..., 0:2] + grids_cat) * strides_cat,
                    torch.exp(outputs[..., 2:4]) * strides_cat,
                    outputs[..., 4:],
                ],
                dim=-1,
            )

        return decoded

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def _forward_train(
        self,
        xin: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        """Training forward: compute losses."""
        outputs: list[torch.Tensor] = []
        o2o_outputs_list: list[torch.Tensor] = []
        origin_preds: list[torch.Tensor] = []
        x_shifts: list[torch.Tensor] = []
        y_shifts: list[torch.Tensor] = []
        expanded_strides: list[torch.Tensor] = []

        for k, (cls_conv, reg_conv, stride_val, x) in enumerate(
            zip(
                self.cls_convs,
                self.reg_convs,
                self.strides,
                xin,
                strict=True,
            )
        ):
            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # Concat raw outputs: [B, reg_ch + 1 + C, H, W]
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self._get_output_and_grid(
                output, k, stride_val, xin[0].dtype
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(stride_val).type_as(xin[0])
            )
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                if self.use_dfl:
                    # For DFL, origin_preds store raw distribution logits
                    reg_flat = reg_output.view(batch_size, 1, -1, hsize, wsize)
                    reg_flat = reg_flat.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4 * (self.reg_max + 1)
                    )
                else:
                    reg_flat = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_flat = reg_flat.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                origin_preds.append(reg_flat.clone())

            outputs.append(output)

            # O2O predictions: use separate prediction layers on shared features
            if self.use_dual_head:
                o2o_cls = self.cls_preds_o2o[k](cls_feat)
                o2o_reg = self.reg_preds_o2o[k](reg_feat)
                # No objectness for O2O; use placeholder zeros for concat format
                o2o_obj = torch.zeros_like(obj_output)
                o2o_out = torch.cat([o2o_reg, o2o_obj, o2o_cls], 1)
                o2o_out, _ = self._get_output_and_grid(
                    o2o_out, k, stride_val, xin[0].dtype
                )
                o2o_outputs_list.append(o2o_out)

        o2o_outputs: torch.Tensor | None = None
        if self.use_dual_head and o2o_outputs_list:
            o2o_outputs = torch.cat(o2o_outputs_list, 1)

        return self._get_losses(
            x_shifts,
            y_shifts,
            expanded_strides,
            targets,
            torch.cat(outputs, 1),
            origin_preds,
            dtype=xin[0].dtype,
            o2o_outputs=o2o_outputs,
        )

    def _get_output_and_grid(
        self,
        output: torch.Tensor,
        k: int,
        stride: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape FPN output and generate grid."""
        grid = self.grids[k]
        batch_size = output.shape[0]
        reg_ch = 4 * (self.reg_max + 1) if self.use_dfl else 4
        n_ch = reg_ch + 1 + self.num_classes
        hsize, wsize = output.shape[-2:]

        grid_changed = grid.shape[2:4] != output.shape[2:4]
        if grid_changed:
            yv, xv = _meshgrid(torch.arange(hsize), torch.arange(wsize))
            grid = (
                torch.stack((xv, yv), 2)
                .view(1, 1, hsize, wsize, 2)
                .to(device=output.device, dtype=dtype)
            )
            self.grids[k] = grid
            # Invalidate anchor cache for this level
            self._anchor_cache.pop(k, None)

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)

        if self.use_dfl:
            # For DFL mode during training: decode distribution to CXCYWH
            # for SimOTA assignment (which needs decoded boxes for IoU)
            reg_raw = output[..., :reg_ch]
            rest = output[..., reg_ch:]

            # DFL integral: [B, N, 4*(reg_max+1)] -> [B, N, 4] LTRB in stride units
            ltrb = self.dfl(reg_raw)

            # Convert LTRB (in stride units) to CXCYWH (in pixel units)
            # Anchor centers only depend on grid+stride, so cache them
            if k in self._anchor_cache:
                anchor_x, anchor_y = self._anchor_cache[k]
            else:
                anchor_x = (grid[..., 0] + 0.5) * stride
                anchor_y = (grid[..., 1] + 0.5) * stride
                self._anchor_cache[k] = (anchor_x, anchor_y)

            left = ltrb[..., 0] * stride
            top = ltrb[..., 1] * stride
            right = ltrb[..., 2] * stride
            bottom = ltrb[..., 3] * stride

            cx = anchor_x - left + (left + right) / 2
            cy = anchor_y - top + (top + bottom) / 2
            w = left + right
            h = top + bottom

            decoded_bbox = torch.stack([cx, cy, w, h], dim=-1)
            output = torch.cat([decoded_bbox, rest], dim=-1)
        else:
            # Standard YOLOX decode: grid offset + exp scale
            output[..., :2] = (output[..., :2] + grid) * stride
            output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, grid

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _get_losses(
        self,
        x_shifts: list[torch.Tensor],
        y_shifts: list[torch.Tensor],
        expanded_strides: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        outputs: torch.Tensor,
        origin_preds: list[torch.Tensor],
        dtype: torch.dtype,
        o2o_outputs: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        """Compute all losses using SimOTA/TAL assignment.

        When ``o2o_outputs`` is provided (dual head), also computes O2O loss
        using Hungarian matching and adds it to the total loss.
        """
        # After _get_output_and_grid, bbox is always decoded to cxcywh pixels
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]

        cls_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        iou_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        obj_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        l1_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        dfl_loss = torch.zeros(1, device=outputs.device, dtype=dtype)

        x_shifts_cat = torch.cat(x_shifts, 1)
        y_shifts_cat = torch.cat(y_shifts, 1)
        expanded_strides_cat = torch.cat(expanded_strides, 1)

        if self.use_l1:
            origin_preds_cat = torch.cat(origin_preds, 1)

        num_fg = 0.0

        # Pre-allocate objectness target (reused per-image, zeroed in-place)
        tgt_obj = torch.zeros(total_num_anchors, 1, device=outputs.device, dtype=dtype)

        for batch_idx in range(outputs.shape[0]):
            num_gt = 0
            if targets is not None and len(targets) > batch_idx:
                target = targets[batch_idx]
                if isinstance(target, dict):
                    gt_bboxes = target.get(
                        "boxes", torch.zeros(0, 4, device=outputs.device)
                    )
                    gt_classes = target.get(
                        "labels", torch.zeros(0, device=outputs.device)
                    )
                else:
                    gt_bboxes = torch.zeros(0, 4, device=outputs.device)
                    gt_classes = torch.zeros(0, device=outputs.device)

                num_gt = 0 if gt_bboxes.numel() == 0 else gt_bboxes.shape[0]
            else:
                gt_bboxes = torch.zeros(0, 4, device=outputs.device)
                gt_classes = torch.zeros(0, device=outputs.device)

            if num_gt == 0:
                tgt_obj.zero_()
                obj_loss += self.bcewithlog_loss(obj_preds[batch_idx], tgt_obj).sum()
                continue

            try:
                assigner_args = (
                    batch_idx,
                    num_gt,
                    total_num_anchors,
                    gt_bboxes,
                    gt_classes,
                    bbox_preds[batch_idx],
                    cls_preds[batch_idx],
                    obj_preds[batch_idx],
                    expanded_strides_cat,
                    x_shifts_cat,
                    y_shifts_cat,
                )
                if self.assigner_type == "tal":
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self._tal_assigner.assign(*assigner_args)
                else:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self._get_assignments(*assigner_args)
            except Exception:
                tgt_obj.zero_()
                obj_loss += self.bcewithlog_loss(obj_preds[batch_idx], tgt_obj).sum()
                continue

            num_fg += num_fg_img

            if num_fg_img > 0:
                # Classification loss
                if self.use_soft_labels:
                    iou_weight = pred_ious_this_matching.pow(self.soft_label_gamma)
                else:
                    iou_weight = pred_ious_this_matching
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * iou_weight.unsqueeze(-1)

                cls_loss_raw = self.bcewithlog_loss(
                    cls_preds[batch_idx][fg_mask], cls_target
                )
                if self.use_mal:
                    cls_sigmoid = cls_preds[batch_idx][fg_mask].sigmoid()
                    matched_cls_idx = gt_matched_classes.to(torch.int64)
                    cls_score_for_gt = cls_sigmoid[
                        torch.arange(len(matched_cls_idx), device=cls_sigmoid.device),
                        matched_cls_idx,
                    ]
                    m = matchability_score(
                        pred_ious_this_matching, cls_score_for_gt, self.mal_gamma
                    )
                    w = mal_weight(m)
                    cls_loss += (cls_loss_raw * w.unsqueeze(-1)).sum()
                else:
                    cls_loss += cls_loss_raw.sum()

                # Regression loss (IoU on decoded cxcywh)
                reg_target = gt_bboxes[matched_gt_inds]
                iou_loss += self.iou_loss(
                    bbox_preds[batch_idx][fg_mask], reg_target
                ).sum()

                # DFL loss on raw distributions
                if self.use_dfl and self.use_l1:
                    # origin_preds stores raw reg logits [B, N, 4*(reg_max+1)]
                    fg_reg_raw = origin_preds_cat[batch_idx][fg_mask]
                    # Convert GT to LTRB targets
                    fg_grid = torch.stack(
                        [
                            x_shifts_cat[0][fg_mask],
                            y_shifts_cat[0][fg_mask],
                        ],
                        dim=-1,
                    )
                    fg_stride = expanded_strides_cat[0][fg_mask]
                    ltrb_targets = _cxcywh_to_ltrb_target(
                        reg_target, fg_grid, fg_stride, self.reg_max
                    )
                    # Reshape for distribution_focal_loss: [N*4, reg_max+1]
                    pred_dfl = fg_reg_raw.reshape(-1, self.reg_max + 1)
                    target_dfl = ltrb_targets.reshape(-1)
                    dfl_loss += distribution_focal_loss(pred_dfl, target_dfl).sum()
                elif self.use_dfl:
                    # Compute DFL loss even without L1 (we always need it
                    # when use_dfl=True). Grab raw preds from origin_preds
                    # or recompute from the stored raw outputs.
                    # Since origin_preds may not be populated without use_l1,
                    # we handle DFL loss via a separate mechanism below.
                    pass

                # L1 loss (standard YOLOX L1 on decoded cxcywh)
                if self.use_l1 and not self.use_dfl:
                    l1_target = self._get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        reg_target,
                        expanded_strides_cat[0][fg_mask],
                        x_shifts_cat[0][fg_mask],
                        y_shifts_cat[0][fg_mask],
                    )
                    l1_loss += self.l1_loss(
                        origin_preds_cat[batch_idx][fg_mask], l1_target
                    ).sum()

            # Objectness loss (reuse pre-allocated tgt_obj)
            tgt_obj.zero_()
            if num_fg_img > 0:
                tgt_obj[fg_mask] = 1.0
            obj_loss += self.bcewithlog_loss(obj_preds[batch_idx], tgt_obj).sum()

        # --- O2O dual head loss ---
        o2o_cls_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        o2o_iou_loss = torch.zeros(1, device=outputs.device, dtype=dtype)
        num_fg_o2o = 0.0

        if self.use_dual_head and o2o_outputs is not None:
            o2o_bbox_preds = o2o_outputs[:, :, :4]
            o2o_cls_preds = o2o_outputs[:, :, 5:]
            # o2o_outputs[:, :, 4:5] is placeholder zeros (no objectness)

            for batch_idx in range(outputs.shape[0]):
                num_gt = 0
                if targets is not None and len(targets) > batch_idx:
                    target = targets[batch_idx]
                    if isinstance(target, dict):
                        gt_bboxes = target.get(
                            "boxes", torch.zeros(0, 4, device=outputs.device)
                        )
                        gt_classes = target.get(
                            "labels", torch.zeros(0, device=outputs.device)
                        )
                    else:
                        gt_bboxes = torch.zeros(0, 4, device=outputs.device)
                        gt_classes = torch.zeros(0, device=outputs.device)

                    num_gt = 0 if gt_bboxes.numel() == 0 else gt_bboxes.shape[0]
                else:
                    gt_bboxes = torch.zeros(0, 4, device=outputs.device)
                    gt_classes = torch.zeros(0, device=outputs.device)

                if num_gt == 0:
                    continue

                try:
                    (
                        o2o_gt_classes,
                        o2o_fg_mask,
                        o2o_pred_ious,
                        o2o_matched_gt_inds,
                        o2o_num_fg_img,
                    ) = self._hungarian_assigner.assign(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes,
                        gt_classes,
                        o2o_bbox_preds[batch_idx],
                        o2o_cls_preds[batch_idx],
                        torch.zeros(
                            total_num_anchors,
                            1,
                            device=outputs.device,
                            dtype=dtype,
                        ),
                        expanded_strides_cat,
                        x_shifts_cat,
                        y_shifts_cat,
                    )
                except Exception:  # noqa: S112
                    continue

                num_fg_o2o += o2o_num_fg_img

                if o2o_num_fg_img > 0:
                    # O2O classification loss (same soft label pattern)
                    if self.use_soft_labels:
                        o2o_iou_w = o2o_pred_ious.pow(self.soft_label_gamma)
                    else:
                        o2o_iou_w = o2o_pred_ious
                    o2o_cls_target = F.one_hot(
                        o2o_gt_classes.to(torch.int64), self.num_classes
                    ) * o2o_iou_w.unsqueeze(-1)
                    o2o_cls_loss += self.bcewithlog_loss(
                        o2o_cls_preds[batch_idx][o2o_fg_mask], o2o_cls_target
                    ).sum()

                    # O2O regression loss
                    o2o_reg_target = gt_bboxes[o2o_matched_gt_inds]
                    o2o_iou_loss += self.iou_loss(
                        o2o_bbox_preds[batch_idx][o2o_fg_mask], o2o_reg_target
                    ).sum()

        # Normalize O2M
        num_fg = max(num_fg, 1)
        cls_loss = cls_loss / num_fg
        iou_loss = iou_loss / num_fg
        obj_loss = obj_loss / num_fg
        l1_loss = l1_loss / num_fg
        dfl_loss = dfl_loss / num_fg

        reg_weight = 5.0
        loss = reg_weight * iou_loss + obj_loss + cls_loss + l1_loss
        if self.use_dfl:
            loss = loss + self.dfl_loss_weight * dfl_loss

        # Add O2O loss (normalized by O2O's own num_fg)
        if self.use_dual_head and o2o_outputs is not None:
            num_fg_o2o = max(num_fg_o2o, 1)
            o2o_cls_loss = o2o_cls_loss / num_fg_o2o
            o2o_iou_loss = o2o_iou_loss / num_fg_o2o
            loss = loss + self.lambda_o2o * (reg_weight * o2o_iou_loss + o2o_cls_loss)

        return (
            loss,
            reg_weight * iou_loss,
            obj_loss,
            cls_loss,
            l1_loss,
            num_fg,
        )

    # ------------------------------------------------------------------
    # SimOTA assignment (self-contained copy from YOLOXHead)
    # ------------------------------------------------------------------

    def _get_assignments(
        self,
        batch_idx: int,
        num_gt: int,
        total_num_anchors: int,
        gt_bboxes_per_image: torch.Tensor,
        gt_classes: torch.Tensor,
        bbox_preds: torch.Tensor,
        cls_preds: torch.Tensor,
        obj_preds: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """SimOTA label assignment."""
        gt_bboxes_per_image = gt_bboxes_per_image.to(bbox_preds.device)
        gt_classes = gt_classes.to(bbox_preds.device)

        fg_mask, is_in_boxes_and_center = self._get_in_boxes_info(
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
            return (
                torch.zeros(0, device=gt_classes.device),
                torch.zeros(
                    total_num_anchors,
                    dtype=torch.bool,
                    device=gt_classes.device,
                ),
                torch.zeros(0, device=gt_classes.device),
                torch.zeros(0, device=gt_classes.device),
                0,
            )

        # Pairwise IoU
        pair_wise_ious = _bboxes_iou(gt_bboxes_per_image, bbox_preds, xyxy=False)
        # Regression cost: -log(IoU) is the existing formulation.
        # The use_log_iou_cost flag exists for ablation explicitness and
        # future GIoU cost alternative; both branches are currently identical.
        if self.use_log_iou_cost:
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        else:
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # Pairwise classification cost
        # Use expand() instead of repeat() to avoid O(num_gt * num_anchors * C)
        # memory allocation — expand() creates a view sharing the same storage.
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .expand(-1, num_in_boxes_anchor, -1)
        )

        # Pre-compute sigmoid once on the original tensors, then broadcast
        # via expand(). Avoids redundant sigmoid on num_gt copies.
        with torch.cuda.amp.autocast(enabled=False):
            cls_sigmoid = cls_preds.float().sigmoid()
            obj_sigmoid = obj_preds.float().sigmoid()

        if self.use_soft_labels:
            # RTMDet soft classification cost (SIMO-03)
            # Y_soft = IoU * one_hot_gt
            soft_label = gt_cls_per_image * pair_wise_ious.unsqueeze(-1)
            # P = cls_sigmoid * obj_sigmoid (combined prediction score)
            pred_scores = cls_sigmoid.unsqueeze(0).expand(
                num_gt, -1, -1
            ) * obj_sigmoid.unsqueeze(0).expand(num_gt, -1, -1)
            # Cost = BCE(P, Y_soft) * |Y_soft - P|^2
            scale_factor = (soft_label - pred_scores).abs().pow(2.0)
            pair_wise_cls_loss = (
                F.binary_cross_entropy(pred_scores, soft_label, reduction="none")
                * scale_factor
            ).sum(-1)
        else:
            # Original YOLOX formulation
            cls_preds_ = cls_sigmoid.unsqueeze(0).expand(
                num_gt, -1, -1
            ) * obj_sigmoid.unsqueeze(0).expand(num_gt, -1, -1)
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt(), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 1e6 * (~is_in_boxes_and_center)
        )

        (
            num_fg_result,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask,
        ) = self._dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg_result,
        )

    def _get_in_boxes_info(
        self,
        gt_bboxes_per_image: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        total_num_anchors: int,
        num_gt: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which anchors are in GT boxes or center regions."""
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        # expand() shares storage instead of allocating num_gt copies
        x_centers = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .expand(num_gt, -1)
        )
        y_centers = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .expand(num_gt, -1)
        )

        # Check 1: inside GT boxes (cxcywh)
        gt_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .expand(-1, total_num_anchors)
        )
        gt_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .expand(-1, total_num_anchors)
        )
        gt_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .expand(-1, total_num_anchors)
        )
        gt_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .expand(-1, total_num_anchors)
        )

        b_l = x_centers - gt_l
        b_r = gt_r - x_centers
        b_t = y_centers - gt_t
        b_b = gt_b - y_centers
        # Boolean & avoids allocating a stacked [N_gt, N_anchor, 4] tensor
        is_in_boxes = (b_l > 0) & (b_r > 0) & (b_t > 0) & (b_b > 0)
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # Check 2: within center radius
        center_radius = 2.5
        gt_cx = gt_bboxes_per_image[:, 0].unsqueeze(1)
        gt_cy = gt_bboxes_per_image[:, 1].unsqueeze(1)
        stride_row = expanded_strides_per_image.unsqueeze(0)
        gt_centers_l = gt_cx.expand(-1, total_num_anchors) - center_radius * stride_row
        gt_centers_r = gt_cx.expand(-1, total_num_anchors) + center_radius * stride_row
        gt_centers_t = gt_cy.expand(-1, total_num_anchors) - center_radius * stride_row
        gt_centers_b = gt_cy.expand(-1, total_num_anchors) + center_radius * stride_row

        c_l = x_centers - gt_centers_l
        c_r = gt_centers_r - x_centers
        c_t = y_centers - gt_centers_t
        c_b = gt_centers_b - y_centers
        is_in_centers = (c_l > 0) & (c_r > 0) & (c_t > 0) & (c_b > 0)
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def _dynamic_k_matching(
        self,
        cost: torch.Tensor,
        pair_wise_ious: torch.Tensor,
        gt_classes: torch.Tensor,
        num_gt: int,
        fg_mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dynamic K matching for SimOTA."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=int(dynamic_ks[gt_idx].item()), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg_result = int(fg_mask_inboxes.sum().item())

        # Clone fg_mask before modification — autograd tracks this tensor
        # through the computation graph, so in-place ops break backward().
        fg_mask = fg_mask.clone()
        fg_idxs = torch.nonzero(fg_mask, as_tuple=True)[0]
        fg_mask[fg_idxs[~fg_mask_inboxes]] = False

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return (
            num_fg_result,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask,
        )

    def _get_l1_target(
        self,
        l1_target: torch.Tensor,
        gt: torch.Tensor,
        stride: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute L1 regression targets (same as YOLOXHead)."""
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target
