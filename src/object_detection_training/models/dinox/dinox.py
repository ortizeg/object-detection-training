"""DINOX model: YOLOPAFPN backbone + DINOXHead.

Matches the structure of YOLOX (backbone + head composition) with
training/inference branching in forward().
"""

from __future__ import annotations

import torch
import torch.nn as nn

from object_detection_training.models.yolox import YOLOPAFPN

from .dinox_head import DINOXHead


class DINOX(nn.Module):
    """DINOX detection model.

    Composes a YOLOPAFPN backbone with a DINOXHead. Follows the same
    forward contract as YOLOX: returns loss tuple during training,
    decoded predictions tensor during inference.

    Args:
        backbone: YOLOPAFPN feature extractor.
        head: DINOXHead detection head.
    """

    def __init__(
        self,
        backbone: YOLOPAFPN | None = None,
        head: DINOXHead | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()  # type: ignore[no-untyped-call]
        if head is None:
            head = DINOXHead(num_classes=80)
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor | float] | torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].
            targets: Optional training targets (list of dicts with
                "boxes" in cxcywh and "labels").

        Returns:
            Training (targets provided): dict with loss components
                matching the YOLOX output contract.
            Inference (no targets): [B, N, 5+C] decoded predictions.
        """
        fpn_outs = self.backbone(x)

        if targets is not None:
            was_training = self.head.training
            self.head.train()
            try:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets=targets
                )
            finally:
                if not was_training:
                    self.head.eval()

            outputs: dict[str, torch.Tensor | float] = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "fpn_features": fpn_outs,
            }

            if not self.training:
                outputs["predictions"] = self.head(fpn_outs)

            return outputs
        else:
            result: torch.Tensor = self.head(fpn_outs)
            return result
