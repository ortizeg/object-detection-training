"""Distribution Focal Loss module and loss function.

Implements the DFL regression approach from "Generalized Focal Loss"
(Li et al., 2020) where bounding box distances are predicted as discrete
distributions over reg_max+1 bins, then converted to continuous values
via softmax + weighted sum (integral).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFLModule(nn.Module):
    """Convert distribution logits to continuous LTRB distances.

    Takes raw logits of shape [..., 4*(reg_max+1)] and produces
    continuous distance values of shape [..., 4] via:
    1. Reshape to [..., 4, reg_max+1]
    2. Softmax over the last dimension
    3. Weighted sum with project buffer [0, 1, ..., reg_max]

    Args:
        reg_max: Maximum bin index. The distribution has reg_max+1 bins.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project",
            torch.arange(0, reg_max + 1, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert distribution logits to point estimates.

        Args:
            x: Logits tensor of shape [..., 4*(reg_max+1)].

        Returns:
            Continuous distances of shape [..., 4].
        """
        # Save leading dims, reshape last dim to [4, reg_max+1]
        shape = x.shape[:-1]
        x = x.reshape(*shape, 4, self.reg_max + 1)
        # softmax over bins, weighted sum with project buffer
        project: torch.Tensor = self.project  # type: ignore[assignment]
        x = (F.softmax(x, dim=-1) * project).sum(dim=-1)
        return x


def distribution_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute Distribution Focal Loss.

    Targets are continuous values that fall between two adjacent bins.
    The loss is a weighted sum of cross-entropy for the floor and ceil bins,
    weighted by the fractional distance to each.

    Args:
        pred: Predicted logits of shape [N, reg_max+1].
        target: Continuous target distances of shape [N], already clamped
            to [0, reg_max - epsilon] by the caller.

    Returns:
        Loss tensor of shape [N] (unreduced).
    """
    target_left = target.long()
    target_right = target_left + 1
    weight_left = target_right.float() - target
    weight_right = target - target_left.float()

    # Compute log_softmax once (cross_entropy = log_softmax + nll_loss,
    # so two cross_entropy calls redundantly compute softmax twice).
    log_probs = F.log_softmax(pred, dim=-1)
    loss_left = F.nll_loss(log_probs, target_left, reduction="none")
    loss_right = F.nll_loss(log_probs, target_right, reduction="none")

    return weight_left * loss_left + weight_right * loss_right
