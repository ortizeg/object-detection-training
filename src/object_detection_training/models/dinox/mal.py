"""Matchability-Aware Loss (MAL) weighting functions.

MAL re-weights the classification loss based on a matchability score combining
IoU and classification quality. It amplifies gradients for low-quality anchor
matches while reducing to standard BCE for high-quality matches.

MAL is a loss-level modification applied on top of existing soft label
assignment targets. It does not change *what* the classification targets are,
but *how much* each positive anchor's classification loss contributes.
"""

from __future__ import annotations

import torch


def matchability_score(
    ious: torch.Tensor,
    cls_scores: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    """Compute matchability score for positive anchors (MAL-01).

    Matchability combines IoU quality and classification confidence into a
    single scalar per anchor: ``IoU^gamma * cls_score^(1-gamma)``.

    Args:
        ious: Pairwise IoU between predicted box and matched GT box for each
            positive anchor. Shape ``[num_fg]``, values in [0, 1]. These come
            from the assignment output and should be detached (no gradient).
        cls_scores: Sigmoid-activated classification prediction for the matched
            GT class at each positive anchor. Shape ``[num_fg]``, values in
            [0, 1]. These carry gradients from the current forward pass.
        gamma: Exponent controlling the IoU vs classification balance.
            Default 1.5 (IoU-dominated).

    Returns:
        Matchability scores, shape ``[num_fg]``, values in [0, 1].
    """
    return ious.pow(gamma) * cls_scores.pow(1.0 - gamma)


def mal_weight(
    matchability: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute MAL loss weight from matchability score (MAL-02).

    Uses a bounded formulation: ``(1 - matchability)^2 + 1.0``.

    This guarantees:
    - Weight = 1.0 when matchability = 1.0 (exact BCE equivalence).
    - Weight > 1.0 when matchability < 1.0 (gradient amplification).
    - Weight is bounded in [1.0, 2.0] for numerical stability.

    Args:
        matchability: Matchability scores from :func:`matchability_score`.
            Shape ``[num_fg]``, values in [0, 1].
        eps: Unused with this formulation, kept for API stability.

    Returns:
        Per-anchor loss weights, shape ``[num_fg]``, values in [1.0, 2.0].
    """
    return (1.0 - matchability).pow(2) + 1.0
