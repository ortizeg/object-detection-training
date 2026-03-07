"""DINOv2 feature distillation module for DINOX.

Loads a frozen DINOv2 teacher and computes per-level MSE loss between
projected student FPN features and teacher intermediate representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationModule(nn.Module):
    """Feature distillation from a frozen DINOv2 teacher.

    Loads a pretrained DINOv2 model as teacher, freezes all its parameters,
    and computes MSE loss between projected student FPN features and teacher
    intermediate ViT block outputs.

    Args:
        student_channels: Channel dimensions of each student FPN level.
        teacher_embed_dim: Teacher embedding dimension (768 for ViT-B).
        teacher_layer_indices: 0-indexed ViT block indices to extract.
        distill_weight: Loss weight (applied externally by Lightning model).
        teacher_model: torch.hub model name for DINOv2.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        student_channels: list[int],
        teacher_embed_dim: int = 768,
        teacher_layer_indices: list[int] | None = None,
        distill_weight: float = 0.5,
        teacher_model: str = "dinov2_vitb14",
    ) -> None:
        super().__init__()

        if teacher_layer_indices is None:
            teacher_layer_indices = [3, 7, 11]

        if len(student_channels) != len(teacher_layer_indices):
            msg = (
                f"student_channels ({len(student_channels)}) must match "
                f"teacher_layer_indices ({len(teacher_layer_indices)})"
            )
            raise ValueError(msg)

        self.teacher_layer_indices = teacher_layer_indices
        self.distill_weight = distill_weight
        self.teacher_embed_dim = teacher_embed_dim

        # Load frozen teacher
        self.teacher: nn.Module = torch.hub.load(  # type: ignore[no-untyped-call]
            "facebookresearch/dinov2", teacher_model, pretrained=True
        )
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # ImageNet normalization buffers
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Projectors: one per student FPN level
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c_in, teacher_embed_dim, 1, bias=False),
                    nn.BatchNorm2d(teacher_embed_dim),
                )
                for c_in in student_channels
            ]
        )

    def train(self, mode: bool = True) -> DistillationModule:
        """Override train to keep teacher always in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self, images: torch.Tensor, student_features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature distillation loss.

        Args:
            images: BGR input images [B, C, H, W] in 0-255 range.
            student_features: List of student FPN feature maps, one per level.

        Returns:
            Sum of per-level MSE losses (NOT weighted by distill_weight).
        """
        # BGR -> RGB, normalize to ImageNet stats
        rgb = images[:, [2, 1, 0], :, :]
        images_norm = (rgb / 255.0 - self.mean) / self.std

        # Resize to nearest multiple of patch_size (DINOv2 uses 14)
        patch_size = 14
        h, w = images_norm.shape[2], images_norm.shape[3]
        new_h = (h // patch_size) * patch_size
        new_w = (w // patch_size) * patch_size
        if new_h != h or new_w != w:
            images_norm = F.interpolate(
                images_norm, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        # Teacher forward (no grad)
        # get_intermediate_layers returns tuple[Tensor, ...] but nn.Module
        # typing doesn't expose it; suppress the callable-on-Tensor false positive.
        with torch.no_grad():
            teacher_features: list[torch.Tensor] = list(
                self.teacher.get_intermediate_layers(  # type: ignore[operator]
                    images_norm, n=self.teacher_layer_indices, reshape=True
                )
            )

        # Per-level MSE loss
        total_loss = torch.tensor(0.0, device=images.device)
        for k, (proj, t_feat) in enumerate(
            zip(self.projectors, teacher_features, strict=True)
        ):
            projected: torch.Tensor = proj(student_features[k])
            projected = F.interpolate(
                projected,
                size=(t_feat.shape[2], t_feat.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            total_loss = total_loss + F.mse_loss(projected, t_feat.detach())

        return total_loss
