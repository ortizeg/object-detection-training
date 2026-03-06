# Architecture Patterns

**Domain:** DINO-X training improvements for YOLOX within PyTorch Lightning
**Researched:** 2026-03-05

## Recommended Architecture

The DINO-X improvements layer on top of the existing YOLOX architecture through **composition, not inheritance**. The backbone (CSPDarknet) and neck (YOLOPAFPN) remain untouched. All new components attach at two integration seams:

1. **Head replacement**: A new `DINOXHead` replaces `YOLOXHead`, adding DFL regression, dual-head (O2M + O2O), and the soft SimOTA assigner.
2. **Lightning module composition**: `DINOXLightningModel` composes with (not inherits from) `YOLOXLightningModel`, adding distillation loss, MAL weighting, and feature projection.

```
                    TRAINING FLOW
                    =============

  Images ──> CSPDarknet ──> YOLOPAFPN ──> FPN Features (P3, P4, P5)
               (frozen)      (frozen)           |
                  |                             |
                  |                    +--------+--------+
                  |                    |                  |
                  |              DINOXHead           DINOXHead
                  |              (O2M branch)        (O2O branch)
                  |                    |                  |
                  |            Soft SimOTA          Top-1 Matching
                  |            Assignment           Assignment
                  |                    |                  |
                  |              MAL + DFL +         MAL + DFL +
                  |              IoU + Obj           IoU (no Obj)
                  |                    |                  |
                  |                    +--------+---------+
                  |                             |
  Images ──> DINOv2-B/14 ──> Proj ──> Distill Loss
              (frozen)                  |
                                        v
                                  Total Loss
                                  (weighted sum)


                    INFERENCE FLOW
                    ==============

  Images ──> CSPDarknet ──> YOLOPAFPN ──> FPN Features
                                              |
                                         DINOXHead
                                         (O2O branch only)
                                              |
                                        NMS-free output
                                         [boxes, scores, labels]
```

### Component Boundaries

| Component | Location | Responsibility | Communicates With | Modifies Existing? |
|-----------|----------|---------------|-------------------|--------------------|
| `DINOXConfig` | `dinox/config.py` | Pydantic config validating all improvement flags | Lightning module, Hydra | No |
| `DINOXHead` | `dinox/heads/dinox_head.py` | Decoupled head with DFL reg branch, dual O2M/O2O outputs | YOLOPAFPN output, assigners, losses | No (replaces YOLOXHead) |
| `DFLModule` | `dinox/heads/dfl.py` | Distribution Focal Loss regression layer (reg_max bins) | DINOXHead regression branch | No |
| `SoftSimOTAAssigner` | `dinox/assigners/soft_simota.py` | Soft label assignment with IoU-weighted targets | DINOXHead O2M branch | No |
| `Top1Assigner` | `dinox/assigners/top1.py` | One-to-one top-1 matching for O2O branch | DINOXHead O2O branch | No |
| `ConsistentMatcher` | `dinox/assigners/consistent.py` | Ensures O2O best sample aligns with O2M best sample | Both assigners | No |
| `MatchabilityAwareLoss` | `dinox/losses/mal.py` | MAL classification loss (IoU^gamma modulated) | DINOXHead loss computation | No |
| `DistributionFocalLoss` | `dinox/losses/dfl_loss.py` | Cross-entropy over distribution bins for box regression | DINOXHead loss computation | No |
| `DistillationModule` | `dinox/distillation/distill.py` | DINOv2 teacher, projection heads, distillation loss | Lightning module, backbone features | No |
| `FeatureProjector` | `dinox/distillation/projector.py` | Adapts CNN feature dims to DINOv2 dims (or vice versa) | DistillationModule | No |
| `DINOXLightningModel` | `models/dinox_lightning.py` | Lightning module orchestrating all components | All above, BaseDetectionModel | No |
| `DINOX` | `dinox/dinox.py` | nn.Module composing backbone + neck + DINOXHead | DINOXHead, YOLOPAFPN | No |

### Data Flow

**Training forward pass (detailed):**

1. **Input**: `images: [B, 3, H, W]`, `targets: list[DetectionTarget]`
2. **Backbone**: `CSPDarknet(images)` produces `{dark3, dark4, dark5}` feature maps
3. **Neck**: `YOLOPAFPN(features)` produces `(P3, P4, P5)` multi-scale features
   - P3: `[B, C*width, H/8, W/8]`
   - P4: `[B, 2C*width, H/16, W/16]`
   - P5: `[B, 4C*width, H/32, W/32]`
4. **DINOXHead O2M branch**: Per FPN level:
   - Stem: `1x1 conv` reduces channels to `256*width`
   - Classification branch: `2x 3x3 conv` then `1x1 conv` producing `[B, num_classes, Hk, Wk]`
   - Regression branch: `2x 3x3 conv` then `1x1 conv` producing `[B, 4*(reg_max+1), Hk, Wk]` (DFL distribution)
   - Objectness branch: `1x1 conv` producing `[B, 1, Hk, Wk]`
   - Flatten and concatenate across levels: `[B, N_anchors, 5 + num_classes + 4*reg_max]`
5. **DINOXHead O2O branch**: Identical architecture, separate weights, same input features
6. **SoftSimOTAAssigner (O2M)**: For each image in batch:
   - Filter anchors within GT boxes or center regions (same as existing SimOTA)
   - Compute cost matrix: `cls_cost + 3.0 * iou_cost + 1e6 * (~in_boxes_and_center)`
   - Dynamic-k selection: `topk_ious.sum().int().clamp(min=1)` per GT
   - **Soft targets**: classification targets are `one_hot * IoU` (not hard 0/1)
   - Returns: `fg_mask, matched_gt_inds, soft_cls_targets, pred_ious`
7. **Top1Assigner (O2O)**: For each image:
   - Same cost metric as O2M: `m = s * p^alpha * IoU^beta` (consistent matching)
   - Select top-1 anchor per GT (not Hungarian -- same performance, less compute)
   - Returns: `fg_mask, matched_gt_inds, soft_cls_targets, pred_ious`
8. **Loss computation (O2M)**:
   - **MAL (classification)**: `MAL(p, q^gamma, y)` where q=IoU, gamma=1.5
   - **DFL (regression)**: Cross-entropy over `reg_max+1` bins for each of 4 coords
   - **IoU loss**: GIoU between decoded DFL boxes and GT
   - **Objectness**: BCE loss (only in O2M branch)
9. **Loss computation (O2O)**: Same as O2M but no objectness loss, top-1 assignment
10. **Distillation loss** (if enabled):
    - `DINOv2-B/14(images)` produces patch tokens `[B, N_patches, 768]`
    - Reshape to spatial: `[B, 768, H/14, W/14]`
    - `FeatureProjector` maps CNN P4 features `[B, 2C*width, H/16, W/16]` to DINOv2 dim
    - Interpolate DINOv2 features to match P4 spatial resolution
    - L2 loss (or cosine similarity loss) between projected student and teacher features
11. **Total loss**: `w_o2m * loss_o2m + w_o2o * loss_o2o + w_distill * loss_distill`

**Inference forward pass:**

1. Images through backbone + neck (same as training)
2. O2O head only: produces decoded boxes + class scores
3. No NMS needed -- O2O produces one prediction per GT by design
4. Output format: `[B, N_anchors, 5 + num_classes]` (same YOLOX format for ONNX compat)

## Patterns to Follow

### Pattern 1: Assigner as Strategy (Pluggable Label Assignment)

**What:** Assigners implement a common protocol and are injected into the head via config.

**When:** Always. This enables the ablation configs (SimOTA vs TAL vs Soft-SimOTA).

**Why:** The existing YOLOXHead bakes `get_assignments` as a method. The new head externalizes assignment as a composable strategy, making ablation configs trivial.

```python
# dinox/assigners/base.py
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class AssignmentResult:
    """Output of a label assigner."""
    fg_mask: torch.Tensor           # [N_anchors] bool
    matched_gt_inds: torch.Tensor   # [N_fg] int
    soft_cls_targets: torch.Tensor  # [N_fg, num_classes] float (IoU-weighted)
    pred_ious: torch.Tensor         # [N_fg] float
    num_fg: int

class BaseAssigner:
    """Protocol for label assignment strategies."""
    def assign(
        self,
        pred_bboxes: torch.Tensor,    # [N_anchors, 4]
        pred_cls: torch.Tensor,       # [N_anchors, num_classes]
        pred_obj: torch.Tensor,       # [N_anchors, 1]
        gt_bboxes: torch.Tensor,      # [N_gt, 4]
        gt_classes: torch.Tensor,     # [N_gt]
        anchor_points: torch.Tensor,  # [N_anchors, 2]
        strides: torch.Tensor,        # [N_anchors]
        num_classes: int,
    ) -> AssignmentResult:
        raise NotImplementedError
```

### Pattern 2: Loss as Module (Composable Loss Functions)

**What:** Each loss function is a standalone `nn.Module` with a consistent interface.

**When:** All new losses (MAL, DFL loss, distillation loss).

**Why:** Enables independent testing, Hydra-driven weight configuration, and clean ablation (set weight=0 to disable).

```python
# dinox/losses/mal.py
class MatchabilityAwareLoss(nn.Module):
    """MAL from DEIM: classification loss modulated by IoU matchability."""

    def __init__(self, gamma: float = 1.5) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        pred_cls: torch.Tensor,   # [N_fg, num_classes] logits
        target_cls: torch.Tensor, # [N_fg, num_classes] soft targets
        ious: torch.Tensor,       # [N_fg] IoU values (matchability)
    ) -> torch.Tensor:
        """Returns scalar loss."""
        ...
```

### Pattern 3: DFL as Head Sub-Module (Distribution Focal Loss Layer)

**What:** A small module that converts `reg_max+1` bin predictions into continuous coordinates via soft-argmax (integral), and provides the DFL loss.

**When:** Used inside DINOXHead for all regression predictions.

**Why:** DFL replaces the direct 4-value regression in YOLOX. The integral (expected value of the distribution) produces continuous coordinates at inference, while the distribution shape provides the loss target at training.

```python
# dinox/heads/dfl.py
class DFLModule(nn.Module):
    """Distribution Focal Loss layer for box regression."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        # Project: register buffer for integral computation
        self.register_buffer(
            "project",
            torch.arange(0, reg_max + 1, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert distribution predictions to continuous coords via integral.

        Args:
            x: [B*N, 4*(reg_max+1)] raw predictions

        Returns:
            [B*N, 4] continuous box coordinates (ltrb format)
        """
        # Reshape to [B*N, 4, reg_max+1]
        x = x.reshape(-1, 4, self.reg_max + 1)
        # Softmax over bins, then weighted sum (soft-argmax)
        x = F.softmax(x, dim=-1)
        x = (x * self.project).sum(dim=-1)
        return x  # [B*N, 4] in ltrb (left, top, right, bottom) format
```

### Pattern 4: Frozen Teacher as Training-Only Component

**What:** DINOv2 teacher is loaded frozen, wrapped in a module that handles feature extraction and projection. It is completely excluded from ONNX export and parameter saving.

**When:** Only when distillation is enabled (`config.enable_distillation = True`).

**Why:** The teacher adds significant GPU memory (ViT-B/14 is ~86M params) but zero inference cost. Must be architecturally separated so it can be cleanly removed.

```python
# dinox/distillation/distill.py
class DistillationModule(nn.Module):
    """DINOv2 feature distillation (training-only)."""

    def __init__(
        self,
        teacher_name: str = "dinov2_vitb14",
        student_channels: int = 384,  # P4 channels for YOLOX-M
        teacher_dim: int = 768,
        loss_type: str = "l2",
    ) -> None:
        super().__init__()
        # Load frozen teacher
        self.teacher = torch.hub.load("facebookresearch/dinov2", teacher_name)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Projection: student features -> teacher dim
        self.projector = FeatureProjector(student_channels, teacher_dim)

    @torch.no_grad()
    def get_teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 patch features, reshaped to spatial map."""
        ...

    def forward(
        self,
        student_features: torch.Tensor,  # P4 from YOLOPAFPN
        images: torch.Tensor,            # original images for teacher
    ) -> torch.Tensor:
        """Compute distillation loss."""
        ...
```

### Pattern 5: Config-Driven Ablation via Pydantic + Hydra

**What:** A single `DINOXConfig` Pydantic model validates all flags. Each ablation config is a Hydra YAML that toggles specific flags.

**When:** Always. This is the primary mechanism for running ablation experiments.

```python
# dinox/config.py
from pydantic import BaseModel, field_validator

class DINOXConfig(BaseModel):
    """Configuration for all DINO-X training improvements."""

    # Assignment
    assigner: str = "soft_simota"  # "soft_simota" | "simota" | "tal"
    soft_targets: bool = True

    # Regression
    use_dfl: bool = True
    reg_max: int = 16

    # Loss
    use_mal: bool = True
    mal_gamma: float = 1.5
    iou_loss_type: str = "giou"

    # Dual head
    use_dual_head: bool = True
    o2o_loss_weight: float = 1.0
    o2m_loss_weight: float = 1.0

    # Distillation
    enable_distillation: bool = False
    teacher_model: str = "dinov2_vitb14"
    distill_loss_weight: float = 1.0
    distill_loss_type: str = "l2"
    distill_feature_level: str = "p4"

    # Scheduler-free
    use_scheduler_free: bool = False

    @field_validator("assigner")
    @classmethod
    def validate_assigner(cls, v: str) -> str:
        valid = {"soft_simota", "simota", "tal"}
        if v not in valid:
            raise ValueError(f"assigner must be one of {valid}")
        return v
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Inheriting from YOLOXLightningModel

**What:** Making `DINOXLightningModel` a subclass of `YOLOXLightningModel`.

**Why bad:** `YOLOXLightningModel.__init__` constructs a `YOLOXHead` and attaches it to `self.model`. The DINO-X head is fundamentally different (DFL output shape, dual branches). Inheritance would require overriding most methods and fighting the parent constructor. The backbone/neck construction logic would need to be duplicated anyway since DINOXHead has different `in_channels` expectations.

**Instead:** Compose. `DINOXLightningModel` extends `BaseDetectionModel` directly. It constructs the same `CSPDarknet` + `YOLOPAFPN` but uses `DINOXHead` instead of `YOLOXHead`. Reuse the weight-loading logic via a shared utility function extracted from `YOLOXLightningModel._load_weights`.

### Anti-Pattern 2: Modifying YOLOXHead In-Place

**What:** Adding DFL/dual-head support by editing `yolox/yolo_head.py`.

**Why bad:** The existing YOLOX code is a third-party copy (Megvii). Modifying it creates merge conflicts with upstream, breaks the existing YOLOX training pipeline, and violates the project constraint ("must not modify existing backbone or neck"). The head is not the backbone/neck, but modifying it still breaks existing YOLOX configs.

**Instead:** Create a new `DINOXHead` that shares the same building blocks (`BaseConv`, `DWConv`) but has its own regression output structure (DFL bins) and dual-branch logic.

### Anti-Pattern 3: Baking DINOv2 Teacher into the Model Graph

**What:** Including the DINOv2 teacher as a sub-module of `DINOX.model` that gets saved to checkpoints and exported to ONNX.

**Why bad:** The teacher is 86M extra params in every checkpoint. It cannot be traced for ONNX export (ViT architecture with dynamic operations). It wastes disk space and confuses parameter counting.

**Instead:** The `DistillationModule` lives on the Lightning module (`self.distillation`), not on `self.model`. It is explicitly excluded from `state_dict` saving via `self.distillation.teacher.requires_grad_(False)` and is skipped during ONNX export since only `self.model` is exported.

### Anti-Pattern 4: Per-Image Loop in Loss Computation

**What:** The existing `YOLOXHead.get_losses` loops over batch items (`for batch_idx in range(outputs.shape[0])`). Replicating this pattern.

**Why bad:** Per-image loops are slow on GPU. They prevent efficient batched operations and limit throughput.

**Instead:** Where possible, batch the operations. The assigner must remain per-image (variable number of GTs per image), but cost matrix computation and loss aggregation can be vectorized. Use `torch.nested` tensors or padding strategies for variable-length GT handling if batch sizes are large enough to justify the engineering effort. For the initial implementation, the per-image loop is acceptable since YOLOX uses it and the project already tolerates it.

## Detailed Component Architecture

### DINOXHead Internal Structure

```
FPN Level k (one of P3, P4, P5):
    |
    stem[k]: BaseConv 1x1 (in_channels[k]*width -> 256*width)
    |
    +-- cls_convs[k]: 2x BaseConv 3x3
    |       |
    |       cls_preds[k]: Conv2d -> [num_classes]  (classification logits)
    |
    +-- reg_convs[k]: 2x BaseConv 3x3
            |
            +-- reg_preds[k]: Conv2d -> [4 * (reg_max+1)]  (DFL distribution)
            |
            +-- obj_preds[k]: Conv2d -> [1]  (objectness, O2M only)

Dual-head structure:
    - O2M branch: cls_convs_o2m, reg_convs_o2m, cls_preds_o2m, reg_preds_o2m, obj_preds_o2m
    - O2O branch: cls_convs_o2o, reg_convs_o2o, cls_preds_o2o, reg_preds_o2o
    - Shared: stems (reduce channel dims from FPN)
    - Note: Stems are shared because the channel reduction is architecture-agnostic.
      The branch-specific convolutions diverge after the stem.
```

### DFL Box Decoding

The existing YOLOX decodes boxes as `(cx + grid) * stride` and `exp(wh) * stride`. With DFL:

1. Head outputs `[B, N, 4*(reg_max+1)]` raw distribution logits
2. `DFLModule.forward` applies softmax per 4 coords, integral to get `[B, N, 4]` in LTRB format (left, top, right, bottom distances from anchor)
3. Convert LTRB to CXCYWH or XYXY: `cx = (l + r) / 2 + anchor_x * stride`, etc.

This is a fundamental output format change from the original YOLOX. The ONNX export must bake in the DFL integral so the output remains `[B, N, 5 + num_classes]`.

### Distillation Feature Alignment

```
Student (YOLOX-M, P4):     [B, 384, H/16, W/16]
                               |
                         FeatureProjector
                         (Conv1x1 + BN + SiLU + Conv1x1)
                               |
                           [B, 768, H/16, W/16]
                               |
                          L2 / Cosine Loss
                               |
Teacher (DINOv2-B/14):    [B, 768, H/14, W/14]
                               |
                      F.interpolate to [B, 768, H/16, W/16]
```

**Why P4 for distillation:** P4 (stride 16) is closest to DINOv2's patch stride (14). Aligning at P3 (stride 8) would require 4x upsampling of teacher features, losing information. P5 (stride 32) is too coarse for feature-level distillation.

**Why projector on student side:** The teacher is frozen, so we project the student to the teacher's space. This is simpler and avoids introducing learnable params on the teacher side.

### Consistent Dual Assignment

The YOLOv10 consistent matching metric ensures the O2O top-1 selection aligns with O2M assignments:

```
m(alpha, beta) = s * p^alpha * IoU(pred, gt)^beta
```

Where `alpha` and `beta` are the same for both branches. The implementation:

1. Compute matching metric for all anchors
2. O2M branch: dynamic-k selection from cost matrix (same as SimOTA)
3. O2O branch: top-1 selection per GT from same cost matrix
4. Consistency guarantee: with identical `alpha, beta`, the top-1 for O2O is always within the top-k for O2M

This is implemented in `ConsistentMatcher` which produces both assignment results in a single pass to avoid redundant computation.

## File Structure

```
src/object_detection_training/
  models/
    dinox/
      __init__.py
      config.py              # DINOXConfig (Pydantic)
      dinox.py                # DINOX nn.Module (backbone + neck + head)
      heads/
        __init__.py
        dinox_head.py         # DINOXHead (dual O2M/O2O with DFL)
        dfl.py                # DFLModule (distribution -> coords)
      assigners/
        __init__.py
        base.py               # BaseAssigner protocol + AssignmentResult
        soft_simota.py         # SoftSimOTAAssigner
        top1.py                # Top1Assigner (O2O)
        consistent.py          # ConsistentMatcher (joint O2M + O2O)
        tal.py                 # TaskAlignedAssigner (ablation alternative)
      losses/
        __init__.py
        mal.py                # MatchabilityAwareLoss
        dfl_loss.py           # DistributionFocalLoss
        iou_loss.py           # GIoU/CIoU loss (reuse or extend existing)
      distillation/
        __init__.py
        distill.py            # DistillationModule (teacher + loss)
        projector.py          # FeatureProjector (student -> teacher dim)
    dinox_lightning.py        # DINOXLightningModel
  conf/
    models/
      dinox_base.yaml         # Base DINO-X config
      dinox_m.yaml            # DINO-X with YOLOX-M backbone
      # Ablation configs:
      dinox_m_a.yaml          # A: Baseline YOLOX-M (control)
      dinox_m_b.yaml          # B: + Soft SimOTA
      dinox_m_c.yaml          # C: + DFL
      dinox_m_d.yaml          # D: + MAL
      dinox_m_e.yaml          # E: + Dual Head
      dinox_m_f.yaml          # F: + DINOv2 Distillation
      dinox_m_g.yaml          # G: + Scheduler-Free
      dinox_m_h.yaml          # H: All combined
```

## Suggested Build Order (Dependencies)

The components have the following dependency graph:

```
Phase 1: Foundation (no dependencies)
  ├── DINOXConfig (Pydantic)
  ├── AssignmentResult dataclass
  ├── BaseAssigner protocol
  └── DFLModule

Phase 2: Core Head (depends on Phase 1)
  ├── DINOXHead (single branch, DFL regression)
  ├── DINOX nn.Module (backbone + neck + new head)
  └── DINOXLightningModel (basic, single O2M branch)
  NOTE: At this point, training should work end-to-end with standard SimOTA

Phase 3: Loss Improvements (depends on Phase 2)
  ├── SoftSimOTAAssigner (soft targets)
  ├── MatchabilityAwareLoss
  ├── DistributionFocalLoss
  └── Wire losses into DINOXHead.get_losses()
  NOTE: Each loss is independently testable and toggleable

Phase 4: Dual Head (depends on Phase 3)
  ├── Top1Assigner
  ├── ConsistentMatcher
  ├── O2O branch in DINOXHead
  └── ONNX export (O2O branch only)
  NOTE: Dual head requires assignment to be externalized (Phase 3)

Phase 5: Distillation (depends on Phase 2, independent of 3-4)
  ├── FeatureProjector
  ├── DistillationModule
  └── Distillation loss integration in Lightning module
  NOTE: Can be developed in parallel with Phases 3-4

Phase 6: Ablation Configs + Integration (depends on all above)
  ├── All Hydra YAML configs
  ├── Integration tests
  └── GCP launcher scripts
```

**Critical path:** Phase 1 -> Phase 2 -> Phase 3 -> Phase 4. Distillation (Phase 5) is off the critical path and can be developed in parallel after Phase 2.

**Why this order:**
- Phase 1 establishes interfaces that all other components depend on
- Phase 2 must produce a training loop before any improvement can be measured
- Phase 3 before Phase 4 because dual-head requires the externalized assigner pattern
- Phase 5 is independent because distillation attaches at the Lightning module level, not the head level
- Phase 6 is integration -- needs everything else working first

## Scalability Considerations

| Concern | Training (L4 24GB) | Training (A100 40GB) | Inference |
|---------|--------------------|-----------------------|-----------|
| DFL memory | +~15% over YOLOX (reg_max=16 means 68 outputs vs 4) | Negligible | Baked into integral at export |
| Dual head | +~30% head params, +~20% training memory | Acceptable | O2O only, same as single head |
| DINOv2 teacher | Does not fit (ViT-B = ~350MB + activations) | Fits with batch_size=16 | Not present |
| Distill projection | Negligible | Negligible | Not present |
| Total training | Phases 1-4 fit on L4 | All phases fit on A100 | Same as YOLOX |

## ONNX Export Strategy

The ONNX export must produce the same output format as existing YOLOX (`[B, N, 5 + num_classes]` with cx, cy, w, h, obj, cls...). This requires:

1. **DFL baking**: The `DFLModule.forward` (softmax + integral) is included in the ONNX graph. The output is 4 continuous coords, not the full distribution.
2. **O2O only**: The O2M branch and its assigners are training-only. The ONNX graph traces only the O2O forward path.
3. **No objectness for O2O**: The O2O branch does not have an objectness head. The output format changes to `[B, N, 4 + num_classes]` OR we insert a constant-1 objectness to maintain format compatibility. Recommend: constant-1 objectness for backward compatibility with existing eval pipeline.
4. **No teacher**: `DistillationModule` is never part of `self.model`, so it is automatically excluded.

## Sources

- [DEIM: DETR with Improved Matching for Fast Convergence](https://arxiv.org/html/2412.04234v1) - MAL formula and Dense O2O
- [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/html/2405.14458v1) - Consistent dual assignments, top-1 matching
- [Generalized Focal Loss (NeurIPS 2020)](https://arxiv.org/abs/2006.04388) - DFL regression with distribution bins
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/html/2304.07193v2) - Teacher model architecture
- [Leveraging Foundation Models via Knowledge Distillation](https://arxiv.org/abs/2407.18288) - DINOv2-to-CNN distillation approach
- [YOLOX SimOTA explanation](https://gmongaras.medium.com/yolox-explanation-simota-for-dynamic-label-assignment-8fa5ae397f76) - SimOTA assignment details
- Existing codebase: `yolox/yolo_head.py` (lines 296-448 for loss computation, lines 450-680 for SimOTA assignment)
