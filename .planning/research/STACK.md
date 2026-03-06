# Technology Stack

**Project:** YOLOX Training Improvements (DINO-X-inspired)
**Researched:** 2026-03-05
**Overall Confidence:** MEDIUM-HIGH

## Context

Adding modern detection training improvements to an existing YOLOX-M codebase:
- Soft SimOTA label assignment
- Distribution Focal Loss (DFL) box regression
- Matchability-Aware Loss (MAL)
- DINOv2-B/14 frozen teacher distillation
- NMS-free dual-head (one-to-one + one-to-many)
- Scheduler-free AdamW optimizer

Constraint: Apache 2.0 only. No ultralytics/AGPL code.

---

## Recommended Stack

### Core Framework (Already in Place)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch | >=2.2 (current pixi) | Training framework | Already used. torch.compile available for 20-50% training speedup on A100 | HIGH |
| PyTorch Lightning | >=2.5 (current pixi) | Training loop abstraction | Already used. Handles DDP, checkpointing, logging. Latest is 2.6.1 | HIGH |
| Hydra | >=1.3 (current pixi) | Configuration management | Already used. New components (DFL, dual-head, distillation) each get their own config group | HIGH |
| Pydantic | >=2.0 (current pixi) | Config validation | Already used. Validate new hyperparameters (DFL bins, distillation weight, etc.) | HIGH |
| torchvision | >=0.17 (current pixi) | CIoU/GIoU loss, NMS ops | `torchvision.ops.complete_box_iou_loss` is built-in. Use instead of custom IoU loss | HIGH |

### New Dependencies

| Technology | Version | Purpose | Why | License | Confidence |
|------------|---------|---------|-----|---------|------------|
| schedulefree | 1.4.1 | Scheduler-free AdamW optimizer | Replaces SGD + cosine annealing. No LR schedule tuning needed. From Facebook Research. Requires train/eval mode switching (see Pitfalls) | Apache 2.0 | HIGH |
| transformers | >=4.49.0,<4.52.0 (already pinned) | Load DINOv2-B/14 as frozen teacher | Already a dependency. `AutoModel.from_pretrained("facebook/dinov2-base")` loads ViT-B/14 | Apache 2.0 | HIGH |
| scipy | * (already present) | Hungarian matching for one-to-one head | `scipy.optimize.linear_sum_assignment` for O2O label assignment | BSD | HIGH |

### No New Dependencies Needed

These components are implemented as custom PyTorch modules (pure `torch.nn` / `torch.nn.functional`):

| Component | Implementation Approach | Why No Library |
|-----------|------------------------|----------------|
| Distribution Focal Loss (DFL) | ~20 lines of PyTorch. `F.cross_entropy` on discretized box distributions | The loss is trivial: softmax over bins, cross-entropy weighted by distance to target. Reference: GFocal paper (NeurIPS 2020) |
| Soft SimOTA | Modify existing `YOLOXHead.get_assignments()` / `dynamic_k_matching()` | Already have SimOTA in `yolo_head.py`. Soft version replaces hard 0/1 targets with IoU-weighted soft targets. ~30 line diff |
| Matchability-Aware Loss (MAL) | Custom loss weighting module | Weights loss per-anchor by "matchability" score (how well-matched an anchor is). Pure tensor ops |
| Quality Focal Loss (QFL) | Replace `BCEWithLogitsLoss` for classification | `F.binary_cross_entropy_with_logits` with IoU-weighted soft targets. ~15 lines |
| One-to-one head | Duplicate `YOLOXHead` structure + Hungarian matching | Second head with identical architecture, different assignment. Stop-gradient from O2O branch. ~100 lines |
| Feature alignment projector | `nn.Conv2d` + `nn.BatchNorm2d` layers | Simple 1x1 conv to project YOLOX features to DINOv2 feature space |

---

## Detailed Technology Decisions

### 1. Optimizer: schedulefree AdamWScheduleFree

**Use** `schedulefree==1.4.1` (Apache 2.0, Facebook Research)

**Why:**
- Eliminates LR scheduler tuning entirely -- no cosine annealing, no warmup epochs config
- Learning rates 1-10x larger than scheduled AdamW work well (paper result)
- v1.4.1 includes RAdam variant as fallback
- v1.3+ fixed weight decay behavior during warmup for consistency with standard AdamW

**Integration with Lightning:**
- Override `model.train()` and `model.eval()` to also call `optimizer.train()` / `optimizer.eval()`
- Override checkpoint save callback to put optimizer in eval mode before saving
- Reference: [Lightning discussion #19759](https://github.com/Lightning-AI/pytorch-lightning/discussions/19759)

**Why not SGD + cosine:** Current codebase uses SGD with 3 parameter groups + LinearLR warmup + CosineAnnealingLR. This is 40+ lines of scheduler config. schedulefree replaces all of it with a single optimizer instantiation.

**Why not standard AdamW:** AdamW still needs a schedule. schedulefree matches or beats scheduled AdamW without the schedule.

**Installation:**
```bash
# Via pixi (pypi dependency)
# Add to pixi.toml [pypi-dependencies]:
schedulefree = ">=1.4.0, <2"
```

### 2. DINOv2 Teacher: transformers (already installed)

**Use** `transformers>=4.49.0,<4.52.0` (already pinned in pixi.toml)

**Why:**
- DINOv2-B/14 is available via `transformers.AutoModel`
- No additional dependency needed
- Model is frozen (no gradient computation), so memory overhead is just forward pass
- DINOv2 is Apache 2.0 licensed (both code and weights)

**Loading:**
```python
from transformers import AutoModel
teacher = AutoModel.from_pretrained("facebook/dinov2-base")
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
```

**Alternative considered:** `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` -- works but transformers is already a dependency and provides better caching/versioning.

**Feature extraction:** DINOv2-B/14 outputs patch tokens of dim 768 from 16x16 patches at 224px or 14x14 patches at 196px. For 640px input, resize to 518px (37x37 patches) or use the model's native interpolation. The student (YOLOX) feature maps at stride 8/16/32 need 1x1 conv projection to match 768-dim teacher features.

### 3. Distribution Focal Loss: Custom Implementation

**Why custom (not a library):**
- DFL is ~20 lines of code. Adding a dependency for this would be over-engineering
- The GFocal reference implementation (implus/GFocal) is built on MMDetection -- pulling that in is absurd for one loss function
- YOLOv8/ultralytics has DFL but is AGPL -- cannot use

**Implementation reference (from GFocal paper, NeurIPS 2020):**
```python
class DFLoss(nn.Module):
    """Distribution Focal Loss for box regression."""
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred_dist: (N, reg_max+1) -- raw logits over bins
        # target: (N,) -- continuous regression target in [0, reg_max]
        target_left = target.long().clamp(0, self.reg_max - 1)
        target_right = (target_left + 1).clamp(max=self.reg_max)
        weight_right = target - target_left.float()
        weight_left = 1.0 - weight_right
        loss = (
            F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
            + F.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        )
        return loss.mean(-1)
```

**Head changes:** Replace the 4-output `reg_preds` conv with `4 * (reg_max + 1)` outputs. Decode via `softmax -> weighted sum` to get continuous box values.

### 4. IoU Loss: torchvision.ops.complete_box_iou_loss

**Use** `torchvision.ops.complete_box_iou_loss` (CIoU)

**Why:**
- Already a dependency
- CIoU > GIoU > IoU for convergence speed and accuracy (well-established)
- Current codebase has custom `IOUloss` class that only supports IoU and GIoU
- torchvision's implementation is tested, CUDA-optimized, and handles edge cases

**Migration:** Replace `self.iou_loss = IOUloss(...)` with `torchvision.ops.complete_box_iou_loss(pred, target, reduction="none")`. Need to convert from cxcywh to xyxy format before calling.

### 5. One-to-One Head: Custom with scipy Hungarian

**Use** `scipy.optimize.linear_sum_assignment` (already a dependency)

**Why:**
- Hungarian algorithm is the standard for one-to-one matching (used by DETR, RT-DETR, YOLOv10)
- scipy's implementation is C-optimized and handles the assignment in <1ms for typical batch sizes
- The O2O head is structurally identical to the existing YOLOXHead -- just different label assignment

**Architecture (from YOLOv10 paper, NeurIPS 2024):**
- During training: both O2M (SimOTA) and O2O (Hungarian) heads compute loss
- O2O branch uses stop-gradient to prevent it from degrading O2M supervision
- During inference: only O2O head is used (NMS-free)
- Consistent matching metric between both heads for harmonious optimization

**Why not copy YOLOv10 code:** YOLOv10 (THU-MIG/yolov10) is AGPL-3.0 licensed. Cannot use. The dual-head concept is described in the paper and is straightforward to implement from the paper description.

### 6. torch.compile for Training Speedup

**Use** `torch.compile(model, mode="reduce-overhead")` on A100 phases

**Why:**
- 20-50% training speedup typical on A100 with no code changes
- Works with Lightning's `Trainer(compile=True)` or manual wrapping
- Compatible with DDP and mixed precision
- August 2025 status report shows 93% model compatibility

**When:** Phase 4+ (A100 phases). Skip on L4 phases (compile overhead may not pay off for shorter runs).

**Caveat:** May need `torch.compiler.disable` around dynamic-shape SimOTA assignment code if it causes graph breaks. LOW confidence on seamless compilation of the full training loop with all improvements.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Optimizer | schedulefree AdamWScheduleFree | SGD + cosine (current) | Too many hyperparameters; schedulefree matches performance without schedule tuning |
| Optimizer | schedulefree AdamWScheduleFree | Prodigy / D-Adaptation | Less mature, fewer users, schedulefree has stronger theoretical backing |
| DINOv2 loading | transformers AutoModel | torch.hub.load | transformers already a dependency, better caching |
| IoU loss | torchvision CIoU | Custom IOUloss (current) | torchvision is tested, maintained, CUDA-optimized |
| DFL | Custom 20-line module | MMDetection DistributionFocalLoss | MMDetection is a massive dependency for one loss function |
| O2O matching | scipy Hungarian | torch-based Hungarian | scipy's is C-optimized; torch alternatives add complexity |
| Label assignment | Modify existing SimOTA | TAL (Task-Aligned Assignment) | SimOTA already works well for YOLOX; soft modification is simpler than replacing entirely |
| Dual head reference | Paper-based implementation | YOLOv10 code | AGPL license -- cannot use |
| Training framework | PyTorch Lightning (keep) | Raw PyTorch | Already invested; Lightning handles DDP, logging, checkpointing |

---

## Stack Summary by Phase

| Phase | New Deps | Key Stack Elements |
|-------|----------|--------------------|
| 1: Soft SimOTA + QFL | None | Modify existing `yolo_head.py` |
| 2: DFL + CIoU | None | Custom DFLoss module + `torchvision.ops.complete_box_iou_loss` |
| 3: MAL | None | Custom loss weighting module |
| 4: DINOv2 Distillation | None (transformers already installed) | `transformers.AutoModel` + custom projector |
| 5: Dual Head | None (scipy already installed) | Custom O2O head + `scipy.optimize.linear_sum_assignment` |
| 6: Scheduler-free | `schedulefree>=1.4.0,<2` | `AdamWScheduleFree` + Lightning integration callback |

**Only one new dependency (schedulefree) is needed.** Everything else is either already installed or implemented as custom PyTorch modules.

---

## Installation

```bash
# Only new dependency needed:
# Add to pixi.toml [pypi-dependencies]:
# schedulefree = ">=1.4.0, <2"

# Then:
/Users/ortizeg/.pixi/bin/pixi install
```

---

## Sources

### HIGH Confidence
- [YOLOX (Apache 2.0)](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) -- existing codebase license
- [DINOv2 (Apache 2.0)](https://github.com/facebookresearch/dinov2) -- teacher model, code and weights
- [schedulefree (Apache 2.0)](https://github.com/facebookresearch/schedule_free) -- v1.4.1, Facebook Research
- [torchvision CIoU](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.complete_box_iou_loss.html) -- built-in loss function
- [Lightning schedulefree discussion](https://github.com/Lightning-AI/pytorch-lightning/discussions/19759) -- integration pattern
- [PyTorch Lightning 2.6.1](https://pypi.org/project/lightning/) -- latest stable

### MEDIUM Confidence
- [GFocal / DFL paper](https://arxiv.org/abs/2006.04388) -- NeurIPS 2020, reference for DFL implementation
- [YOLOv10 paper](https://arxiv.org/html/2405.14458v2) -- NeurIPS 2024, reference for dual-head architecture
- [torch.compile training status](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/) -- August 2025 report

### LOW Confidence (Needs Validation)
- torch.compile compatibility with dynamic SimOTA assignment -- may cause graph breaks
- Exact DINOv2 feature resolution at 640px input -- need to verify patch interpolation behavior
- schedulefree hyperparameter transfer -- learning rate range may need tuning vs. paper claims

### License Verification
- YOLOX: Apache 2.0 -- VERIFIED
- DINOv2: Apache 2.0 -- VERIFIED
- schedulefree: Apache 2.0 -- VERIFIED
- YOLOv10 (THU-MIG): AGPL-3.0 -- CANNOT USE code, paper concepts only
- ultralytics: AGPL-3.0 -- CANNOT USE
- GFocal (implus): Built on MMDetection (Apache 2.0) -- loss function math is public domain (paper), safe to reimplement
