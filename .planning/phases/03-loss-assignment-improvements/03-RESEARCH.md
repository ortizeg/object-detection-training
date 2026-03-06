# Phase 3: Loss and Assignment Improvements - Research

**Researched:** 2026-03-06
**Domain:** Matchability-Aware Loss (MAL), Task-Aligned Label Assignment (TAL), gradient-weighted classification losses
**Confidence:** HIGH

## Summary

Phase 3 adds two components to the DINOXHead: (1) Matchability-Aware Loss (MAL), which re-weights the classification loss based on a matchability score combining IoU and classification quality, amplifying gradients for low-quality anchor matches while reducing to standard BCE for high-quality matches; and (2) Task-Aligned Label Assignment (TAL), an alternative assigner that replaces SimOTA's dynamic-k optimal transport with a simpler top-k selection based on an alignment metric `m = cls_score^alpha * IoU^beta`.

MAL is a loss-level modification that sits on top of the existing soft label assignment from Phase 2. It does not change *what* the classification targets are (soft IoU^gamma targets remain), but *how much* each positive anchor's classification loss contributes to the total. TAL is a separate assigner class that can completely replace `_get_assignments` in DINOXHead via a config flag (`assigner: "tal"`), enabling ablation experiments E3/E4.

**Primary recommendation:** Implement MAL as a standalone loss weighting function in a new `mal.py` module, and TAL as a separate `TaskAlignedAssigner` class in a new `tal.py` module. Both are gated by DINOXConfig flags (`use_mal`, `assigner`) and wired through DINOXHead constructor parameters following the established Phase 2 pattern.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | existing | All tensor operations, autograd for gradient verification | Already in project |
| torch.nn.functional | existing | `binary_cross_entropy_with_logits`, `one_hot` | Standard PyTorch |

### Supporting
No new libraries needed. Both MAL and TAL are pure PyTorch tensor math within existing modules.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom MAL impl | QFL from GFocal | QFL is a different concept -- joint quality/class score. MAL is specifically about gradient amplification for low-quality matches, which is the novel contribution here |
| Custom TAL impl | Ultralytics TAL | License incompatible (AGPL). Must implement from TOOD paper (ICCV 2021, MIT license) |

**Installation:**
```bash
# No new packages needed
```

## Architecture Patterns

### Recommended File Structure
```
src/object_detection_training/models/dinox/
  config.py           # Update assigner Literal, validate MAL+soft_labels dependency
  dinox_head.py        # Wire MAL weighting in _get_losses, TAL as alt assigner
  mal.py               # NEW: matchability_score(), mal_weight() functions
  tal.py               # NEW: TaskAlignedAssigner class
tests/
  test_dinox_mal.py    # NEW: MAL unit tests (TEST-03)
  test_dinox_tal.py    # NEW: TAL unit tests
```

### Pattern 1: MAL as Loss Weighting (Not a New Loss Function)

**What:** MAL computes a per-anchor weight applied to the existing BCE classification loss. It does not replace BCE -- it multiplies the per-element loss by a weighting factor derived from the matchability score.

**When to use:** When `use_mal=True` in config. Requires `use_soft_labels=True` (already validated by DINOXConfig).

**How it works:**

1. **Matchability score** (MAL-01): For each positive anchor, compute:
   ```
   matchability = IoU^gamma * cls_score^(1-gamma)
   ```
   where `IoU` is the pairwise IoU between the predicted box and matched GT box, `cls_score` is the predicted classification confidence for the matched GT class (after sigmoid), and `gamma` is `mal_gamma` (default 1.5).

2. **MAL weight** (MAL-02): The weight amplifies gradients for low-quality matches:
   ```
   weight = -log(matchability + eps)
   ```
   When matchability=1.0 (perfect match): weight approaches 0, so loss contribution is minimal (like standard BCE which already has low loss for correct predictions). When matchability is low: weight is large, amplifying the gradient signal for that anchor.

   Alternative formulation that ensures BCE equivalence at matchability=1.0:
   ```
   weight = (1 - matchability)^2 + 1.0
   ```
   This guarantees weight=1.0 at matchability=1.0 (exact BCE equivalence) and weight>1.0 for matchability<1.0 (amplification).

3. **Integration** (MAL-03): The MAL weight is applied element-wise to the classification loss:
   ```python
   cls_loss_per_anchor = bce_loss(cls_pred, soft_target)  # [num_fg, C]
   mal_w = mal_weight(matchability)                        # [num_fg, 1]
   cls_loss = (cls_loss_per_anchor * mal_w).sum()
   ```
   Soft IoU targets from Phase 2 provide the `soft_target`. MAL provides the `mal_w`. No circular dependency because:
   - `soft_target` comes from `pred_ious_this_matching` (assignment output, detached)
   - `cls_score` for matchability uses the current forward pass predictions (gradient flows)
   - `IoU` for matchability uses `pred_ious_this_matching` (assignment output, detached)

**Example:**
```python
# mal.py
def matchability_score(
    ious: torch.Tensor,       # [num_fg] - IoU from assignment (detached)
    cls_scores: torch.Tensor,  # [num_fg] - cls sigmoid for matched GT class
    gamma: float = 1.5,
) -> torch.Tensor:
    """MAL-01: Compute matchability = IoU^gamma * cls_score^(1-gamma)."""
    return ious.pow(gamma) * cls_scores.pow(1.0 - gamma)

def mal_weight(
    matchability: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MAL-02: Weight that amplifies gradient for low-quality matches.

    Returns 1.0 when matchability=1.0 (BCE equivalence), >1.0 otherwise.
    """
    # Using focal-style formulation for clean BCE equivalence at boundary
    return (1.0 - matchability).pow(2) + 1.0
```

### Pattern 2: TAL as Separate Assigner Class

**What:** TaskAlignedAssigner is a self-contained class that replaces `_get_assignments` in DINOXHead. It uses a top-k selection based on alignment metric instead of SimOTA's dynamic-k optimal transport.

**When to use:** When `assigner="tal"` in config.

**How it works (from TOOD, ICCV 2021):**

1. **Alignment metric** (TAL-01): For each (GT, anchor) pair:
   ```
   m = cls_score^alpha * IoU^beta
   ```
   Default: alpha=1.0, beta=6.0 (from Ultralytics/YOLOv8 implementation, verified).

2. **Positive selection**: Select top-k anchors per GT by alignment metric (default topk=13). Anchors must also be within GT boxes or center region (same spatial filter as SimOTA).

3. **Conflict resolution**: If an anchor is assigned to multiple GTs, keep the assignment with the highest alignment metric (same as SimOTA).

4. **Output contract**: Must return the same 5-tuple as `_get_assignments`:
   ```python
   (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)
   ```
   This ensures TAL is a drop-in replacement.

**Example:**
```python
# tal.py
class TaskAlignedAssigner:
    """TAL-01: Task-aligned label assignment from TOOD (ICCV 2021).

    Uses alignment metric m = cls_score^alpha * IoU^beta
    with top-k selection instead of SimOTA's dynamic-k.
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        num_classes: int = 80,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def assign(
        self,
        num_gt: int,
        gt_bboxes: torch.Tensor,      # [num_gt, 4] cxcywh
        gt_classes: torch.Tensor,      # [num_gt]
        bbox_preds: torch.Tensor,      # [total_anchors, 4] decoded cxcywh
        cls_preds: torch.Tensor,       # [total_anchors, C] logits
        obj_preds: torch.Tensor,       # [total_anchors, 1] logits
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        total_num_anchors: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Returns same 5-tuple as SimOTA _get_assignments."""
        ...
```

### Pattern 3: Config-Driven Assigner Dispatch

**What:** DINOXHead uses `self.assigner` string to dispatch between SimOTA and TAL.

**Example:**
```python
# In DINOXHead._get_losses, replace direct _get_assignments call:
if self.assigner_type == "tal":
    result = self._tal_assigner.assign(...)
else:
    result = self._get_assignments(...)  # existing SimOTA
```

### Anti-Patterns to Avoid

- **Modifying SimOTA code for TAL**: TAL should be a completely separate code path, not conditional branches inside `_get_assignments`. This keeps both assigners clean and independently testable.

- **Making MAL change the assignment targets**: MAL is a loss weighting mechanism only. The soft IoU targets from Phase 2 remain unchanged. MAL multiplies the per-anchor loss, it does not modify `cls_target`.

- **Circular gradient through matchability**: The IoU component of matchability must come from the assignment output (which is detached/no-gradient). Only the cls_score component should carry gradients. If both IoU and cls_score carry gradients, the matchability score creates a circular dependency between the loss and the predictions being optimized.

- **Computing MAL matchability from raw logits**: The cls_score in matchability should be the sigmoid-activated prediction for the matched GT class, not raw logits. The matchability score needs to be in [0, 1] for the weight formula to work correctly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| IoU computation | New IoU function | Existing `_bboxes_iou` in dinox_head.py | Already handles cxcywh format, well-tested |
| Spatial filtering | New in-box check | Existing `_get_in_boxes_info` from DINOXHead | TAL uses same spatial pre-filtering as SimOTA |
| BCE loss | Custom gradient loss | `nn.BCEWithLogitsLoss(reduction="none")` | MAL applies weight *after* BCE, not inside it |

**Key insight:** MAL and TAL are both thin layers on top of existing infrastructure. MAL adds ~20 lines of weighting logic. TAL reuses spatial filtering from SimOTA and only replaces the matching algorithm (top-k vs dynamic-k).

## Common Pitfalls

### Pitfall 1: Gradient Circularity in MAL
**What goes wrong:** If you compute matchability using the current forward pass's IoU (by re-computing IoU between predicted and GT boxes), then backprop through the loss creates a circular dependency: loss -> weights -> matchability -> IoU -> box predictions -> loss.
**Why it happens:** Temptation to use "fresh" IoU values for matchability.
**How to avoid:** Use `pred_ious_this_matching` from the assignment output. This is already detached (computed inside `torch.no_grad()` in `_get_assignments`). Only the `cls_score` component should carry gradients.
**Warning signs:** Training loss oscillates wildly or diverges in early epochs.

### Pitfall 2: MAL Weight Explosion at Low Matchability
**What goes wrong:** If using `-log(matchability)` as the weight, values near zero produce very large weights (e.g., matchability=0.01 gives weight=4.6), which can destabilize training.
**Why it happens:** Log-based weights have an unbounded range.
**How to avoid:** Either clamp the weight (`torch.clamp(weight, max=5.0)`) or use the bounded `(1-m)^2 + 1.0` formulation which stays in [1.0, 2.0]. The requirement says "amplified gradients" but doesn't specify the amplification must be unbounded.
**Warning signs:** NaN losses in early training when predictions are poor and matchability is near zero.

### Pitfall 3: TAL Top-K Returns Zero Positives
**What goes wrong:** With fixed top-k=13, small images with few candidate anchors might have fewer than k candidates in the spatial filter, leading to padding issues.
**Why it happens:** SimOTA's dynamic-k adapts to the number of good matches; TAL's fixed top-k doesn't.
**How to avoid:** Use `min(topk, num_candidates)` when selecting top-k. Also handle the edge case where all alignment metrics are zero (all cls_scores are near zero early in training).
**Warning signs:** `num_fg=0` consistently in early training epochs with TAL.

### Pitfall 4: Assigner Output Contract Mismatch
**What goes wrong:** TAL returns outputs in different format/semantics than SimOTA, causing downstream loss computation to silently compute wrong values.
**Why it happens:** The 5-tuple from `_get_assignments` has specific semantics (e.g., `pred_ious_this_matching` is per-positive-anchor IoU with matched GT).
**How to avoid:** Write a contract test that verifies both assigners produce outputs with identical shapes and value ranges for the same input.
**Warning signs:** Training runs but mAP is much worse with TAL despite correct implementation.

### Pitfall 5: Forgetting to Propagate New Config Flags Through Lightning
**What goes wrong:** Config flags exist in DINOXConfig but DINOXLightningModel doesn't pass them to DINOXHead constructor.
**Why it happens:** The Phase 2 flag propagation pattern requires updates at multiple levels: DINOXConfig -> DINOXLightningModel.__init__ -> DINOXHead constructor -> stored as instance attributes.
**How to avoid:** Follow the established chain: add to DINOXConfig, add constructor param to DINOXLightningModel, pass through to DINOXHead. Add a test that creates a model with `use_mal=True` and verifies it trains without error.
**Warning signs:** Config flag has no effect on training behavior.

## Code Examples

### MAL Integration in _get_losses

```python
# In DINOXHead._get_losses, inside the `if num_fg_img > 0:` block:

# Classification loss (existing soft label path from Phase 2)
if self.use_soft_labels:
    iou_weight = pred_ious_this_matching.pow(self.soft_label_gamma)
else:
    iou_weight = pred_ious_this_matching

cls_target = F.one_hot(
    gt_matched_classes.to(torch.int64), self.num_classes
) * iou_weight.unsqueeze(-1)

# Compute raw BCE loss per element
cls_loss_raw = self.bcewithlog_loss(
    cls_preds[batch_idx][fg_mask], cls_target
)  # [num_fg, C]

if self.use_mal:
    # Get cls_score for matched GT class (sigmoid of prediction)
    cls_sigmoid = cls_preds[batch_idx][fg_mask].sigmoid()
    matched_cls_idx = gt_matched_classes.to(torch.int64)
    cls_score_for_gt = cls_sigmoid[
        torch.arange(len(matched_cls_idx)), matched_cls_idx
    ]  # [num_fg]

    # Matchability score (IoU from assignment is already detached)
    m = matchability_score(
        pred_ious_this_matching, cls_score_for_gt, self.mal_gamma
    )
    # MAL weight
    w = mal_weight(m)  # [num_fg]
    cls_loss += (cls_loss_raw * w.unsqueeze(-1)).sum()
else:
    cls_loss += cls_loss_raw.sum()
```

### TAL Alignment Metric Computation

```python
# Core of TaskAlignedAssigner.assign():

# 1. Spatial filter (reuse SimOTA logic)
fg_mask, is_in_boxes_and_center = get_in_boxes_info(
    gt_bboxes, expanded_strides, x_shifts, y_shifts,
    total_num_anchors, num_gt
)

# 2. Compute alignment metric for candidates
# cls_score: sigmoid of cls_preds for each GT class
cls_sigmoid = cls_preds[fg_mask].sigmoid()  # [num_candidates, C]
gt_cls_one_hot = F.one_hot(gt_classes.long(), self.num_classes).float()

# For each GT, get the cls_score of the GT class at each candidate
# bbox_scores[g, a] = cls_sigmoid[a, gt_class[g]]
bbox_scores = cls_sigmoid[:, None, :] * gt_cls_one_hot[None, :, :]
bbox_scores = bbox_scores.sum(-1).T  # [num_gt, num_candidates]

# Pairwise IoU
pair_wise_ious = _bboxes_iou(gt_bboxes, bbox_preds[fg_mask], xyxy=False)

# Alignment metric
align_metric = bbox_scores.pow(self.alpha) * pair_wise_ious.pow(self.beta)

# 3. Select top-k per GT
topk = min(self.topk, align_metric.shape[1])
_, topk_idxs = align_metric.topk(topk, dim=1)

# Build matching matrix from topk
matching_matrix = torch.zeros_like(align_metric, dtype=torch.uint8)
for gt_idx in range(num_gt):
    matching_matrix[gt_idx, topk_idxs[gt_idx]] = 1

# Apply spatial constraint
matching_matrix *= is_in_boxes_and_center.to(matching_matrix.dtype)

# 4. Resolve conflicts (same as SimOTA)
anchor_matching_gt = matching_matrix.sum(0)
if (anchor_matching_gt > 1).sum() > 0:
    _, cost_argmin = align_metric[:, anchor_matching_gt > 1].max(dim=0)
    matching_matrix[:, anchor_matching_gt > 1] *= 0
    matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
```

### DINOXConfig Updates

```python
# In config.py - update assigner Literal to include "tal"
assigner: Literal["simota", "tal"] = "simota"

# Add TAL hyperparameters
tal_topk: int = 13
tal_alpha: float = 1.0
tal_beta: float = 6.0

# Existing MAL fields already present:
# use_mal: bool = False
# mal_gamma: float = 1.5

# Add validator: TAL + MAL combination is valid
# (MAL is a loss weighting, TAL is an assigner -- they're orthogonal)
```

### Hydra Config for E3/E4 Ablation

```yaml
# conf/models/dinox_m_e3.yaml (TAL with SimOTA baseline features)
defaults:
  - dinox_m_baseline
  - _self_

assigner: tal
tal_topk: 13
tal_alpha: 1.0
tal_beta: 6.0

# conf/models/dinox_m_e4.yaml (TAL + soft labels)
defaults:
  - dinox_m_baseline
  - _self_

assigner: tal
tal_topk: 13
tal_alpha: 1.0
tal_beta: 6.0
use_soft_labels: true
soft_label_gamma: 2.0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Binary label assignment (YOLOX) | Soft IoU^gamma targets (RTMDet) | 2022 | ~0.5-1% mAP improvement |
| SimOTA (YOLOX, 2021) | TAL (TOOD/YOLOv8, 2021-2023) | 2021-2023 | TAL simpler, comparable or better mAP |
| Fixed loss weights | Quality-aware loss weighting | 2020+ (GFL family) | Better gradient signal for hard examples |

**Key context:** TAL has largely superseded SimOTA in newer detectors (YOLOv8, YOLOv6, PP-YOLOE all use TAL). However, this project's thesis is improving YOLOX-M through training innovations, so having both assigners for ablation comparison is the point -- not choosing one over the other.

## Open Questions

1. **MAL weight function shape**
   - What we know: Requirements specify "amplified gradients when matchability is low" and "equivalent to BCE when matchability=1.0"
   - What's unclear: The exact weight function is not prescribed. The `-log(m)` formulation does NOT satisfy BCE equivalence at m=1.0 (gives weight=0, not 1.0). The `(1-m)^2 + 1.0` formulation satisfies the boundary condition.
   - Recommendation: Use `(1-m)^2 + 1.0` as default. This is bounded [1.0, 2.0], numerically stable, and satisfies both boundary conditions. The planner should treat this as a design decision for the implementer, with the boundary condition tests as acceptance criteria.

2. **E3/E4 experiment definitions**
   - What we know: TAL-02 requires E3/E4 configs using TAL. The FEATURES.md ablation matrix shows E1-E4 for DINOv2 distillation variants.
   - What's unclear: There appears to be a conflict -- the FEATURES.md ablation matrix maps E1-E4 to distillation experiments, but REQUIREMENTS.md maps E3/E4 to TAL. The requirements doc takes precedence.
   - Recommendation: Create E3/E4 Hydra configs that use TAL assigner. E3 = baseline + TAL. E4 = soft labels + TAL. Defer reconciliation with the broader ablation matrix to Phase 7.

3. **TAL topk and alpha/beta defaults**
   - What we know: Ultralytics YOLOv8 uses topk=13, alpha=1.0, beta=6.0. TOOD paper does not explicitly state these defaults (they vary by config).
   - What's unclear: Whether these defaults are optimal for YOLOX-M architecture with 3 FPN levels.
   - Recommendation: Use Ultralytics defaults (topk=13, alpha=1.0, beta=6.0) as they're well-validated. These are config parameters, so ablation can tune them later.

## Sources

### Primary (HIGH confidence)
- DINOXHead source code (`dinox_head.py`) -- complete SimOTA implementation inspected, modification points identified
- DINOXConfig source code (`config.py`) -- existing MAL fields and validator confirmed
- Phase 2 soft label implementation -- confirmed `pred_ious_this_matching` is detached from autograd
- [Ultralytics TaskAlignedAssigner](https://docs.ultralytics.com/reference/utils/tal/) -- verified alpha=1.0, beta=6.0, topk=13 defaults and `bbox_scores.pow(alpha) * overlaps.pow(beta)` formula

### Secondary (MEDIUM confidence)
- [TOOD: Task-aligned One-stage Object Detection (ICCV 2021)](https://arxiv.org/abs/2108.07755) -- original TAL paper, alignment metric formula `t = s^alpha * u^beta` verified as multiplication
- [RTMDet (arXiv 2212.07784)](https://ar5iv.labs.arxiv.org/html/2212.07784) -- soft classification cost formulation, `-log(IoU)` regression cost
- [Ultralytics tal.py source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py) -- reference implementation for positive selection, conflict resolution

### Tertiary (LOW confidence)
- MAL concept is this project's novel contribution -- no direct prior art with this exact formulation. The MADet paper (IEEE 2023) uses a related mutual-assistance idea but different mechanism. The specific matchability formula `IoU^gamma * cls_score^(1-gamma)` and weight function are defined in the requirements, not derived from a paper.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, pure PyTorch math
- Architecture (MAL): HIGH -- modification points in DINOXHead are clear, pattern follows Phase 2
- Architecture (TAL): HIGH -- well-documented algorithm with reference implementations
- Pitfalls: HIGH -- gradient circularity and weight explosion are well-understood failure modes
- MAL weight function: MEDIUM -- the exact function shape is a design choice; boundary conditions from requirements are clear but optimal amplification curve is unknown

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable domain, no fast-moving dependencies)
