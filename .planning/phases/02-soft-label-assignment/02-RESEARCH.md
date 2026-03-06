# Phase 2: Soft Label Assignment - Research

**Researched:** 2026-03-06
**Domain:** SimOTA label assignment, soft classification targets, RTMDet cost formulations
**Confidence:** HIGH

## Summary

Phase 2 modifies the SimOTA label assignment in DINOXHead to use IoU-weighted soft targets instead of binary 1.0 for positive samples, replacing the standard YOLOX classification cost with the RTMDet formulation, and making the regression cost formula toggleable. The existing DINOXHead code (from Phase 1) already contains a complete self-contained SimOTA implementation with well-defined modification points.

The core changes are surgical: three specific locations in `_get_assignments` and `_get_losses` need modification, plus a new config flag (`use_log_iou_cost`) must be added to `DINOXConfig`. The existing `use_soft_labels` and `soft_label_gamma` fields are already defined in the config. Each change is independently toggleable per SIMO-04.

**Primary recommendation:** Modify `_get_assignments` to optionally use RTMDet soft classification cost and `-log(IoU)` regression cost, and modify `_get_losses` to use `IoU^gamma` weighted targets instead of raw IoU. All changes gated behind existing/new config flags passed from DINOXConfig through the DINOXHead constructor.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | existing | All tensor operations, BCE, softmax | Already in project |
| torch.nn.functional | existing | `binary_cross_entropy_with_logits`, `one_hot` | Standard PyTorch |

### Supporting
No new libraries needed. All changes are pure PyTorch tensor math within existing modules.

## Architecture Patterns

### Modification Points in Existing Code

The changes touch exactly three files:

```
src/object_detection_training/models/dinox/
  config.py           # Add use_log_iou_cost flag
  dinox_head.py        # Modify _get_assignments and _get_losses
tests/
  test_dinox_soft_labels.py  # New test file
```

### Pattern 1: Flag Propagation from Config to Head

**What:** DINOXConfig flags must flow from config -> DINOXHead constructor -> stored as instance attributes -> used in _get_assignments/_get_losses.
**When to use:** Every soft label change.
**Example:**
```python
# In DINOXHead.__init__:
def __init__(
    self,
    ...
    use_soft_labels: bool = False,
    soft_label_gamma: float = 2.0,
    use_log_iou_cost: bool = False,
) -> None:
    ...
    self.use_soft_labels = use_soft_labels
    self.soft_label_gamma = soft_label_gamma
    self.use_log_iou_cost = use_log_iou_cost

# Where DINOXHead is constructed (DINOXLightningModel or DINOX),
# pass from config:
head = DINOXHead(
    ...
    use_soft_labels=config.use_soft_labels,
    soft_label_gamma=config.soft_label_gamma,
    use_log_iou_cost=config.use_log_iou_cost,
)
```

### Pattern 2: Conditional Code Paths with Feature Flags

**What:** Each soft label change is gated by its own boolean flag, enabling independent ablation.
**When to use:** All three SIMO changes.
**Example:**
```python
# SIMO-01: Soft classification targets in _get_losses (line ~748-750)
# BEFORE (current code):
cls_target = F.one_hot(
    gt_matched_classes.to(torch.int64), self.num_classes
) * pred_ious_this_matching.unsqueeze(-1)

# AFTER (with soft label flag):
if self.use_soft_labels:
    iou_weight = pred_ious_this_matching.pow(self.soft_label_gamma)
else:
    iou_weight = pred_ious_this_matching
cls_target = F.one_hot(
    gt_matched_classes.to(torch.int64), self.num_classes
) * iou_weight.unsqueeze(-1)
```

### Pattern 3: RTMDet Soft Classification Cost in Assignment

**What:** Replace YOLOX classification cost `BCE(sqrt(cls*obj), one_hot)` with RTMDet formulation `BCE(P, Y_soft) * (Y_soft - P)^2` where `Y_soft = IoU * one_hot`.
**When to use:** When `use_soft_labels=True`.
**Example:**
```python
# SIMO-03: In _get_assignments, replace lines 893-901
if self.use_soft_labels:
    # RTMDet soft classification cost
    # Y_soft = IoU * one_hot_gt, shape [num_gt, num_anchors, num_classes]
    soft_label = gt_cls_per_image * pair_wise_ious.unsqueeze(-1)

    # P = cls_sigmoid * obj_sigmoid (combined prediction score)
    pred_scores = (
        cls_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
        * obj_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
    )

    # Cost = BCE(P, Y_soft) * (Y_soft - P)^2
    scale_factor = (soft_label - pred_scores).abs().pow(2.0)
    pair_wise_cls_loss = (
        F.binary_cross_entropy(
            pred_scores, soft_label, reduction="none"
        ) * scale_factor
    ).sum(-1)
else:
    # Original YOLOX formulation (unchanged)
    with torch.cuda.amp.autocast(enabled=False):
        cls_preds_ = (
            cls_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            * obj_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        )
        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)
    del cls_preds_
```

### Pattern 4: Toggleable Regression Cost

**What:** The `-log(IoU)` regression cost is already used in both YOLOX and DINOXHead. The `use_log_iou_cost` flag makes this explicit and could enable switching to GIoU cost.
**When to use:** SIMO-02 compliance.
**Example:**
```python
# SIMO-02: In _get_assignments
# Current code (already -log(IoU)):
pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

# Note: The requirement says "replaces GIoU cost" but the current
# YOLOX codebase already uses -log(IoU). The flag exists to make
# this toggleable if we want to compare with GIoU cost:
if self.use_log_iou_cost:
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
else:
    # GIoU cost (would need _bboxes_giou helper)
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # same for now
```

### Anti-Patterns to Avoid
- **Modifying _dynamic_k_matching:** The dynamic-k selection should remain unchanged. Soft labels affect cost computation and loss targets, not the matching strategy itself.
- **Applying gamma to assignment IoUs:** The `soft_label_gamma` should only affect classification TARGET weighting in `_get_losses`, not the IoU values used during assignment cost computation. Assignment uses raw IoU; loss targets use IoU^gamma.
- **Mutating IoU tensor in-place:** Use `.pow()` not `.pow_()` since `pred_ious_this_matching` may be needed elsewhere unchanged.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BCE loss | Custom cross-entropy | `F.binary_cross_entropy` / `F.binary_cross_entropy_with_logits` | Numerically stable, handles edge cases |
| IoU computation | New IoU function | Existing `_bboxes_iou` helper | Already tested, handles cxcywh format |

## Common Pitfalls

### Pitfall 1: In-place Operations Breaking Autograd
**What goes wrong:** Using `.sigmoid_()` or `.pow_()` on tensors that participate in the backward pass causes autograd errors.
**Why it happens:** In-place ops modify tensors already recorded in the computation graph.
**How to avoid:** The existing YOLOX code uses `.sigmoid_()` inside `torch.cuda.amp.autocast(enabled=False)` which is intentional. For the new RTMDet path, use non-in-place `.sigmoid()` since the computation is different.
**Warning signs:** RuntimeError about in-place modification of a leaf variable.

### Pitfall 2: AMP Precision Issues in Cost Computation
**What goes wrong:** Mixed precision causes NaN or Inf in BCE with very small/large logit values.
**Why it happens:** Float16 has limited range; log(0) or BCE with 0/1 predictions is numerically unstable.
**How to avoid:** Keep the `torch.cuda.amp.autocast(enabled=False)` guard around classification cost computation. Clamp IoU values before log: `torch.log(pair_wise_ious.clamp(min=1e-8))`.
**Warning signs:** NaN loss values, especially early in training when predictions are random.

### Pitfall 3: Zero Foreground Assignments
**What goes wrong:** Soft label cost function assigns different costs than binary YOLOX, potentially reducing positive samples to zero.
**Why it happens:** RTMDet formulation weighs cost differently, especially when predictions are far from ground truth early in training.
**How to avoid:** The existing num_fg fallback (`max(num_fg, 1)`) handles this. Also monitor num_fg during training. Consider whether cost weights (currently `3.0` for regression) need adjustment with the new cost formulation.
**Warning signs:** num_fg consistently at 0 or 1 across batches.

### Pitfall 4: Gamma=0 Edge Case
**What goes wrong:** `IoU^0 = 1.0` for all positive samples, which reverts to binary labels.
**Why it happens:** gamma=0 makes all IoU values equal to 1.0.
**How to avoid:** This is actually the correct fallback behavior. Document that gamma=0 produces YOLOX-equivalent behavior. Validate gamma >= 0 in DINOXConfig.
**Warning signs:** None -- this is intentionally the bridge between soft and binary labels.

### Pitfall 5: Misaligning BCE Formulation (Logits vs Probabilities)
**What goes wrong:** Using `binary_cross_entropy_with_logits` when inputs are already sigmoid'd, or vice versa.
**Why it happens:** The RTMDet formulation in mmdetection uses `binary_cross_entropy_with_logits` with raw logits for the cost, while the existing YOLOX code applies sigmoid first then uses `binary_cross_entropy`.
**How to avoid:** Be consistent. The RTMDet mmdetection code uses logits (`binary_cross_entropy_with_logits`). But since our code already computes `cls_sigmoid * obj_sigmoid` as combined predictions, use `binary_cross_entropy` (not with_logits) for the cost computation to avoid double-sigmoid.
**Warning signs:** Loss values that are unexpectedly large or training that diverges.

## Code Examples

### Example 1: Complete Soft Classification Cost (RTMDet formulation)
```python
# Source: mmdetection DynamicSoftLabelAssigner
# Adapted for YOLOX-style combined cls*obj predictions

# pair_wise_ious: [num_gt, num_anchors]
# gt_cls_per_image: [num_gt, num_anchors, num_classes] (one-hot, repeated)
# cls_preds: [num_anchors, num_classes] (raw logits)
# obj_preds: [num_anchors, 1] (raw logits)

# Soft label: IoU * one_hot
soft_label = gt_cls_per_image * pair_wise_ious.unsqueeze(-1)

# Combined prediction: cls_sigmoid * obj_sigmoid
pred_scores = (
    cls_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
    * obj_preds.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
)

# RTMDet cost: BCE(P, Y_soft) * |Y_soft - P|^2
scale_factor = (soft_label - pred_scores).abs().pow(2.0)
pair_wise_cls_loss = (
    F.binary_cross_entropy(pred_scores, soft_label, reduction="none")
    * scale_factor
).sum(dim=-1)
```

### Example 2: IoU^gamma Weighted Classification Targets in Loss
```python
# Source: RTMDet / GFL (Generalized Focal Loss) pattern
# In _get_losses, after SimOTA assignment returns pred_ious_this_matching

if self.use_soft_labels:
    # Soft targets: IoU^gamma weighted
    iou_weight = pred_ious_this_matching.pow(self.soft_label_gamma)
else:
    # Original YOLOX: raw IoU as weight
    iou_weight = pred_ious_this_matching

cls_target = (
    F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes)
    * iou_weight.unsqueeze(-1)
)
```

### Example 3: Unit Test for IoU^gamma Targets
```python
def test_soft_label_iou_gamma_weighting() -> None:
    """Verify IoU^gamma produces correct target values."""
    gamma = 2.0
    ious = torch.tensor([0.8, 0.5, 0.3])
    expected = ious.pow(gamma)  # [0.64, 0.25, 0.09]

    # With 2 classes, class 0 positive for all:
    classes = torch.zeros(3, dtype=torch.int64)
    one_hot = F.one_hot(classes, 2).float()  # [[1,0],[1,0],[1,0]]
    targets = one_hot * expected.unsqueeze(-1)

    assert torch.allclose(targets[:, 0], expected)
    assert torch.allclose(targets[:, 1], torch.zeros(3))
```

### Example 4: Unit Test for -log(IoU) Cost
```python
def test_log_iou_cost_values() -> None:
    """Verify -log(IoU) cost is correct for known values."""
    ious = torch.tensor([[0.5, 0.8, 0.1]])
    cost = -torch.log(ious + 1e-8)
    expected = torch.tensor([[0.6931, 0.2231, 2.3026]])
    assert torch.allclose(cost, expected, atol=1e-3)
```

### Example 5: Unit Test for RTMDet Soft Classification Cost
```python
def test_rtmdet_soft_cls_cost() -> None:
    """Verify CE(P, Y_soft) * (Y_soft - P)^2 against hand-computed values."""
    # 1 GT, 1 anchor, 2 classes. GT class=0, IoU=0.7
    pred_score = torch.tensor([[[0.6, 0.1]]])  # already sigmoid'd
    soft_label = torch.tensor([[[0.7, 0.0]]])   # IoU * one_hot

    bce = F.binary_cross_entropy(pred_score, soft_label, reduction="none")
    scale = (soft_label - pred_score).abs().pow(2.0)
    cost = (bce * scale).sum(dim=-1)

    # Hand compute for class 0:
    # BCE(0.6, 0.7) = -(0.7*log(0.6) + 0.3*log(0.4)) = 0.6108
    # scale = |0.7 - 0.6|^2 = 0.01
    # class 0 contrib = 0.6108 * 0.01 = 0.006108
    # Hand compute for class 1:
    # BCE(0.1, 0.0) = -(0*log(0.1) + 1*log(0.9)) = 0.1054
    # scale = |0.0 - 0.1|^2 = 0.01
    # class 1 contrib = 0.1054 * 0.01 = 0.001054
    expected = torch.tensor([[0.006108 + 0.001054]])
    assert torch.allclose(cost, expected, atol=1e-3)
```

## Detailed Change Specification

### Change 1: DINOXConfig (config.py)

Add `use_log_iou_cost: bool = False` field. This is the only config change needed -- `use_soft_labels` and `soft_label_gamma` already exist.

```python
# Add after soft_label_gamma:
use_log_iou_cost: bool = False
```

### Change 2: DINOXHead.__init__ (dinox_head.py)

Add three new constructor parameters: `use_soft_labels`, `soft_label_gamma`, `use_log_iou_cost`. Store as instance attributes.

### Change 3: DINOXHead._get_assignments (dinox_head.py, lines ~881-908)

Modify classification cost computation to conditionally use RTMDet formulation when `self.use_soft_labels=True`. The regression cost (`-log(IoU)`) is already correct; add `use_log_iou_cost` toggle for explicitness.

### Change 4: DINOXHead._get_losses (dinox_head.py, lines ~748-750)

Modify classification target computation to use `IoU^gamma` when `self.use_soft_labels=True`.

### Change 5: DINOXHead construction sites

Update all places where DINOXHead is constructed to pass the new flags. Check:
- `DINOX.__init__` default construction
- `DINOXLightningModel` (Lightning module that creates DINOX from config)
- Test helpers (`_make_model` in test_dinox_head.py)

### Change 6: Unit tests (new file: tests/test_dinox_soft_labels.py)

- Test IoU^gamma target values against hand-computed examples
- Test -log(IoU) cost values against hand-computed examples
- Test RTMDet soft classification cost against hand-computed examples
- Test flag independence: enabling one flag doesn't affect others
- Test gamma=0 produces binary-equivalent targets
- Test gradient flow with soft labels enabled

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Binary 1.0 targets (YOLOX) | IoU-weighted soft targets (RTMDet/GFL) | 2022-2023 | +0.5% AP typical improvement |
| BCE(sqrt(cls*obj), one_hot) cost | BCE(P, Y_soft) * (Y_soft-P)^2 cost | RTMDet 2022 | Better cls-reg alignment in assignment |
| GIoU regression cost | -log(IoU) regression cost | RTMDet 2022 | More discriminative at low IoU |

**Key insight:** Both YOLOX and our DINOXHead already use `-log(IoU)` for regression cost in SimOTA (not GIoU as the requirement text suggests). The `use_log_iou_cost` flag is still worth adding for ablation completeness, but the default behavior already matches the target.

## Open Questions

1. **Cost weight rebalancing with RTMDet formulation**
   - What we know: Current cost uses `cls_cost + 3.0 * reg_cost`. RTMDet uses `1.0 * cls_cost + 3.0 * reg_cost`.
   - What's unclear: Whether the RTMDet soft cls cost produces values at similar scale to the original YOLOX cls cost. The `(Y_soft - P)^2` scaling factor could make values much smaller.
   - Recommendation: Keep the `3.0` weight for now. If training shows assignment issues, this is the first thing to tune.

2. **AMP compatibility of new cost path**
   - What we know: Existing code wraps cls cost in `torch.cuda.amp.autocast(enabled=False)`.
   - What's unclear: Whether the new RTMDet formulation needs the same guard.
   - Recommendation: Apply the same `autocast(enabled=False)` guard to the new soft label cost path for safety.

3. **Whether `use_log_iou_cost` should default to True when `use_soft_labels=True`**
   - What we know: Both already use `-log(IoU)`. The flag is for ablation.
   - What's unclear: Whether the two should be coupled or independent.
   - Recommendation: Keep independent per SIMO-04. Both default to False. When the user enables soft labels, they can independently toggle each sub-feature.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/object_detection_training/models/dinox/dinox_head.py` -- current SimOTA implementation, modification points identified
- Existing codebase: `src/object_detection_training/models/dinox/config.py` -- existing DINOXConfig with use_soft_labels, soft_label_gamma
- [mmdetection DynamicSoftLabelAssigner](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py) -- reference implementation of RTMDet soft label assignment

### Secondary (MEDIUM confidence)
- [RTMDet paper (arXiv:2212.07784)](https://arxiv.org/pdf/2212.07784) -- Section 3.3, soft label assignment formulation
- [RTMDet HTML version](https://ar5iv.labs.arxiv.org/html/2212.07784) -- Equation 1, cost matrix formulation

### Tertiary (LOW confidence)
- None -- all findings verified against source code or paper

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, pure PyTorch tensor math
- Architecture: HIGH -- modification points precisely identified in existing code with line numbers
- Pitfalls: HIGH -- known issues from YOLOX/RTMDet community, verified against existing code patterns
- Code examples: HIGH -- verified against mmdetection reference implementation

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable domain, no rapidly changing dependencies)
