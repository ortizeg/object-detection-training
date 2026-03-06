# Phase 4: NMS-Free Dual Head - Research

**Researched:** 2026-03-06
**Domain:** Dual-head object detection with O2O/O2M matching, NMS-free inference
**Confidence:** HIGH

## Summary

Phase 4 adds a second detection head (O2O) alongside the existing O2M head in DINOXHead, enabling NMS-free inference. The architecture follows the YOLOv10 dual label assignment pattern: during training, both heads share the same backbone/FPN features and conv stacks but have separate prediction parameters, with a consistent alignment metric ensuring the O2O head's single best prediction per GT aligns with the O2M head's top match. The O2O head uses Hungarian matching (scipy `linear_sum_assignment`) for strict 1:1 assignment, while the O2M head continues using SimOTA or TAL. At ONNX export time, only the O2O branch runs, producing NMS-free output.

The existing codebase is well-structured for this. DINOXHead already has per-FPN-level ModuleLists for stems, cls_convs, reg_convs, cls_preds, reg_preds, obj_preds. The O2O head needs a parallel set of prediction layers (cls_preds_o2o, reg_preds_o2o, obj_preds_o2o) that share the conv stacks but have separate final 1x1 conv weights. The assigner dispatch pattern (`assigner_type` config) and the 5-tuple return contract from `_get_assignments` / `TaskAlignedAssigner.assign` provide a clean interface for the Hungarian assigner.

**Primary recommendation:** Implement the O2O head as additional ModuleLists on DINOXHead (not a separate nn.Module), add a HungarianAssigner class matching the existing 5-tuple return contract, compute combined loss in `_get_losses`, and modify `_forward_inference` to use O2O prediction layers when `use_dual_head=True`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | (already in deps) | `linear_sum_assignment` for Hungarian matching | Standard LSAP solver, used by DETR family (rfdetr/matcher.py already uses it) |
| torch | (existing) | O2O head layers, loss computation | Core framework |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| onnx | (existing) | Validate O2O-only export | ONNX export tests |
| onnxruntime | (existing) | Runtime shape verification | ONNX round-trip tests |

**No new dependencies required.** scipy is already a project dependency and is already used in `rfdetr/matcher.py`.

## Architecture Patterns

### O2O Head Structure Within DINOXHead

The O2O head is NOT a separate class. It adds parallel prediction layers to the existing DINOXHead:

```
DINOXHead (modified)
  stems[]          # shared (existing)
  cls_convs[]      # shared (existing)
  reg_convs[]      # shared (existing)
  cls_preds[]      # O2M prediction layers (existing)
  reg_preds[]      # O2M prediction layers (existing)
  obj_preds[]      # O2M prediction layers (existing)
  cls_preds_o2o[]  # NEW: O2O prediction layers (separate params)
  reg_preds_o2o[]  # NEW: O2O prediction layers (separate params)
  obj_preds_o2o[]  # NEW: O2O prediction layers (separate params)
```

This follows DUAL-01: "O2O head has identical structure to O2M head but separate parameters." The conv stacks (feature extraction) are shared; only the final 1x1 prediction convolutions are duplicated.

### Pattern 1: Hungarian Assigner (DUAL-02)

**What:** A `HungarianAssigner` class that uses `scipy.optimize.linear_sum_assignment` to produce strict 1:1 GT-to-prediction matching.

**When to use:** O2O head label assignment during training.

**Key design:** Must return the same 5-tuple as `_get_assignments` and `TaskAlignedAssigner.assign`:
```python
(gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)
```

**Cost matrix construction:** Uses the same alignment metric as the O2M head for consistency (DUAL-03):
```python
# alignment_metric = cls_score^alpha * IoU^beta
# cost = -alignment_metric (minimize cost = maximize alignment)
```

The spatial filtering (`_get_in_boxes_info`) should be applied BEFORE Hungarian matching to reduce the cost matrix size. The cost matrix is `[num_gt, num_candidates]`. Since `num_gt << num_candidates`, the result assigns exactly 1 prediction per GT.

**Implementation sketch:**
```python
from scipy.optimize import linear_sum_assignment

class HungarianAssigner:
    def __init__(self, alpha: float, beta: float, num_classes: int) -> None:
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    @torch.no_grad()
    def assign(
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
        # 1. Spatial filtering (reuse _get_in_boxes_info)
        # 2. Compute pairwise IoU [num_gt, num_candidates]
        # 3. Compute cls score for each GT class [num_gt, num_candidates]
        # 4. alignment = cls_score^alpha * IoU^beta
        # 5. cost = -alignment (+ spatial penalty)
        # 6. row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        # 7. Convert to 5-tuple return contract
        ...
```

### Pattern 2: Combined Loss (DUAL-04)

**What:** Training loss sums both head losses with a weighting factor.

**Formula:** `L_total = L_o2m + lambda_o2o * L_o2o`

**Implementation:** The `_get_losses` method computes O2M loss as it does today, then additionally runs the O2O assignment and computes O2O losses. Both are combined before returning.

**Config addition needed:** `lambda_o2o: float = 1.0` in DINOXConfig.

### Pattern 3: ONNX Export O2O-Only Path (DUAL-05)

**What:** During inference/export, only the O2O prediction layers are used.

**How:** Modify `_forward_inference` to check `self.use_dual_head`. When True, use `cls_preds_o2o[k]`, `reg_preds_o2o[k]`, `obj_preds_o2o[k]` instead of the O2M prediction layers. The objectness output should be constant 1.0 (since O2O doesn't need objectness filtering, but downstream evaluation code expects the `[cx, cy, w, h, obj_conf, cls...]` format).

**Constant-1 objectness approach:**
```python
# In _forward_inference, when use_dual_head:
obj_output = torch.ones_like(obj_output_o2o)  # constant 1
```

This ensures backward compatibility with `get_predictions()` which multiplies `scores = cls_conf.max(dim=1) * obj_conf`.

### Pattern 4: Consistent Alignment Metric (DUAL-03)

**What:** Both O2M and O2O heads use identical alpha/beta values in their matching metric.

**Implementation:** The `HungarianAssigner` receives the same `tal_alpha` and `tal_beta` values as the TAL assigner (or equivalent SimOTA cost formulation). Config validation in `DINOXConfig` should enforce this.

**Note from YOLOv10 paper:** The consistent metric ensures "the highest-ranked positive sample in the O2M branch is also the best for one-to-one head." Using `r=1` (identical hyperparameters) is the standard approach.

### Anti-Patterns to Avoid

- **Separate O2O nn.Module class:** Would complicate weight loading, ONNX export, and shared conv stack reuse. Keep O2O as additional ModuleLists within DINOXHead.
- **Running O2O conv stacks during O2M-only inference:** When `use_dual_head=False`, the O2O layers should not exist or be used. Gate creation on the flag.
- **Using the O2M objectness prediction in O2O path:** The O2O head should output constant-1 objectness so confidence = cls_score only (NMS-free means each prediction is already the best).
- **Hungarian matching on full anchor set:** Always apply spatial filtering first. With ~8400 anchors and ~5 GTs, the full cost matrix is unnecessary and wasteful. Spatial filtering typically reduces to ~100-500 candidates.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LSAP solver | Custom Hungarian impl | `scipy.optimize.linear_sum_assignment` | O(n^3) algorithm with edge cases; scipy's C implementation is fast and correct |
| IoU computation | New IoU function | Existing `_bboxes_iou` from dinox_head.py | Already tested, handles cxcywh and xyxy |
| Spatial filtering | New in-box checks | Existing `_get_in_boxes_info` (or TAL's copy) | Already validated, handles center radius |
| Cost matrix alignment metric | Custom cost | Same `cls_score^alpha * IoU^beta` as TAL | Consistency requirement (DUAL-03) |

**Key insight:** The Hungarian assigner's cost matrix should reuse the same alignment metric computation as TAL. The only difference is HOW the cost matrix is solved (Hungarian 1:1 vs TAL top-k).

## Common Pitfalls

### Pitfall 1: scipy on GPU tensors
**What goes wrong:** `linear_sum_assignment` takes numpy arrays. Forgetting `.cpu().numpy()` causes errors.
**Why it happens:** Cost matrix is computed on GPU, scipy needs CPU numpy.
**How to avoid:** Explicitly move cost to CPU before calling scipy: `cost_np = cost.detach().cpu().numpy()`. Convert row/col indices back to torch tensors on the correct device.
**Warning signs:** RuntimeError about numpy/cuda incompatibility.

### Pitfall 2: O2O num_fg always equals num_gt
**What goes wrong:** Unlike O2M which can match multiple predictions per GT, O2O always matches exactly `num_gt` predictions (1 per GT). If code assumes `num_fg` can be larger, the loss normalization will be wrong.
**Why it happens:** Hungarian matching is 1:1 by definition.
**How to avoid:** This is correct behavior -- just ensure the combined loss normalization accounts for O2M and O2O having different num_fg values. Normalize each head's loss by its own num_fg before combining.

### Pitfall 3: Gradient flow through O2O predictions
**What goes wrong:** The assigner runs under `@torch.no_grad()` (correct), but the LOSS computation must use the O2O predictions WITH gradients.
**Why it happens:** Assignment is stop-gradient (as in DETR), but losses need gradients for backprop.
**How to avoid:** Run feature extraction and O2O prediction with gradients enabled. Only the assignment step (cost matrix + Hungarian) is no-grad.

### Pitfall 4: ONNX export tracing with dual head
**What goes wrong:** ONNX tracing follows the code path taken during `torch.onnx.export`. If the inference path doesn't cleanly select O2O-only, both heads' parameters get traced.
**Why it happens:** Python control flow based on `self.use_dual_head` flag is fine for tracing (it's a constant during export), but must be clean.
**How to avoid:** The `_forward_inference` method should have a clean branch: when `use_dual_head=True`, ONLY reference O2O prediction layers. The O2M prediction layers should not appear in the traced graph.
**Warning signs:** ONNX model size roughly doubled (both heads' params included).

### Pitfall 5: Weight loading with new O2O layers
**What goes wrong:** Pretrained checkpoints don't have O2O layer weights. `load_state_dict(strict=True)` will fail.
**Why it happens:** O2O prediction layers are new parameters not in existing checkpoints.
**How to avoid:** The existing `_load_weights` method already uses `strict=False` and logs mismatches. O2O layers will initialize randomly. Consider copying O2M weights to O2O as initialization in `__init__` for faster convergence.

### Pitfall 6: Empty GT handling in Hungarian matching
**What goes wrong:** `linear_sum_assignment` with an empty cost matrix (0 GTs) crashes or returns unexpected results.
**Why it happens:** Edge case when an image has no ground truth objects.
**How to avoid:** Check `num_gt == 0` before calling Hungarian matching and return the zero-fg 5-tuple early (same pattern as SimOTA and TAL).

## Code Examples

### Hungarian Assigner Return Contract
```python
# Source: Existing codebase pattern from tal.py and dinox_head.py _get_assignments
# The 5-tuple return contract ALL assigners must follow:
(
    gt_matched_classes,       # [num_fg] class labels for matched GTs
    fg_mask,                  # [total_anchors] bool mask of foreground anchors
    pred_ious_this_matching,  # [num_fg] IoU with matched GT
    matched_gt_inds,          # [num_fg] index of matched GT per fg anchor
    num_fg,                   # int: number of foreground anchors
)
# For O2O: num_fg == num_gt (exactly 1 prediction per GT)
```

### Cost Matrix for Hungarian Matching
```python
# Source: YOLOv10 paper + existing TAL alignment metric
# Uses SAME alpha/beta as O2M head for consistency (DUAL-03)
pair_wise_ious = _bboxes_iou(gt_bboxes, bbox_preds_cand, xyxy=False)  # [G, C]
cls_sigmoid = cls_preds_cand.sigmoid()  # [C, num_classes]
gt_one_hot = F.one_hot(gt_classes.long(), num_classes).float()  # [G, num_classes]
bbox_scores = (cls_sigmoid.unsqueeze(1) * gt_one_hot.unsqueeze(0)).sum(-1).T  # [G, C]

alignment = bbox_scores.pow(alpha) * pair_wise_ious.pow(beta)
cost = -alignment  # minimize cost = maximize alignment
cost += 1e6 * (~is_in_boxes_and_center)  # spatial constraint

# Solve
row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
```

### O2O Inference Path with Constant Objectness
```python
# Source: Architecture requirement DUAL-05
# In _forward_inference, when use_dual_head=True:
for k, x_in in enumerate(xin):
    x = self.stems[k](x_in)
    cls_feat = self.cls_convs[k](x)
    cls_output = self.cls_preds_o2o[k](cls_feat)  # O2O cls pred

    reg_feat = self.reg_convs[k](x)
    reg_output = self.reg_preds_o2o[k](reg_feat)  # O2O reg pred

    # Constant-1 objectness for backward compat with eval code
    obj_output = torch.ones(
        reg_output.shape[0], 1, reg_output.shape[2], reg_output.shape[3],
        device=reg_output.device, dtype=reg_output.dtype,
    )

    output = torch.cat([reg_output, obj_output, cls_output.sigmoid()], 1)
```

### DINOXConfig Validation for Dual Head
```python
# Source: Existing config.py validation pattern
# Add to model_validator:
if self.use_dual_head and not self.use_soft_labels:
    raise ValueError("use_dual_head=True requires use_soft_labels=True")
# Already exists in config.py!

# NEW: Add lambda_o2o config field
lambda_o2o: float = 1.0
```

### O2O Weight Initialization from O2M
```python
# Copy O2M prediction weights to O2O for better initialization
if use_dual_head:
    for i in range(len(in_channels)):
        self.cls_preds_o2o[i].load_state_dict(self.cls_preds[i].state_dict())
        self.reg_preds_o2o[i].load_state_dict(self.reg_preds[i].state_dict())
        self.obj_preds_o2o[i].load_state_dict(self.obj_preds[i].state_dict())
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NMS post-processing | Dual O2M/O2O heads | YOLOv10 (May 2024) | Eliminates NMS latency (~4ms savings) |
| Hungarian matching for O2O | Top-1 selection | YOLOv10 (May 2024) | Same accuracy, less training overhead |
| Separate O2O/O2M architectures | Shared backbone + conv stacks | YOLOv10 (May 2024) | Minimal parameter increase |

**Note on top-1 vs Hungarian:** The YOLOv10 paper found that top-1 selection (picking the single highest-alignment-metric prediction per GT) achieves the same accuracy as full Hungarian matching with less training time. However, the requirements specify Hungarian matching (DUAL-02: "O2O label assignment uses Hungarian matching (scipy)"), so we implement Hungarian. Top-1 could be a future optimization.

## Open Questions

1. **O2O head initialization strategy**
   - What we know: O2O layers are randomly initialized when loading pretrained O2M weights.
   - What's unclear: Whether copying O2M weights to O2O (vs random init) improves convergence.
   - Recommendation: Copy O2M weights to O2O at init time. This is cheap and follows the intuition that O2O should start near O2M's learned representation.

2. **lambda_o2o default value**
   - What we know: YOLOv10 uses lambda=1.0. Some papers use 0.5 or 0.25 for auxiliary heads.
   - What's unclear: Optimal value for this specific architecture.
   - Recommendation: Default to 1.0 (YOLOv10 default), expose as config parameter for tuning.

3. **Whether to keep obj_preds_o2o layers or hardcode constant-1**
   - What we know: O2O output needs constant-1 objectness for eval code compatibility.
   - What's unclear: Whether to still have obj_preds_o2o layers (trained but ignored at inference) or skip them entirely.
   - Recommendation: Skip obj_preds_o2o entirely. The O2O head doesn't need objectness -- output constant 1 during inference. During training, compute O2O loss without objectness loss (or with trivial constant target). This saves parameters and simplifies the architecture.

## Sources

### Primary (HIGH confidence)
- Codebase: `dinox_head.py` - Full DINOXHead implementation with SimOTA/TAL assigner dispatch, loss computation, inference/training branching
- Codebase: `tal.py` - TaskAlignedAssigner with alignment metric `cls^alpha * IoU^beta`
- Codebase: `config.py` - DINOXConfig with existing `use_dual_head` flag and validation
- Codebase: `rfdetr/matcher.py` - Existing `HungarianMatcher` using `scipy.optimize.linear_sum_assignment`
- Codebase: `dinox_lightning.py` - Lightning wrapper with ONNX export, `_export_mode` flag
- [YOLOv10 paper (arxiv)](https://arxiv.org/html/2405.14458v1) - Dual label assignment architecture, consistent matching metric, O2O/O2M training strategy

### Secondary (MEDIUM confidence)
- [scipy linear_sum_assignment docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) - API reference for Hungarian solver
- [YOLOv10 Ultralytics docs](https://docs.ultralytics.com/models/yolov10/) - Implementation details and benchmarks
- [LearnOpenCV YOLOv10 explainer](https://learnopencv.com/yolov10/) - Architecture walkthrough

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - scipy already in project, pattern proven in rfdetr/matcher.py
- Architecture: HIGH - DINOXHead structure is well understood, O2O addition is mechanical
- Hungarian matching: HIGH - Well-documented algorithm, scipy API is stable, DETR matcher exists as reference
- ONNX export: HIGH - Existing `_forward_inference` branching pattern is clean, `_export_mode` flag works
- Loss combination: HIGH - Straightforward additive combination with lambda weight
- Pitfalls: HIGH - Based on actual codebase patterns and known scipy/torch interop issues

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable domain, no fast-moving dependencies)
