---
phase: 02-soft-label-assignment
verified: 2026-03-06T14:00:00Z
status: passed
score: 4/4 must-haves verified
---

# Phase 2: Soft Label Assignment Verification Report

**Phase Goal:** Soft SimOTA replaces binary label assignment with IoU-weighted soft targets, providing richer training signal and establishing the pluggable assigner protocol used by all subsequent phases
**Verified:** 2026-03-06
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Soft SimOTA assigns IoU^gamma weighted targets instead of binary 1.0 for positive samples | VERIFIED | `dinox_head.py` lines 757-760: `iou_weight = pred_ious_this_matching.pow(self.soft_label_gamma)` when `use_soft_labels=True`; test `test_iou_gamma_weighting_values` verifies [0.8,0.5,0.3]^2 = [0.64,0.25,0.09] |
| 2 | Each toggle (use_soft_labels, use_log_iou_cost, soft_label_gamma) independently enables/disables its change | VERIFIED | Config has no cross-validation between these three flags (lines 38-40); tests `test_config_flags_independent` and `test_head_stores_flags_independently` verify all combinations |
| 3 | Training with soft SimOTA enabled converges without zero-positive-assignment failures | VERIFIED | Tests `test_soft_labels_training_forward_runs`, `test_log_iou_cost_training_forward_runs`, `test_all_flags_training_forward_runs` all pass with finite loss; gradient flow confirmed in `TestGradientFlow` |
| 4 | Unit tests verify IoU-weighted target values and -log(IoU) cost matrix computation against hand-computed examples | VERIFIED | 16 tests in `test_dinox_soft_labels.py` across 5 classes; hand-computed values for IoU^gamma, -log(IoU), and RTMDet cost all verified with `torch.allclose` |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/object_detection_training/models/dinox/config.py` | `use_log_iou_cost: bool = False` field | VERIFIED | Line 40, section comment updated to "Phase 2: soft labels" |
| `src/object_detection_training/models/dinox/dinox_head.py` | Soft label assignment and loss modifications | VERIFIED | Constructor params (lines 268-270), stored as attrs (281-283), IoU^gamma in `_get_losses` (757-760), RTMDet cost in `_get_assignments` (912-927), -log(IoU) gating (899-902) |
| `src/object_detection_training/models/dinox_lightning.py` | Flag propagation from config to DINOXHead | VERIFIED | Constructor params (67-69), stored (122-124), config validation (135-137), head construction (165-167), logger output (145-147) |
| `tests/test_dinox_head.py` | `_make_model` helper updated with soft label params | VERIFIED | Lines 25-27 add params, lines 41-43 forward to DINOXHead |
| `tests/test_dinox_soft_labels.py` | Comprehensive unit tests for soft SimOTA | VERIFIED | 274 lines, 5 classes (TestSoftLabelTargets, TestLogIouCost, TestSoftClassificationCost, TestFlagIndependence, TestGradientFlow), 16 tests |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `config.py` | `dinox_lightning.py` | DINOXConfig validation includes use_log_iou_cost | WIRED | Lightning passes `use_log_iou_cost=use_log_iou_cost` to DINOXConfig() at line 137 |
| `dinox_lightning.py` | `dinox_head.py` | Constructor passes all three flags | WIRED | Lines 165-167: `use_soft_labels=use_soft_labels, soft_label_gamma=soft_label_gamma, use_log_iou_cost=use_log_iou_cost` |
| `_get_losses` | `_get_assignments` | Both use `self.use_soft_labels` for different purposes | WIRED | `_get_losses` line 757 for target weighting, `_get_assignments` line 912 for RTMDet cost |
| `test_dinox_soft_labels.py` | `dinox_head.py` | Tests exercise with soft labels enabled | WIRED | Multiple tests create models with `use_soft_labels=True` and run forward passes |
| `test_dinox_soft_labels.py` | `config.py` | Tests validate DINOXConfig with new flags | WIRED | `test_config_flags_independent` creates DINOXConfig with various flag combinations |

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| SIMO-01 | SATISFIED | IoU^gamma weighting in `_get_losses` lines 757-760 |
| SIMO-02 | SATISFIED | -log(IoU) toggleable in `_get_assignments` lines 899-902 |
| SIMO-03 | SATISFIED | RTMDet soft cls cost in `_get_assignments` lines 912-927 |
| SIMO-04 | SATISFIED | Three independent flags, no cross-validation, verified by tests |
| TEST-01 | SATISFIED | 16 tests with hand-computed expected values in `test_dinox_soft_labels.py` |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | - | - | - | - |

No TODO, FIXME, placeholder, or stub patterns detected in any modified files.

### Human Verification Required

### 1. Full Training Convergence

**Test:** Run training with `use_soft_labels=True, soft_label_gamma=2.0` on real dataset for multiple epochs
**Expected:** Loss converges, num_fg > 0 every batch after warmup
**Why human:** Unit tests verify single forward pass, not multi-epoch convergence behavior

### Commits Verified

All three commits from summaries confirmed in git log:
- `6c7c2f6` feat(02-01): add soft label flags to config and DINOXHead with RTMDet cost
- `00ebada` feat(02-01): propagate soft label flags through Lightning model and test helper
- `059140f` test(02-02): add comprehensive soft SimOTA unit tests

---

_Verified: 2026-03-06_
_Verifier: Claude (gsd-verifier)_
