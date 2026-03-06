---
phase: 03-loss-assignment-improvements
verified: 2026-03-06T15:00:00Z
status: passed
score: 4/4 must-haves verified
---

# Phase 3: Loss & Assignment Improvements Verification Report

**Phase Goal:** MAL amplifies gradient signal for low-quality matches and TAL provides an alternative assigner for ablation comparison, completing the label assignment and loss toolkit
**Verified:** 2026-03-06T15:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | MAL loss is equivalent to standard BCE when matchability=1.0, and produces amplified gradients when matchability is low, both verified by unit test | VERIFIED | `test_mal_weight_at_one` asserts weight=1.0 at matchability=1.0; `test_mal_weight_monotonic` proves decreasing weights as matchability increases; `test_mal_weight_at_zero` asserts weight=2.0 at matchability=0.0. Formula `(1-m)^2+1.0` confirmed in mal.py line 64. |
| 2 | TAL is swappable with SimOTA via a single config flag change, and ablation configs E3/E4 use TAL correctly | VERIFIED | `dinox_head.py` line 766 dispatches on `self.assigner_type == "tal"`. E3 yaml has `assigner: tal` without soft labels; E4 yaml has `assigner: tal` with `use_soft_labels: true`. `test_tal_contract_matches_simota` proves both produce same loss dict keys with finite losses. Parameterization tests verify E3/E4 config loading. |
| 3 | MAL integrates with soft SimOTA targets without circular gradient dependency | VERIFIED | `dinox_head.py` lines 804-815: `pred_ious_this_matching` comes from assigner (TAL uses `@torch.no_grad()`, SimOTA uses discrete matching_matrix). Only `cls_score_for_gt` carries gradients through `matchability_score`. `test_mal_integration_forward_backward` confirms finite loss and successful backward pass with both MAL and soft labels enabled. |
| 4 | Unit tests for MAL verify gradient amplification behavior and BCE equivalence at boundary conditions | VERIFIED | 11 MAL tests in `test_dinox_mal.py`: 4 matchability_score tests (perfect match, zero IoU, formula, batch), 4 mal_weight tests (boundary at 1.0, boundary at 0.0, monotonicity, range [1.0, 2.0]), 3 gradient tests (cls_scores grad flow, ious transparency, integration forward+backward). All 28 tests pass. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/object_detection_training/models/dinox/mal.py` | matchability_score and mal_weight functions | VERIFIED | 66 lines, two pure functions with full docstrings and type annotations. Correct formulas. |
| `src/object_detection_training/models/dinox/tal.py` | TaskAlignedAssigner class | VERIFIED | 317 lines. `assign()` method matches `_get_assignments` signature and returns same 5-tuple. Self-contained with duplicated `_bboxes_iou` and `_get_in_boxes_info`. |
| `tests/test_dinox_mal.py` | MAL unit tests | VERIFIED | 187 lines, 11 tests covering boundary conditions, formula, monotonicity, range, gradient flow, integration. |
| `tests/test_dinox_tal.py` | TAL unit tests | VERIFIED | 263 lines, 7 tests covering output contract, shapes, top-k limits, conflict resolution, no-candidates edge case, SimOTA contract compatibility, integration. |
| `src/object_detection_training/conf/models/dinox_m_e3.yaml` | E3 ablation config (baseline + TAL) | VERIFIED | Inherits from dinox_m_baseline, sets `assigner: tal` with TAL hyperparams, no soft labels. |
| `src/object_detection_training/conf/models/dinox_m_e4.yaml` | E4 ablation config (soft labels + TAL) | VERIFIED | Inherits from dinox_m_baseline, sets `assigner: tal` with TAL hyperparams and `use_soft_labels: true`. |
| `src/object_detection_training/models/dinox/config.py` | Updated DINOXConfig with TAL params | VERIFIED | `assigner: Literal["simota", "tal"]`, `tal_topk`, `tal_alpha`, `tal_beta` fields present. Validator enforces `use_mal requires use_soft_labels`. |
| `src/object_detection_training/models/dinox/dinox_head.py` | MAL integration in _get_losses and TAL dispatch | VERIFIED | Lines 766-781: assigner dispatch. Lines 804-815: MAL weighting applied per-anchor after BCE. |
| `src/object_detection_training/models/dinox_lightning.py` | Flag propagation for use_mal, assigner, tal_* | VERIFIED | Constructor accepts and stores all params, passes to DINOXConfig and DINOXHead. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| dinox_head.py | mal.py | `from .mal import mal_weight, matchability_score` | WIRED | Imported and called in _get_losses (lines 811-814) |
| dinox_head.py | tal.py | `from .tal import TaskAlignedAssigner` | WIRED | Imported and instantiated in __init__ (line 352), called in _get_losses (line 773) |
| dinox_lightning.py | dinox_head.py | Constructor param propagation | WIRED | `use_mal`, `assigner_type=assigner`, `tal_topk/alpha/beta` all passed through (lines 194-199) |
| tests/test_dinox_mal.py | mal.py | `from object_detection_training.models.dinox.mal import` | WIRED | Imports and tests both `matchability_score` and `mal_weight` |
| tests/test_dinox_tal.py | tal.py | `from object_detection_training.models.dinox.tal import TaskAlignedAssigner` | WIRED | Imports and tests `TaskAlignedAssigner.assign()` |
| dinox_m_e3.yaml | dinox_m_baseline.yaml | Hydra defaults inheritance | WIRED | `defaults: [dinox_m_baseline, _self_]` verified by parameterization test |
| dinox_m_e4.yaml | dinox_m_baseline.yaml | Hydra defaults inheritance | WIRED | `defaults: [dinox_m_baseline, _self_]` verified by parameterization test |

### Anti-Patterns Found

None. No TODOs, FIXMEs, placeholders, or empty implementations found in any phase artifacts.

### Test Results

All 28 tests pass (5.34s):
- 11 MAL tests (matchability + mal_weight + gradients)
- 7 TAL tests (contract + behavior + integration)
- 10 parameterization tests (including E3/E4)

### Human Verification Required

None required. All success criteria are programmatically verifiable through unit tests and code inspection.

---

_Verified: 2026-03-06T15:00:00Z_
_Verifier: Claude (gsd-verifier)_
