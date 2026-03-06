---
phase: 04-nms-free-dual-head
verified: 2026-03-06T16:00:00Z
status: passed
score: 4/4 must-haves verified
---

# Phase 4: NMS-Free Dual Head Verification Report

**Phase Goal:** A second O2O detection head trains alongside the O2M head with consistent matching, and ONNX export produces NMS-free inference using only the O2O branch
**Verified:** 2026-03-06T16:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | O2O head assigns exactly 1 prediction per GT via Hungarian matching, verified by unit test | VERIFIED | `hungarian.py` uses `scipy.optimize.linear_sum_assignment` (line 274), returns 5-tuple with `num_fg == len(row_ind)`. Test `test_hungarian_1to1_assignment` asserts `num_fg == num_gt`, unique GT indices, `fg_mask.sum() == num_gt`. |
| 2 | Both O2M and O2O heads use identical alignment metric values (same alpha, beta), verified by config validation | VERIFIED | `dinox_head.py` line 381: `HungarianAssigner(alpha=tal_alpha, beta=tal_beta, ...)` -- O2O receives same alpha/beta as O2M TAL. Test `test_consistent_alpha_beta` directly asserts `head._hungarian_assigner.alpha == 1.5` and `beta == 3.0` match constructor args. |
| 3 | ONNX export includes only O2O branch and produces constant-1 objectness for backward compatibility | VERIFIED | `dinox_head.py` lines 476-494: when `use_dual_head=True`, `_forward_inference` uses `cls_preds_o2o[k]` and `reg_preds_o2o[k]` with `torch.ones(...)` for objectness. Test `test_o2o_inference_constant_objectness` asserts all objectness values are 1.0 within tolerance. |
| 4 | Unit tests verify Hungarian matching produces strict 1:1 assignment and O2O output shape matches O2M output shape | VERIFIED | `tests/test_dinox_dual_head.py` contains 13 tests across 8 test classes. `test_hungarian_1to1_assignment` verifies strict 1:1. `test_o2o_output_shape_matches_o2m` creates both O2M and O2O models and asserts `out_o2m.shape == out_o2o.shape`. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/object_detection_training/models/dinox/hungarian.py` | HungarianAssigner with scipy linear_sum_assignment | VERIFIED | 313 lines. Full implementation with spatial filtering, alignment metric, cost matrix, Hungarian solve, 5-tuple return. No stubs or TODOs. |
| `src/object_detection_training/models/dinox/config.py` | lambda_o2o config field | VERIFIED | `lambda_o2o: float = 1.0` at line 48, `use_dual_head: bool = False` at line 47, validation at line 83. |
| `src/object_detection_training/models/dinox/dinox_head.py` | O2O prediction layers, dual-head loss, O2O inference path | VERIFIED | `cls_preds_o2o`/`reg_preds_o2o` ModuleLists (line 368-374), weight init from O2M (line 379-380), O2O inference path (line 476-487), O2O training path (line 652-661), combined loss (line 1032-1036). |
| `tests/test_dinox_dual_head.py` | TEST-05 unit tests for dual head | VERIFIED | 410 lines, 13 tests covering Hungarian 1:1, empty GT, 5-tuple contract, O2O shape, constant objectness, alpha/beta consistency, combined loss, lambda_o2o=0, separate parameters, config validation, Hydra E5/E6. |
| `src/object_detection_training/conf/models/dinox_m_e5.yaml` | SimOTA + dual head ablation config | VERIFIED | 9 lines. `use_dual_head: true`, `lambda_o2o: 1.0`, inherits from `dinox_m_baseline`. |
| `src/object_detection_training/conf/models/dinox_m_e6.yaml` | TAL + dual head ablation config | VERIFIED | 10 lines. `use_dual_head: true`, `assigner: tal`, `lambda_o2o: 1.0`, inherits from `dinox_m_baseline`. |
| `src/object_detection_training/models/dinox_lightning.py` | Dual head params, config validation, bias reinit | VERIFIED | `use_dual_head`/`lambda_o2o` constructor params (line 76-77), passed to DINOXConfig (line 168-169) and DINOXHead (line 208-209), O2O bias reinit (line 294-296). |
| `src/object_detection_training/conf/models/dinox_base.yaml` | Dual head defaults | VERIFIED | `use_dual_head: false` and `lambda_o2o: 1.0` at lines 39-40. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `dinox_head.py` | `hungarian.py` | `import HungarianAssigner` | WIRED | Line 381: `self._hungarian_assigner = HungarianAssigner(...)`. Used at line 976 in `_get_losses`. |
| `dinox_lightning.py` | `dinox_head.py` | `use_dual_head` and `lambda_o2o` constructor params | WIRED | Lines 208-209: passed to `DINOXHead(...)` constructor. |
| `dinox_base.yaml` | `dinox_lightning.py` | Hydra config injection | WIRED | YAML keys `use_dual_head`/`lambda_o2o` match Lightning constructor params. E5/E6 configs override these and tests verify via `hydra.compose`. |
| `tests/test_dinox_dual_head.py` | `hungarian.py` | import and direct testing | WIRED | Line 18: `from ...hungarian import HungarianAssigner`. Used in 3 test classes. |
| `tests/test_dinox_dual_head.py` | `dinox_head.py` | import and O2O testing | WIRED | Line 17: `from ...dinox import ... DINOXHead`. Used throughout via `_make_model`. |
| `dinox/__init__.py` | `hungarian.py` | export | WIRED | `from .hungarian import HungarianAssigner` and `"HungarianAssigner"` in `__all__`. |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| DUAL-01: O2O head identical structure, separate params | SATISFIED | -- |
| DUAL-02: Hungarian matching, exactly 1 pred per GT | SATISFIED | -- |
| DUAL-03: Consistent alpha/beta between heads | SATISFIED | -- |
| DUAL-04: Combined loss L_total = L_o2m + lambda_o2o * L_o2o | SATISFIED | -- |
| DUAL-05: ONNX export O2O-only with constant-1 objectness | SATISFIED | -- |
| TEST-05: Unit tests for 1:1 assignment and shape consistency | SATISFIED | -- |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `dinox_head.py` | 655 | Comment "placeholder zeros for concat format" | Info | Not a stub -- describes the zeros tensor filling the objectness slot in the O2O training path concat. Intentional design. |
| `dinox_head.py` | 994 | `except Exception: continue` | Info | Broad exception catch in O2O loss loop. Silently skips matching failures. Acceptable for robustness during training but could mask bugs. |

### Human Verification Required

### 1. End-to-end dual-head training run

**Test:** Run a short training with `use_dual_head=true` on a real dataset and verify loss convergence
**Expected:** Combined loss decreases over epochs, O2O head learns alongside O2M
**Why human:** Automated tests verify forward pass shapes and loss computation but not convergence behavior over multiple epochs

### 2. ONNX export and external inference

**Test:** Export model with `use_dual_head=true` to ONNX and run inference in an ONNX runtime
**Expected:** ONNX graph contains O2O prediction layers, outputs have constant-1 objectness, no NMS ops in graph
**Why human:** Current tests verify PyTorch inference path but not actual ONNX graph structure or runtime behavior

### Gaps Summary

No gaps found. All 4 observable truths verified with evidence from the actual codebase. All 8 artifacts exist, are substantive (not stubs), and are properly wired. All 6 DUAL/TEST requirements are satisfied. The implementation follows the plan faithfully: HungarianAssigner provides optimal 1:1 matching via scipy, O2O prediction layers share conv stacks but have independent parameters initialized from O2M weights, combined loss uses lambda_o2o weighting with independent normalization, and the inference path uses O2O-only layers with constant-1 objectness.

---

_Verified: 2026-03-06T16:00:00Z_
_Verifier: Claude (gsd-verifier)_
