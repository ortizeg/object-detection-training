---
phase: 02-soft-label-assignment
plan: 02
subsystem: testing
tags: [simota, soft-labels, rtmdet, iou-cost, dinox, unit-tests]

# Dependency graph
requires:
  - phase: 02-soft-label-assignment
    provides: Soft SimOTA implementation in DINOXHead with IoU^gamma targets, RTMDet cost, -log(IoU) cost
provides:
  - Comprehensive unit tests for all 4 SIMO requirements
  - Hand-computed verification of IoU^gamma, -log(IoU), and RTMDet cost
  - Flag independence validation for use_soft_labels and use_log_iou_cost
  - Gradient flow verification through soft label paths
affects: [03-mal, 04-dual-head]

# Tech tracking
tech-stack:
  added: []
  patterns: [hand-computed-test-values, flag-independence-testing]

key-files:
  created:
    - tests/test_dinox_soft_labels.py
  modified: []

key-decisions:
  - "Duplicated _make_model/_make_targets helpers instead of cross-test imports (tests/__init__.py prevents module-level imports between test files)"

patterns-established:
  - "Soft label test pattern: hand-computed expected values verified with torch.allclose"
  - "Flag independence pattern: test each flag combination independently then together"

# Metrics
duration: 3min
completed: 2026-03-06
---

# Phase 2 Plan 2: Soft Label Tests Summary

**16 unit tests validating IoU^gamma targets, -log(IoU) cost, RTMDet soft classification cost, flag independence, and gradient flow for soft SimOTA**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-06T13:28:04Z
- **Completed:** 2026-03-06T13:31:41Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Created 16 unit tests across 5 test classes covering all 4 SIMO requirements
- Verified IoU^gamma=2.0 on [0.8, 0.5, 0.3] produces [0.64, 0.25, 0.09] with hand-computed values
- Verified -log(IoU) cost: -log(0.5) = 0.6931, -log(0.8) = 0.2231, -log(0.1) = 2.3026
- Verified RTMDet soft classification cost BCE(P, Y_soft) * |Y_soft - P|^2 matches hand-computed value
- Verified gamma=0 produces binary-equivalent targets (IoU^0 = 1.0)
- Confirmed use_soft_labels and use_log_iou_cost toggle independently in config and head
- Confirmed training forward passes succeed with each flag combination and all combined
- Confirmed gradient flows through cls_preds with soft label paths enabled
- All 477 existing tests continue to pass (12 skipped)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create unit tests for soft SimOTA label assignment** - `059140f` (test)

## Files Created/Modified
- `tests/test_dinox_soft_labels.py` - 16 tests in 5 classes: TestSoftLabelTargets, TestLogIouCost, TestSoftClassificationCost, TestFlagIndependence, TestGradientFlow

## Decisions Made
- Duplicated `_make_model` and `_make_targets` helpers from test_dinox_head.py rather than cross-importing, since tests/ has `__init__.py` which prevents bare module-level imports between test files

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed test helper import**
- **Found during:** Task 1
- **Issue:** Plan specified `from test_dinox_head import _make_model, _make_targets` but tests/__init__.py prevents cross-test module imports
- **Fix:** Duplicated the two small helper functions inline in test_dinox_soft_labels.py
- **Files modified:** tests/test_dinox_soft_labels.py
- **Verification:** All 16 tests pass
- **Committed in:** 059140f

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Trivial import fix, no scope change.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- TEST-01 requirement fully satisfied with 16 passing tests
- All soft label features (SIMO-01 through SIMO-04) verified with hand-computed values
- Ready for Phase 3 (MAL) which depends on use_soft_labels being available and tested

## Self-Check: PASSED

All files verified on disk. Commit 059140f confirmed in git log. Test file is 273 lines (exceeds 100-line minimum).

---
*Phase: 02-soft-label-assignment*
*Completed: 2026-03-06*
