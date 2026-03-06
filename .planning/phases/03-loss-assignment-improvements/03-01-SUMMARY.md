---
phase: 03-loss-assignment-improvements
plan: 01
subsystem: models
tags: [mal, tal, loss-weighting, label-assignment, pytorch]

# Dependency graph
requires:
  - phase: 02-soft-label-assignment
    provides: soft label targets (IoU^gamma) and pred_ious_this_matching detached from autograd
provides:
  - matchability_score and mal_weight functions for gradient amplification
  - TaskAlignedAssigner class as drop-in replacement for SimOTA
  - Config flags (use_mal, assigner="tal") with full propagation chain
affects: [03-02 (MAL/TAL tests), phase-05 (distillation may use TAL), ablation experiments E3/E4]

# Tech tracking
tech-stack:
  added: []
  patterns: [loss-weighting-via-per-anchor-multiplier, assigner-dispatch-pattern]

key-files:
  created:
    - src/object_detection_training/models/dinox/mal.py
    - src/object_detection_training/models/dinox/tal.py
  modified:
    - src/object_detection_training/models/dinox/config.py
    - src/object_detection_training/models/dinox/dinox_head.py
    - src/object_detection_training/models/dinox_lightning.py
    - src/object_detection_training/models/dinox/__init__.py

key-decisions:
  - "Duplicated _bboxes_iou in tal.py to avoid circular import (dinox_head imports tal, tal cannot import dinox_head)"
  - "Used bounded MAL weight (1-m)^2+1.0 in [1.0, 2.0] for numerical stability per research recommendation"

patterns-established:
  - "Assigner dispatch: self.assigner_type string selects between SimOTA and TAL in _get_losses"
  - "Loss weighting: MAL applies per-anchor weight after BCE computation, not inside it"

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 3 Plan 1: MAL/TAL Modules Summary

**Matchability-Aware Loss weighting and Task-Aligned Label Assignment with config-driven assigner dispatch in DINOXHead**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-06T14:21:08Z
- **Completed:** 2026-03-06T14:27:18Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- MAL module with matchability_score (IoU^gamma * cls_score^(1-gamma)) and bounded mal_weight ((1-m)^2+1.0)
- TaskAlignedAssigner with top-k alignment metric selection, same 5-tuple output as SimOTA
- Config-driven assigner dispatch (assigner="simota"|"tal") with full flag propagation through Lightning model
- All 504 existing tests pass, lint clean, typecheck clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Create MAL and TAL modules** - `cfdbfbc` (feat)
2. **Task 2: Wire MAL and TAL into config, head, and Lightning model** - `f2ea64a` (feat)

## Files Created/Modified
- `src/object_detection_training/models/dinox/mal.py` - matchability_score and mal_weight pure functions
- `src/object_detection_training/models/dinox/tal.py` - TaskAlignedAssigner class with self-contained spatial filtering
- `src/object_detection_training/models/dinox/config.py` - Added assigner="tal" Literal and TAL hyperparams
- `src/object_detection_training/models/dinox/dinox_head.py` - Assigner dispatch and MAL weighting in _get_losses
- `src/object_detection_training/models/dinox_lightning.py` - Propagated use_mal, assigner, tal_* params
- `src/object_detection_training/models/dinox/__init__.py` - Exported new symbols

## Decisions Made
- Duplicated `_bboxes_iou` in `tal.py` instead of importing from `dinox_head.py` to break circular import (dinox_head imports from tal via __init__.py)
- Used bounded MAL weight formulation `(1-m)^2 + 1.0` (range [1.0, 2.0]) per research recommendation for numerical stability

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Circular import between dinox_head and tal**
- **Found during:** Task 2 (wiring TAL into dinox_head)
- **Issue:** Plan specified importing `_bboxes_iou` from `dinox_head` into `tal.py`, but `dinox_head.py` now imports `TaskAlignedAssigner` from `tal.py`, creating a circular import
- **Fix:** Duplicated `_bboxes_iou` function in `tal.py` (same pattern as duplicating `_get_in_boxes_info` which plan already specified)
- **Files modified:** `src/object_detection_training/models/dinox/tal.py`
- **Verification:** All imports succeed, 504 tests pass
- **Committed in:** f2ea64a (Task 2 commit)

**2. [Rule 1 - Bug] MyPy no-any-return in mal_weight**
- **Found during:** Task 2 (typecheck verification)
- **Issue:** `(1.0 - matchability).pow(2) + 1.0` returns `Any` type per MyPy strict checking
- **Fix:** Assigned to typed local variable before returning
- **Files modified:** `src/object_detection_training/models/dinox/mal.py`
- **Verification:** `pixi run typecheck` passes with 0 errors
- **Committed in:** f2ea64a (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MAL and TAL modules ready for unit testing (03-02-PLAN.md)
- All flags default to off, so existing training behavior unchanged
- Config validation ensures use_mal requires use_soft_labels

---
*Phase: 03-loss-assignment-improvements*
*Completed: 2026-03-06*
