---
phase: 04-nms-free-dual-head
plan: 01
subsystem: models
tags: [hungarian-matching, dual-head, o2o, nms-free, scipy]

# Dependency graph
requires:
  - phase: 03-loss-assignment
    provides: "TAL assigner, soft labels, MAL weighting, DINOXConfig validation"
provides:
  - "HungarianAssigner for 1:1 label assignment"
  - "O2O prediction layers in DINOXHead (cls_preds_o2o, reg_preds_o2o)"
  - "Combined O2M + O2O training loss with lambda_o2o weighting"
  - "O2O-only inference path with constant-1 objectness"
  - "use_dual_head and lambda_o2o config wiring through Lightning and Hydra"
affects: [04-02, phase-05-distillation]

# Tech tracking
tech-stack:
  added: [scipy.optimize.linear_sum_assignment]
  patterns: [dual-head-shared-convs, hungarian-matching, o2o-inference-path]

key-files:
  created:
    - src/object_detection_training/models/dinox/hungarian.py
  modified:
    - src/object_detection_training/models/dinox/config.py
    - src/object_detection_training/models/dinox/dinox_head.py
    - src/object_detection_training/models/dinox/__init__.py
    - src/object_detection_training/models/dinox_lightning.py
    - src/object_detection_training/conf/models/dinox_base.yaml

key-decisions:
  - "Duplicated _bboxes_iou and _get_in_boxes_info in hungarian.py (same pattern as tal.py) to avoid circular imports"
  - "O2O has no objectness prediction or loss -- constant-1 objectness at inference per research recommendation"
  - "O2O prediction layers initialized by copying O2M weights for better convergence"
  - "O2O loss normalized by O2O's own num_fg independently from O2M normalization"

patterns-established:
  - "Dual head pattern: shared conv stacks, separate prediction Conv2d layers"
  - "Hungarian assigner follows same 5-tuple return contract as SimOTA and TAL"

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 4 Plan 1: NMS-Free Dual Head Summary

**HungarianAssigner with scipy linear_sum_assignment for 1:1 O2O matching, dual prediction layers sharing conv stacks, combined O2M+O2O training loss**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-06T15:01:13Z
- **Completed:** 2026-03-06T15:07:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- HungarianAssigner with optimal 1:1 matching via scipy, returning same 5-tuple contract as TAL/SimOTA
- O2O prediction layers (cls_preds_o2o, reg_preds_o2o) sharing conv stacks with O2M, initialized from O2M weights
- Combined training loss: L_o2m + lambda_o2o * (5 * iou_loss_o2o + cls_loss_o2o)
- O2O-only inference path with constant-1 objectness for NMS-free deployment
- Full config/Lightning/Hydra wiring with validation through DINOXConfig

## Task Commits

Each task was committed atomically:

1. **Task 1: HungarianAssigner + O2O head layers + config** - `8149f81` (feat)
2. **Task 2: Lightning wiring + Hydra config defaults** - `cbb557f` (feat)

## Files Created/Modified
- `src/object_detection_training/models/dinox/hungarian.py` - HungarianAssigner with 1:1 label assignment
- `src/object_detection_training/models/dinox/config.py` - Added lambda_o2o field
- `src/object_detection_training/models/dinox/dinox_head.py` - O2O layers, inference path, combined loss
- `src/object_detection_training/models/dinox/__init__.py` - Export HungarianAssigner
- `src/object_detection_training/models/dinox_lightning.py` - Dual head params, config validation, bias reinit
- `src/object_detection_training/conf/models/dinox_base.yaml` - Dual head defaults

## Decisions Made
- Duplicated spatial filtering helpers in hungarian.py (same pattern as tal.py) to avoid circular imports
- O2O skips objectness entirely -- no obj_preds_o2o, constant-1 at inference
- O2O weights initialized from O2M for warm-start convergence
- O2O loss uses its own num_fg for normalization (independent from O2M)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Dual head architecture complete and tested
- Ready for Plan 2: tests and experiment configs for dual head training
- All 524 existing tests pass (no regressions)

---
*Phase: 04-nms-free-dual-head*
*Completed: 2026-03-06*
