---
phase: 02-soft-label-assignment
plan: 01
subsystem: models
tags: [simota, soft-labels, rtmdet, iou-cost, dinox]

# Dependency graph
requires:
  - phase: 01-dfl-foundation
    provides: DINOXHead with SimOTA assignment and DFL regression
provides:
  - use_log_iou_cost flag in DINOXConfig
  - IoU^gamma soft targets in _get_losses (SIMO-01)
  - RTMDet soft classification cost in _get_assignments (SIMO-03)
  - Toggleable -log(IoU) regression cost in _get_assignments (SIMO-02)
  - End-to-end flag propagation through DINOXLightningModel
affects: [02-soft-label-assignment, 03-mal, 04-dual-head]

# Tech tracking
tech-stack:
  added: []
  patterns: [soft-label-gated-code-paths, rtmdet-cost-formulation]

key-files:
  created: []
  modified:
    - src/object_detection_training/models/dinox/config.py
    - src/object_detection_training/models/dinox/dinox_head.py
    - src/object_detection_training/models/dinox_lightning.py
    - tests/test_dinox_head.py

key-decisions:
  - "Both -log(IoU) branches identical by design for ablation explicitness and future GIoU cost alternative"
  - "Non-in-place .sigmoid() in RTMDet path per research pitfall #1; autocast(enabled=False) per pitfall #2"

patterns-established:
  - "Soft label gating: if self.use_soft_labels branches in both _get_losses and _get_assignments"
  - "Flag propagation: config -> lightning model -> DINOXHead constructor -> instance attributes"

# Metrics
duration: 3min
completed: 2026-03-06
---

# Phase 2 Plan 1: Soft Label Assignment Summary

**Soft SimOTA with IoU^gamma targets, RTMDet classification cost, and toggleable -log(IoU) regression cost wired end-to-end through DINOXHead**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-06T13:22:53Z
- **Completed:** 2026-03-06T13:25:59Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added `use_log_iou_cost` flag to DINOXConfig alongside existing soft label flags
- Implemented IoU^gamma soft target weighting in `_get_losses` (SIMO-01)
- Implemented RTMDet soft classification cost BCE(P, Y_soft) * |Y_soft - P|^2 in `_get_assignments` (SIMO-03)
- Added toggleable -log(IoU) regression cost with flag gating in `_get_assignments` (SIMO-02)
- Propagated all three flags through DINOXLightningModel, DINOXConfig validation, and test helper
- All 13 existing tests pass unchanged (defaults preserve standard YOLOX-M behavior)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add use_log_iou_cost flag and soft label params to config and DINOXHead** - `6c7c2f6` (feat)
2. **Task 2: Propagate soft label flags through all DINOXHead construction sites** - `00ebada` (feat)

## Files Created/Modified
- `src/object_detection_training/models/dinox/config.py` - Added use_log_iou_cost field, updated section comment
- `src/object_detection_training/models/dinox/dinox_head.py` - Added 3 constructor params, soft target weighting in _get_losses, RTMDet cost and -log(IoU) gating in _get_assignments
- `src/object_detection_training/models/dinox_lightning.py` - Propagated 3 flags through constructor, config validation, head construction, and logger
- `tests/test_dinox_head.py` - Updated _make_model helper to accept and forward soft label flags

## Decisions Made
- Both -log(IoU) branches are intentionally identical -- the flag exists for ablation explicitness and future GIoU cost alternative (per SIMO-04)
- Used non-in-place .sigmoid() in RTMDet path to avoid corrupting shared tensors (research pitfall #1)
- Wrapped RTMDet path in autocast(enabled=False) matching YOLOX path (research pitfall #2)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All three soft label flags wired end-to-end and defaulting to False
- Ready for 02-02-PLAN.md (soft label tests and Hydra configs)
- MAL (Phase 3) and dual head (Phase 4) depend on use_soft_labels being available

## Self-Check: PASSED

All 4 modified files verified on disk. Both commits (6c7c2f6, 00ebada) confirmed in git log.

---
*Phase: 02-soft-label-assignment*
*Completed: 2026-03-06*
