---
phase: 04-nms-free-dual-head
plan: 02
subsystem: testing
tags: [hungarian-matching, dual-head, unit-tests, hydra-configs, ablation]

# Dependency graph
requires:
  - phase: 04-nms-free-dual-head
    plan: 01
    provides: "HungarianAssigner, O2O dual head layers, combined loss, config wiring"
provides:
  - "13 unit tests covering DUAL-01 through DUAL-05 and TEST-05"
  - "E5 ablation config: SimOTA + soft labels + dual head"
  - "E6 ablation config: TAL + soft labels + dual head"
affects: [phase-05-distillation]

# Tech tracking
tech-stack:
  added: []
  patterns: [dual-head-test-pattern, ablation-config-pattern]

key-files:
  created:
    - tests/test_dinox_dual_head.py
    - src/object_detection_training/conf/models/dinox_m_e5.yaml
    - src/object_detection_training/conf/models/dinox_m_e6.yaml

key-decisions:
  - "Used width=0.25 and depth=0.33 for fast test execution (smallest viable model)"
  - "400-anchor grid with stride=8 to ensure spatial filtering covers GT positions"
  - "Duplicated test helpers per project convention (no cross-test imports)"

patterns-established:
  - "Dual head test pattern: separate tests for Hungarian, O2O shape, objectness, parameters, config"
  - "Ablation config naming: E5=SimOTA+dual, E6=TAL+dual (extends E3/E4 pattern)"

# Metrics
duration: 7min
completed: 2026-03-06
---

# Phase 4 Plan 2: Dual Head Tests and Ablation Configs Summary

**13 unit tests verifying Hungarian 1:1 matching, O2O inference, parameter independence, and E5/E6 Hydra ablation configs for NMS-free dual head experiments**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-06T15:09:30Z
- **Completed:** 2026-03-06T15:16:31Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- 13 comprehensive tests covering all DUAL requirements (DUAL-01 through DUAL-05) and TEST-05
- Hungarian matching verified: strict 1:1 assignment with num_fg == num_gt, unique GT indices
- O2O inference verified: output shape matches O2M, constant-1 objectness, separate parameters
- E5 (SimOTA + dual head) and E6 (TAL + dual head) ablation configs validated via Hydra composition
- Full test suite passes with 537 tests, no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Unit tests for dual head (TEST-05)** - `fbba88b` (test)
2. **Task 2: E5/E6 ablation Hydra configs** - `f8c2236` (feat)

## Files Created/Modified
- `tests/test_dinox_dual_head.py` - 13 tests covering Hungarian, O2O, config validation, Hydra configs
- `src/object_detection_training/conf/models/dinox_m_e5.yaml` - SimOTA + soft labels + dual head config
- `src/object_detection_training/conf/models/dinox_m_e6.yaml` - TAL + soft labels + dual head config

## Decisions Made
- Used width=0.25 and depth=0.33 for fast test execution (smallest viable model)
- Used 400-anchor grid at stride=8 to ensure spatial filtering overlap with GT positions for Hungarian tests
- Duplicated test helpers (_make_model, _make_targets) per project convention (no cross-test imports)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed anchor grid coverage for Hungarian tests**
- **Found during:** Task 1 (Hungarian 1:1 assignment test)
- **Issue:** Initial test used 100 anchors (10x10 grid at stride=8 = 80px coverage) but GT boxes at 100-200px were outside grid, causing 0 matches
- **Fix:** Repositioned GT boxes within grid coverage (60-120px) and increased to 400 anchors (20x20 grid = 160px coverage)
- **Files modified:** tests/test_dinox_dual_head.py
- **Verification:** Hungarian returns exactly num_gt matches with unique GT indices
- **Committed in:** fbba88b (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix in test setup)
**Impact on plan:** Test setup fix necessary for correct verification. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 complete: dual head architecture fully implemented and tested
- All 537 tests pass (13 new dual head tests + 524 existing)
- Ready for Phase 5: Knowledge Distillation

---
*Phase: 04-nms-free-dual-head*
*Completed: 2026-03-06*
