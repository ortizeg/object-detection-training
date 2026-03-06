---
phase: 03-loss-assignment-improvements
plan: 02
subsystem: testing
tags: [mal, tal, unit-tests, hydra-config, ablation, pytorch]

# Dependency graph
requires:
  - phase: 03-loss-assignment-improvements
    plan: 01
    provides: MAL/TAL modules, config flags, assigner dispatch in DINOXHead
provides:
  - TEST-03 unit tests for MAL boundary conditions and gradient behavior
  - TAL unit tests with output contract verification against SimOTA
  - E3/E4 Hydra ablation configs for TAL experiments
affects: [phase-05 (distillation experiments may reference E3/E4 configs)]

# Tech tracking
tech-stack:
  added: []
  patterns: [duplicated-test-helpers-per-convention, hydra-config-inheritance-via-defaults]

key-files:
  created:
    - tests/test_dinox_mal.py
    - tests/test_dinox_tal.py
    - src/object_detection_training/conf/models/dinox_m_e3.yaml
    - src/object_detection_training/conf/models/dinox_m_e4.yaml
  modified:
    - tests/test_dinox_parameterization.py
    - src/object_detection_training/conf/models/dinox_base.yaml

key-decisions:
  - "Added soft label, MAL, and assigner defaults to dinox_base.yaml for Hydra schema completeness"

patterns-established:
  - "E3/E4 ablation configs inherit from dinox_m_baseline via Hydra defaults chain"

# Metrics
duration: 5min
completed: 2026-03-06
---

# Phase 3 Plan 2: MAL/TAL Tests and E3/E4 Configs Summary

**TEST-03 MAL/TAL unit tests (18 tests) plus E3/E4 Hydra ablation configs for TAL experiments**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-06T14:29:59Z
- **Completed:** 2026-03-06T14:35:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- 11 MAL tests: matchability boundary/formula, mal_weight monotonicity/range/boundary, gradient flow, integration forward+backward
- 7 TAL tests: output tuple structure/shapes, top-k limits, conflict resolution, no-candidates edge case, SimOTA contract compatibility, integration
- E3 config (baseline + TAL) and E4 config (soft labels + TAL) for ablation experiments
- Parameterization tests verify both configs load and validate correctly
- Full suite: 524 passed, 0 failures

## Task Commits

Each task was committed atomically:

1. **Task 1: MAL and TAL unit tests** - `2d95b37` (test)
2. **Task 2: E3/E4 Hydra configs and parameterization tests** - `c84795f` (feat)

## Files Created/Modified
- `tests/test_dinox_mal.py` - TEST-03 unit tests for matchability_score and mal_weight
- `tests/test_dinox_tal.py` - TAL unit tests with contract verification against SimOTA
- `src/object_detection_training/conf/models/dinox_m_e3.yaml` - E3 ablation config (baseline + TAL)
- `src/object_detection_training/conf/models/dinox_m_e4.yaml` - E4 ablation config (soft labels + TAL)
- `tests/test_dinox_parameterization.py` - Added E3/E4 config validation test cases
- `src/object_detection_training/conf/models/dinox_base.yaml` - Added soft label, MAL, and assigner defaults

## Decisions Made
- Added Phase 2/3 feature flag defaults (use_soft_labels, use_mal, assigner, tal_*) to dinox_base.yaml for Hydra schema completeness -- required for E3/E4 configs to compose correctly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added missing feature flag defaults to dinox_base.yaml**
- **Found during:** Task 2 (E3/E4 Hydra config creation)
- **Issue:** E3/E4 configs inherit from dinox_m_baseline which chains through dinox_base.yaml. OmegaConf strict mode rejected `use_soft_labels`, `assigner`, and `tal_*` fields because they weren't in the schema.
- **Fix:** Added soft label, MAL, and assigner defaults to dinox_base.yaml (all defaulting to off/simota for backward compatibility)
- **Files modified:** `src/object_detection_training/conf/models/dinox_base.yaml`
- **Verification:** All 524 tests pass, no regressions, E3/E4 configs compose correctly
- **Committed in:** c84795f (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Schema fix necessary for Hydra config composition. No scope creep -- all defaults match existing behavior.

## Issues Encountered
None beyond the deviation documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 complete: MAL/TAL modules (03-01) and tests/configs (03-02) all in place
- All feature flags default to off, existing training behavior unchanged
- E3/E4 configs ready for ablation experiments
- 524 tests passing, lint and typecheck clean

---
*Phase: 03-loss-assignment-improvements*
*Completed: 2026-03-06*
