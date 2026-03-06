---
phase: 06-scheduler-free-optimizer
plan: 01
subsystem: training
tags: [schedulefree, adamw, optimizer, lightning-hooks, scheduler-free]

# Dependency graph
requires:
  - phase: 01-dfl-integration
    provides: DINOXLightningModel with configure_optimizers and param groups
provides:
  - AdamWScheduleFree optimizer integration with Lightning lifecycle hooks
  - OptimizerConfig with optional lr_scheduler support
  - E6 Hydra experiment config for scheduler-free ablation
affects: [07-knowledge-distillation]

# Tech tracking
tech-stack:
  added: [schedulefree>=1.4]
  patterns: [conditional optimizer selection via use_scheduler_free flag, TypedDict inheritance for optional fields]

key-files:
  created:
    - src/object_detection_training/conf/models/dinox_m_e6.yaml
    - tests/test_dinox_scheduler_free.py
  modified:
    - src/object_detection_training/types.py
    - src/object_detection_training/models/dinox_lightning.py
    - src/object_detection_training/conf/models/dinox_base.yaml
    - pixi.toml

key-decisions:
  - "Conditional import of schedulefree inside configure_optimizers to avoid import-time dependency"
  - "TypedDict inheritance pattern (_OptimizerConfigRequired + total=False) for optional lr_scheduler"
  - "type: ignore[import-untyped] for schedulefree (no py.typed marker)"

patterns-established:
  - "Scheduler-free train/eval mode switching via on_validation_model_eval/train hooks"
  - "Checkpoint safety via on_save_checkpoint calling optimizer.eval()"

# Metrics
duration: 7min
completed: 2026-03-06
---

# Phase 6 Plan 01: Scheduler-Free Optimizer Summary

**AdamWScheduleFree integration with Lightning lifecycle hooks for train/eval mode switching and E6 ablation config**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-06T15:33:06Z
- **Completed:** 2026-03-06T15:40:27Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Integrated Facebook Research's scheduler-free AdamW as alternative to SGD+cosine in DINOXLightningModel
- Added four Lightning lifecycle hooks (on_train_epoch_start, on_validation_model_eval, on_validation_model_train, on_save_checkpoint) for optimizer train/eval mode switching
- Made OptimizerConfig.lr_scheduler optional via TypedDict inheritance, enabling optimizer-only configs
- Created E6 experiment config (dinox_m_e6.yaml) with scheduler-free enabled at LR 0.0025
- 10 unit tests covering optimizer creation, param groups, hooks, checkpoint safety, Hydra config, and SGD regression

## Task Commits

Each task was committed atomically:

1. **Task 1: Install schedulefree and implement optimizer + hooks** - `6cce870` (feat)
2. **Task 2: Unit tests for scheduler-free integration** - `0954c78` (test)

## Files Created/Modified
- `pixi.toml` - Added schedulefree>=1.4,<2.0 dependency
- `src/object_detection_training/types.py` - OptimizerConfig with optional lr_scheduler via TypedDict inheritance
- `src/object_detection_training/models/dinox_lightning.py` - AdamWScheduleFree branch in configure_optimizers, lifecycle hooks
- `src/object_detection_training/conf/models/dinox_base.yaml` - use_scheduler_free: false default
- `src/object_detection_training/conf/models/dinox_m_e6.yaml` - E6 experiment config with scheduler-free
- `tests/test_dinox_scheduler_free.py` - 10 unit tests for scheduler-free integration

## Decisions Made
- Conditional import of schedulefree inside configure_optimizers to avoid import-time dependency when not using scheduler-free
- TypedDict inheritance pattern (_OptimizerConfigRequired + total=False) for optional lr_scheduler -- cleanest way to make one field optional while keeping optimizer required
- Added type: ignore[import-untyped] for schedulefree since it has no py.typed marker

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed line-too-long in docstring**
- **Found during:** Task 1 (commit attempt)
- **Issue:** Docstring line for use_scheduler_free parameter was 89 chars (max 88)
- **Fix:** Wrapped docstring line across two lines
- **Files modified:** src/object_detection_training/models/dinox_lightning.py
- **Committed in:** 6cce870 (part of Task 1 commit, fixed before successful commit)

**2. [Rule 1 - Bug] Fixed mypy import-untyped error for schedulefree**
- **Found during:** Task 1 (typecheck verification)
- **Issue:** schedulefree package lacks py.typed marker, causing mypy error
- **Fix:** Added type: ignore[import-untyped] comment on import
- **Files modified:** src/object_detection_training/models/dinox_lightning.py
- **Committed in:** 6cce870

**3. [Rule 1 - Bug] Fixed test trainer.model property has no setter**
- **Found during:** Task 2 (test execution)
- **Issue:** Lightning Trainer.model is a read-only property; tests used trainer.model = model
- **Fix:** Used MagicMock trainer with estimated_stepping_batches and max_epochs, attached via model._trainer
- **Files modified:** tests/test_dinox_scheduler_free.py
- **Committed in:** 0954c78

---

**Total deviations:** 3 auto-fixed (3 bugs)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Scheduler-free optimizer integration complete and tested
- E6 config ready for ablation experiments
- SGD+cosine path unchanged (no regression)
- Ready for Phase 7 (knowledge distillation) or training experiments

---
*Phase: 06-scheduler-free-optimizer*
*Completed: 2026-03-06*
