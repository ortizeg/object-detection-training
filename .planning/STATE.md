# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Beat RF-DETR-S (53.0% COCO mAP) with YOLOX-M architecture using training innovations alone, all Apache 2.0
**Current focus:** Phase 5 - Knowledge Distillation

## Current Position

Phase: 5 of 7 (Knowledge Distillation)
Plan: 0 of N in current phase
Status: Phase 4 Complete
Last activity: 2026-03-06 -- Completed 04-02-PLAN.md (Dual Head Tests + E5/E6 Ablation Configs)

Progress: [██████████░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 4.8min
- Total execution time: 0.72 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 16min | 5.3min |
| 02 | 2 | 6min | 3min |
| 03 | 2 | 11min | 5.5min |
| 04 | 2 | 13min | 6.5min |

**Recent Trend:**
- Last 5 plans: 3min, 6min, 5min, 6min, 7min
- Trend: stable

*Updated after each plan completion*
| Phase 01 P02 | 4min | 2 tasks | 6 files |
| Phase 01 P03 | 5min | 2 tasks | 6 files |
| Phase 02 P01 | 3min | 2 tasks | 4 files |
| Phase 02 P02 | 3min | 1 tasks | 1 files |
| Phase 03 P01 | 6min | 2 tasks | 6 files |
| Phase 03 P02 | 5min | 2 tasks | 6 files |
| Phase 04 P01 | 6min | 2 tasks | 6 files |
| Phase 04 P02 | 7min | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- DFL lands in Phase 1 because it changes the fundamental regression output format that all subsequent phases depend on
- Tests co-locate with features (TEST-02 in Phase 1 with DFL, TEST-01 in Phase 2 with SimOTA, etc.)
- DEPLOY-04 (A100 config) co-locates with Phase 5 (distillation) since that is the phase requiring A100 memory
- DINOXHead is fresh nn.Module (not YOLOXHead subclass) for decoupling from third-party code
- SimOTA assignment self-contained in DINOXHead; forward uses targets-not-None branching for ONNX compat
- [Phase 01]: No task_manager.py changes needed -- existing import chain auto-discovers registered DINO-X models
- [Phase 01]: Reuse YOLOX_CHECKPOINT_URLS and download_checkpoint from yolox_lightning for DINO-X weight loading
- [Phase 01]: Phase 1 tests cover dinox_m_baseline and dinox_m_dfl only; ablation configs A-H deferred to later phases
- [Phase 02]: Both -log(IoU) branches identical by design for ablation explicitness and future GIoU cost alternative
- [Phase 02]: Non-in-place .sigmoid() in RTMDet path; autocast(enabled=False) per research pitfalls
- [Phase 02]: Duplicated test helpers instead of cross-test imports (tests/__init__.py prevents module-level imports)
- [Phase 03]: Duplicated _bboxes_iou in tal.py to avoid circular import (dinox_head imports tal, tal cannot import dinox_head)
- [Phase 03]: Used bounded MAL weight (1-m)^2+1.0 in [1.0, 2.0] for numerical stability
- [Phase 03]: Added soft label, MAL, and assigner defaults to dinox_base.yaml for Hydra schema completeness
- [Phase 04]: Duplicated spatial filtering helpers in hungarian.py (same pattern as tal.py) to avoid circular imports
- [Phase 04]: O2O has no objectness prediction/loss -- constant-1 objectness at inference per research
- [Phase 04]: O2O prediction layers initialized from O2M weights for warm-start convergence
- [Phase 04]: O2O loss normalized by its own num_fg independently from O2M normalization
- [Phase 04]: Used width=0.25 and depth=0.33 for fast dual head test execution
- [Phase 04]: 400-anchor grid at stride=8 for Hungarian test coverage of GT positions

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-06
Stopped at: Completed 04-02-PLAN.md (Dual Head Tests + E5/E6 Configs) -- Phase 4 complete
Resume file: None
