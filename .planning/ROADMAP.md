# Roadmap: DINO-X

## Overview

Transform YOLOX-M from 46.9% to 53.0%+ COCO mAP through six training innovations -- soft label assignment, Distribution Focal Loss, Matchability-Aware Loss, DINOv2 distillation, NMS-free dual head, and scheduler-free optimization -- all independently toggleable for ablation studies. The roadmap builds from the most invasive architectural change (DFL regression format) outward through progressively independent improvements, ending with the full ablation experiment deployment.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: DFL Foundation and Infrastructure** - New detection head with DFL regression, config system, Lightning integration
- [ ] **Phase 2: Soft Label Assignment** - Soft SimOTA with IoU-weighted targets and pluggable assigner protocol
- [ ] **Phase 3: Loss and Assignment Improvements** - MAL cross-branch learning and TAL alternative assigner
- [ ] **Phase 4: NMS-Free Dual Head** - O2O branch with consistent matching for NMS-free inference
- [ ] **Phase 5: DINOv2 Feature Distillation** - Frozen DINOv2-B/14 teacher with feature projection and A100 config
- [ ] **Phase 6: Scheduler-Free Optimizer** - Schedule-free AdamW integration with Lightning hooks
- [ ] **Phase 7: Ablation Deployment** - GCP launcher scripts and full experiment matrix execution

## Phase Details

### Phase 1: DFL Foundation and Infrastructure
**Goal**: A new DINOXHead with DFL box regression trains end-to-end through the existing pipeline, exports to ONNX with the same output contract, and all improvement flags are off by default reproducing standard YOLOX-M behavior
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, DFL-01, DFL-02, DFL-03, DFL-04, DFL-05, TEST-02, TEST-06
**Success Criteria** (what must be TRUE):
  1. Running training with all DINOXConfig flags set to default (off) reproduces standard YOLOX-M loss curves within noise
  2. DFL-enabled training produces a model whose ONNX export has identical output shape [B, N, 5+C] to existing YOLOX exports
  3. All ablation Hydra configs (A through H, E1-E4, F1-F4) load and validate without errors
  4. Unit tests for DFL verify output shape (4 * (reg_max+1) per anchor), loss gradient flow, and Integral decode correctness
  5. EMA callback correctly handles the new DINOXHead parameters without state dict mismatches
**Plans:** 3 plans

Plans:
- [ ] 01-01-PLAN.md -- Core DINOX modules: DINOXConfig, DFLModule, DINOXHead, DINOX wrapper
- [ ] 01-02-PLAN.md -- Lightning integration: DINOXLightningModel, Hydra configs, model registration
- [ ] 01-03-PLAN.md -- Tests: DFL unit tests, config validation, Hydra parameterization, ONNX round-trip

### Phase 2: Soft Label Assignment
**Goal**: Soft SimOTA replaces binary label assignment with IoU-weighted soft targets, providing richer training signal and establishing the pluggable assigner protocol used by all subsequent phases
**Depends on**: Phase 1
**Requirements**: SIMO-01, SIMO-02, SIMO-03, SIMO-04, TEST-01
**Success Criteria** (what must be TRUE):
  1. Soft SimOTA assigns IoU^gamma weighted targets instead of binary 1.0 for positive samples, verified by unit test
  2. Each soft SimOTA toggle (use_soft_labels, use_log_iou_cost, soft_label_gamma) independently enables/disables its respective change without affecting the others
  3. Training with soft SimOTA enabled converges without zero-positive-assignment failures (num_fg > 0 every batch after warmup)
  4. Unit tests verify IoU-weighted target values and -log(IoU) cost matrix computation against hand-computed examples
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Loss and Assignment Improvements
**Goal**: MAL amplifies gradient signal for low-quality matches and TAL provides an alternative assigner for ablation comparison, completing the label assignment and loss toolkit
**Depends on**: Phase 2
**Requirements**: MAL-01, MAL-02, MAL-03, TAL-01, TAL-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. MAL loss is equivalent to standard BCE when matchability score equals 1.0, and produces amplified gradients when matchability is low, both verified by unit test
  2. TAL is swappable with SimOTA via a single config flag change, and ablation configs E3/E4 use TAL correctly
  3. MAL integrates with soft SimOTA targets (soft IoU targets provide cls target, MAL provides loss weighting) without circular gradient dependency
  4. Unit tests for MAL verify gradient amplification behavior and BCE equivalence at boundary conditions
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: NMS-Free Dual Head
**Goal**: A second O2O detection head trains alongside the O2M head with consistent matching, and ONNX export produces NMS-free inference using only the O2O branch
**Depends on**: Phase 1, Phase 2
**Requirements**: DUAL-01, DUAL-02, DUAL-03, DUAL-04, DUAL-05, TEST-05
**Success Criteria** (what must be TRUE):
  1. O2O head assigns exactly 1 prediction per ground truth via Hungarian matching, verified by unit test
  2. Both O2M and O2O heads use identical alignment metric values (same alpha, beta), verified by config validation
  3. ONNX export includes only the O2O branch and produces output with constant-1 objectness for backward compatibility with existing evaluation code
  4. Unit tests verify Hungarian matching produces strict 1:1 assignment and O2O output shape matches O2M output shape
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: DINOv2 Feature Distillation
**Goal**: A frozen DINOv2-B/14 teacher provides feature-level supervision during training with zero inference overhead, and A100 trainer config supports the increased memory requirements
**Depends on**: Phase 1
**Requirements**: DIST-01, DIST-02, DIST-03, DIST-04, DIST-05, TEST-04, DEPLOY-04
**Success Criteria** (what must be TRUE):
  1. DINOv2 teacher loads from torch.hub with all parameters frozen, and is completely excluded from ONNX export and inference forward pass
  2. Projector modules correctly align student feature spatial dimensions and channel counts to teacher embedding dimension via bilinear interpolation
  3. Distillation loss is zero when student features exactly match teacher features (verified by unit test with identity projector)
  4. A100 trainer config (gpu_a100.yaml) sets appropriate batch size and precision for distillation training without OOM
  5. Unit tests verify teacher output shape, projector dimension alignment, and zero-loss identity condition
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Scheduler-Free Optimizer
**Goal**: Scheduler-free AdamW is available as an optimizer option that eliminates learning rate schedule tuning while integrating correctly with Lightning training hooks
**Depends on**: Phase 1
**Requirements**: OPT-01, OPT-02, OPT-03
**Success Criteria** (what must be TRUE):
  1. Scheduler-free AdamW from facebookresearch/schedule_free is selectable via OptimizerConfig flag alongside existing SGD
  2. optimizer.train() and optimizer.eval() calls are correctly placed in Lightning on_train_epoch_start and on_validation_epoch_start hooks
  3. Training with scheduler-free AdamW converges on a short smoke test (few epochs, small subset) without NaN losses or crashed gradients
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

### Phase 7: Ablation Deployment
**Goal**: GCP batch launcher scripts submit the full ablation experiment matrix to Vertex AI with correct GPU tier assignments per experiment
**Depends on**: Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03
**Success Criteria** (what must be TRUE):
  1. GCP batch launcher script submits ablation experiments A-E to Vertex AI targeting NVIDIA L4 instances
  2. GCP batch launcher script submits distillation experiments F2, F3, G, H to Vertex AI targeting NVIDIA A100 instances
  3. Launcher scripts produce valid Vertex AI job configurations that pass dry-run validation
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
Note: Phases 4, 5, and 6 depend only on Phase 1 (and Phase 4 also on Phase 2), so they could execute in parallel after their dependencies are met.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. DFL Foundation and Infrastructure | 0/3 | Not started | - |
| 2. Soft Label Assignment | 0/2 | Not started | - |
| 3. Loss and Assignment Improvements | 0/2 | Not started | - |
| 4. NMS-Free Dual Head | 0/2 | Not started | - |
| 5. DINOv2 Feature Distillation | 0/2 | Not started | - |
| 6. Scheduler-Free Optimizer | 0/1 | Not started | - |
| 7. Ablation Deployment | 0/1 | Not started | - |
