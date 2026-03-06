# Requirements: DINO-X

**Defined:** 2026-03-05
**Core Value:** Beat RF-DETR-S (53.0% COCO mAP) with YOLOX-M architecture using training innovations alone, all Apache 2.0

## v1 Requirements

### Infrastructure

- [ ] **INFRA-01**: DINOXConfig Pydantic model validates all improvement flags with safe defaults (all off = standard YOLOX-M)
- [ ] **INFRA-02**: Hydra config files for all ablation variants (A through H, E1-E4, F1-F4) load and validate correctly
- [ ] **INFRA-03**: DINOXLightningModel integrates with existing training pipeline (task_manager.py, callbacks, logging)
- [ ] **INFRA-04**: EMA callback handles dynamic parameter additions (distillation projectors, dual-head) without breaking

### Soft SimOTA

- [ ] **SIMO-01**: Soft classification targets replace binary 1.0 with IoU^gamma weighted targets for positive samples
- [ ] **SIMO-02**: -log(IoU) regression cost replaces GIoU cost in SimOTA cost matrix computation
- [ ] **SIMO-03**: Soft classification cost uses RTMDet formulation: CE(P, Y_soft) * (Y_soft - P)^2
- [ ] **SIMO-04**: SimOTAConfig flags (use_soft_labels, use_log_iou_cost, soft_label_gamma) independently toggle each change

### Distribution Focal Loss

- [ ] **DFL-01**: Regression head outputs 4 * (reg_max + 1) values per anchor instead of 4 scalar offsets
- [ ] **DFL-02**: DFL loss function correctly computes cross-entropy between predicted distribution and discretized target offset
- [ ] **DFL-03**: Box decoding converts distribution to point estimate via weighted sum: offset = sum(softmax(pred) * arange(0, reg_max+1))
- [ ] **DFL-04**: Combined regression loss is L_GIoU + lambda_dfl * L_DFL with configurable weight
- [ ] **DFL-05**: ONNX export bakes distribution-to-point conversion into graph so inference code doesn't change

### Matchability-Aware Loss

- [ ] **MAL-01**: Matchability score computed as IoU^gamma * cls_score^(1-gamma) for each positive anchor
- [ ] **MAL-02**: MAL classification loss amplifies gradient for low-quality matches while behaving like standard BCE for high-quality matches
- [ ] **MAL-03**: MAL integrates on top of soft label assignment (soft IoU targets provide cls target, MAL provides loss weighting)

### Task-Aligned Assigner

- [ ] **TAL-01**: TAL implemented as separate assigner class using alignment metric m = cls_score^alpha * IoU^beta
- [ ] **TAL-02**: TAL is swappable with SimOTA via config flag for ablation comparison (E3, E4 configs)

### DINOv2 Distillation

- [ ] **DIST-01**: DINOv2-B/14 teacher loads from torch.hub (Apache 2.0), all parameters frozen
- [ ] **DIST-02**: Feature extraction at configurable ViT layer indices (default: layers 4, 8, 12) with spatial reshape
- [ ] **DIST-03**: Projector modules (1x1 Conv + BN) map student channels to teacher embedding dimension with bilinear interpolation for spatial alignment
- [ ] **DIST-04**: Total loss includes configurable distillation weight: L_total = L_detection + lambda_distill * sum(L_level)
- [ ] **DIST-05**: Teacher, projectors, and distillation loss completely removed at inference and ONNX export

### NMS-Free Dual Head

- [ ] **DUAL-01**: O2O head has identical structure to O2M head but separate parameters
- [ ] **DUAL-02**: O2O label assignment uses Hungarian matching (scipy) assigning exactly 1 prediction per GT
- [ ] **DUAL-03**: Both O2M and O2O heads use consistent alignment metric (same alpha, beta values)
- [ ] **DUAL-04**: Training loss combines both heads: L_total = L_o2m + lambda_o2o * L_o2o
- [ ] **DUAL-05**: ONNX export includes only O2O head branch, producing NMS-free inference

### Scheduler-Free Optimizer

- [ ] **OPT-01**: Scheduler-free AdamW from facebookresearch/schedule_free integrates as optimizer option
- [ ] **OPT-02**: optimizer.train()/optimizer.eval() calls correctly placed in Lightning training hooks
- [ ] **OPT-03**: OptimizerConfig supports switching between SGD and adamw_sf via config flag

### Testing

- [ ] **TEST-01**: Unit tests for soft SimOTA verify IoU-weighted targets and -log(IoU) cost values
- [ ] **TEST-02**: Unit tests for DFL verify output shape, loss gradient, and Integral conversion
- [ ] **TEST-03**: Unit tests for MAL verify equivalence to BCE at matchability=1.0 and gradient amplification at low matchability
- [ ] **TEST-04**: Unit tests for distillation verify teacher output shape, projector dimension alignment, and zero-loss when student=teacher
- [ ] **TEST-05**: Unit tests for dual-head verify Hungarian matching produces 1:1 assignment and output shape consistency
- [ ] **TEST-06**: Unit tests for all ablation configs verify they load and validate correctly

### Experiment Deployment

- [ ] **DEPLOY-01**: GCP batch launcher scripts submit ablation experiments to Vertex AI with correct GPU tier
- [ ] **DEPLOY-02**: Primary ablation script (A, B, C, D, E) targets NVIDIA L4
- [ ] **DEPLOY-03**: Distillation ablation script (F2, F3, G, H) targets NVIDIA A100
- [ ] **DEPLOY-04**: A100 trainer config (gpu_a100.yaml) sets appropriate batch size and precision for distillation phases

## v2 Requirements

### Extended Ablation

- **ABLAT-01**: CopyBlend augmentation integration for distillation (F4 config)
- **ABLAT-02**: Objects365 pretraining comparison experiment
- **ABLAT-03**: Basketball dataset transfer evaluation across all configs

### Paper

- **PAPER-01**: Full comparison table vs SOTA models
- **PAPER-02**: Ablation results analysis and visualization

## Out of Scope

| Feature | Reason |
|---------|--------|
| Backbone architecture changes | Thesis is "same architecture, better training" |
| Any ultralytics/AGPL code | Licensing constraint — Apache 2.0 only |
| DINOv3 weights | Custom Meta license, not Apache 2.0 |
| Mobile/edge deployment optimization | Server-first deployment target |
| torch.compile optimization | Dynamic shapes in SimOTA may cause issues, defer |
| Multi-GPU distributed training | Single GPU per experiment sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
| SIMO-01 | Phase 2 | Pending |
| SIMO-02 | Phase 2 | Pending |
| SIMO-03 | Phase 2 | Pending |
| SIMO-04 | Phase 2 | Pending |
| DFL-01 | Phase 3 | Pending |
| DFL-02 | Phase 3 | Pending |
| DFL-03 | Phase 3 | Pending |
| DFL-04 | Phase 3 | Pending |
| DFL-05 | Phase 3 | Pending |
| MAL-01 | Phase 4 | Pending |
| MAL-02 | Phase 4 | Pending |
| MAL-03 | Phase 4 | Pending |
| TAL-01 | Phase 4 | Pending |
| TAL-02 | Phase 4 | Pending |
| DIST-01 | Phase 5 | Pending |
| DIST-02 | Phase 5 | Pending |
| DIST-03 | Phase 5 | Pending |
| DIST-04 | Phase 5 | Pending |
| DIST-05 | Phase 5 | Pending |
| DUAL-01 | Phase 6 | Pending |
| DUAL-02 | Phase 6 | Pending |
| DUAL-03 | Phase 6 | Pending |
| DUAL-04 | Phase 6 | Pending |
| DUAL-05 | Phase 6 | Pending |
| OPT-01 | Phase 7 | Pending |
| OPT-02 | Phase 7 | Pending |
| OPT-03 | Phase 7 | Pending |
| TEST-01 | Phase 2 | Pending |
| TEST-02 | Phase 3 | Pending |
| TEST-03 | Phase 4 | Pending |
| TEST-04 | Phase 5 | Pending |
| TEST-05 | Phase 6 | Pending |
| TEST-06 | Phase 1 | Pending |
| DEPLOY-01 | Phase 7 | Pending |
| DEPLOY-02 | Phase 7 | Pending |
| DEPLOY-03 | Phase 7 | Pending |
| DEPLOY-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after initial definition*
