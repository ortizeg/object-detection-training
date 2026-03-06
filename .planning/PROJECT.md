# DINO-X: Foundation Model Distillation with Matchability-Aware Training

## What This Is

A set of training improvements to YOLOX-M that aim to beat RF-DETR-S (53.0% COCO mAP@50:95) without changing the CSPDarknet backbone. The improvements include soft label assignment, distribution focal loss, matchability-aware loss, DINOv2 feature distillation, NMS-free dual-head training, and scheduler-free optimization — all independently toggleable for ablation studies. This builds on the existing object-detection-training codebase which already has working YOLOX, RF-DETR, and ONNX evaluation pipelines.

## Core Value

Demonstrate that training innovations alone — DINOv2 distillation, modern label assignment, and matchability-aware loss — can close the 6+ mAP gap between a 2021 YOLO detector and 2025 SOTA, producing a competitive Apache 2.0 licensed detector.

## Requirements

### Validated

- Working YOLOX training with PyTorch Lightning — existing
- Working RF-DETR training — existing
- ONNX export and evaluation pipeline — existing
- Hydra configuration system — existing
- Pydantic config validation — existing
- Ruff + MyPy strict code quality — existing
- W&B experiment tracking — existing

### Active

- [ ] Soft SimOTA label assignment with IoU-weighted soft targets and -log(IoU) regression cost
- [ ] Distribution Focal Loss (DFL) for box regression with reg_max=16
- [ ] Matchability-Aware Loss (MAL) adapted from DEIM for O2M detectors
- [ ] DINOv2-B/14 feature distillation (training-only, zero inference overhead)
- [ ] NMS-free dual-head training (O2M + O2O consistent dual assignments)
- [ ] Scheduler-free AdamW optimizer option
- [ ] Task-Aligned Assigner (TAL) as alternative for ablation comparison
- [ ] All ablation Hydra configs (A through H, E1-E4, F1-F4)
- [ ] GCP batch launcher scripts for ablation experiments
- [ ] Unit tests for all new modules
- [ ] ONNX export compatibility (DFL baked in, dual-head O2O-only export)

### Out of Scope

- Backbone architecture changes — thesis is "same architecture, better training"
- Any ultralytics/AGPL code — licensing constraint
- DINOv3 weights — custom Meta license, not Apache 2.0
- Objects365 pretraining — stretch goal only, not in v1
- CopyBlend augmentation (F4 config) — low priority tertiary ablation
- Mobile/edge deployment optimization — server-first

## Context

The existing codebase has working YOLOX and RF-DETR training with PyTorch Lightning, Hydra configs, Pydantic validation, and ONNX export. RF-DETR-S achieves 53.0% COCO mAP — the target to beat. Current YOLOX-M baseline is 46.9%. The basketball dataset is a secondary evaluation domain where DINOv2 distillation is expected to show disproportionately larger gains due to foundation model transfer.

All new code goes under `src/object_detection_training/models/dinox/` with sub-packages for losses, assigners, distillation, and heads. The DINO-X Lightning module extends/wraps the existing YOLOX Lightning module.

Key licensing constraint: all components must be Apache 2.0 or MIT. Never use ultralytics code.

GPU strategy: Phases 1-3 on NVIDIA L4 (24GB), Phases 4-6 on NVIDIA A100 (40GB+) for DINOv2 teacher memory.

## Constraints

- **Licensing**: Apache 2.0 only — no ultralytics, no AGPL code, no DINOv3
- **Architecture**: CSPDarknet backbone and PAN neck must remain unchanged
- **Compatibility**: ONNX export must produce same output format as existing YOLOX exports
- **Code patterns**: Must follow existing Pydantic + Hydra config pattern, PyTorch Lightning modules
- **GPU**: L4 for phases 1-3, A100 for phases 4-6 (DINOv2 teacher requires extra VRAM)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep CSPDarknet backbone unchanged | Cleaner research story: "same architecture, better training" | — Pending |
| MA-Soft-SimOTA over MA-TAL as primary | SimOTA is YOLOX native, TAL risks "we added YOLOv8 features" framing | — Pending (ablation E1-E4 will validate) |
| DINOv2-B/14 as teacher (not larger) | Apache 2.0, good quality/compute tradeoff, fits A100 memory budget | — Pending |
| reg_max=16 for DFL | GFL paper default, standard across all implementations | — Pending |
| Dual-head reimplemented from paper | YOLOv10 code is AGPL-tainted via ultralytics, paper algorithm is freely implementable | — Pending |

---
*Last updated: 2026-03-05 after initialization*
