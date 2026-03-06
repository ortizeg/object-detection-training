# Feature Landscape

**Domain:** Object detection training improvements (DINO-X: modernizing YOLOX-M toward 53%+ COCO mAP)
**Researched:** 2026-03-05

## Table Stakes

Features that must exist or the system is non-functional / the ablation study is meaningless.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Independent feature toggles via Hydra config** | Each of the 6 improvements must be independently enable/disable-able for ablation configs A-H, E1-E4, F1-F4. Without this, no ablation study is possible. | Medium | Hydra's config group composition is ideal. Each improvement gets a boolean flag + associated hyperparams in a `dinox:` config block. |
| **Backward-compatible baseline** | Config A (baseline YOLOX-M) must reproduce the original YOLOX-M results exactly. If the baseline regresses, all ablation comparisons are invalid. | Low | Keep original `YOLOXHead` + `SimOTA` as default path. New code paths only activate when toggled on. |
| **Clean ONNX export for all configurations** | Existing eval pipeline (`ONNXExportCallback`, `scripts/export_and_eval_v2.py`) must work with every ablation config. Dual-head NMS-free must export only the one-to-one head. | High | The dual-head architecture requires careful handling: training uses both heads, but ONNX export must strip the one-to-many head and export only the NMS-free one-to-one head. Output tensor shape changes. |
| **Consistent loss logging** | All new loss terms (DFL loss, distillation loss, auxiliary head losses) must be logged to W&B/TensorBoard alongside existing losses (iou_loss, cls_loss, obj_loss, l1_loss). | Low | Extend the existing `ModelOutputs` dict returned from `YOLOX.forward()`. Lightning handles the rest. |
| **Soft SimOTA label assignment** | Replace hard one-hot assignment targets with soft targets weighted by IoU quality. The current code at line 396 of `yolo_head.py` already multiplies one-hot by `pred_ious_this_matching` -- this IS a soft target. Need to verify if this is already soft enough or if TAL-style alignment scores are needed. | Medium | Current YOLOX SimOTA already uses IoU-weighted soft targets for classification. The "soft" improvement likely means switching to a Task-Aligned Assignment (TAL) metric: `alignment_metric = cls_score^alpha * iou^beta`, which jointly considers classification and localization quality. This is what PP-YOLOE, YOLOv8, and YOLO26 use. |
| **Distribution Focal Loss (DFL) for box regression** | Replace direct `(cx, cy, w, h)` regression with distribution-based regression over discrete bins. Standard in YOLOv8+, PP-YOLOE+, and GFocal. | High | Requires architectural change to the regression head: output changes from 4 values to `4 * (reg_max + 1)` values per anchor. Need new `dfl_loss` function, new decode logic for training and inference, and updated ONNX export to handle the distribution-to-box conversion. This is the most invasive single change. |
| **EMA compatibility** | Existing EMA callback must work with all new model components. | Low | Already handled by Lightning EMA callback operating on full model state dict. No changes needed unless new modules are excluded from EMA. |
| **Reproducible training** | Seed management, deterministic operations where possible. | Low | Already exists via `utils/seed.py`. |

## Differentiators

Features that provide competitive advantage or novel contributions. Not expected in a standard YOLOX, but the whole point of DINO-X.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **DINOv2 feature distillation** | Use frozen DINOv2-B/L as teacher to distill rich self-supervised features into the YOLOX-M backbone. This is the most impactful single improvement -- DINOv2 features are exceptionally good at capturing semantic structure that YOLOX's ImageNet-pretrained CSPDarknet lacks. | High | Requires: (1) loading frozen DINOv2 model alongside YOLOX, (2) feature alignment projector (MLP or 1x1 conv) to match channel dims, (3) distillation loss (L2 or cosine similarity on intermediate features), (4) matching spatial resolutions between DINOv2 patch tokens and YOLOX FPN features. The project already has DINOv2 code in `rfdetr/backbone/dinov2.py` and `dinov2_with_windowed_attn.py` -- reuse these. Memory overhead: frozen DINOv2-B adds ~86M params to GPU memory (inference only, no gradients). |
| **Mutual Auxiliary Learning (MAL)** | Novel contribution: classification and regression branches provide auxiliary supervision to each other. Classification features generate box offsets to refine regression; regression IoU quality guides classification confidence. This addresses the well-known task misalignment problem in decoupled heads. | High | This is the novel/research contribution. No off-the-shelf implementation exists. Requires: (1) cross-branch feature routing with stop-gradient to prevent collapse, (2) auxiliary loss terms with careful weighting, (3) ablation to demonstrate it helps beyond what TAL already provides. Related work: MADet (IEEE 2023) uses similar mutual-assistance idea but different mechanism. |
| **NMS-free dual-head training** | Train with both one-to-many (SimOTA/TAL) and one-to-one (Hungarian matching) heads. At inference, use only the one-to-one head -- no NMS needed. Proven in YOLOv10 and YOLO26. | High | Requires: (1) second detection head with independent cls/reg predictions, (2) one-to-one matching (Hungarian or simpler top-1 assignment), (3) consistent matching metric between heads so one-to-one learns from one-to-many's best assignments, (4) head dropout during ONNX export. YOLO26 showed this improves both mAP and inference latency (saves 4-5ms from NMS removal). |
| **Scheduler-free AdamW** | Drop learning rate schedulers entirely. Meta's Schedule-Free AdamW won MLCommons AlgoPerf 2024. Eliminates the `total_epochs` dependency from the optimizer, simplifying training and removing one hyperparameter axis. | Low | Drop-in replacement. `pip install schedulefree`. Set `optimizer.train()` / `optimizer.eval()` at appropriate points in Lightning hooks. Learning rate should be 3-10x larger than with cosine schedule. Only complexity: need to handle the `optimizer.eval()` call before validation/checkpointing (the optimizer maintains separate eval params). |
| **GIoU/CIoU box regression loss** | Upgrade from IoU loss to CIoU loss for better convergence on boxes with no overlap. Current `IOUloss` class supports IoU and GIoU but not CIoU/DIoU. | Low | Add CIoU to the existing `IOUloss` class. Well-understood, standard improvement. Small but reliable mAP gain (~0.3-0.5%). |
| **Quality Focal Loss (QFL)** | Replace BCE classification loss with QFL, which jointly represents classification and localization quality in a single score. Part of the GFocal family alongside DFL. | Medium | QFL targets are continuous IoU values instead of {0, 1}, eliminating the need for a separate objectness branch. This naturally pairs with DFL. If implementing full GFocal (QFL + DFL), the objectness head can be removed entirely, simplifying the architecture. |

## Anti-Features

Features to explicitly NOT build. These are tempting but counterproductive.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Backbone architecture changes (e.g., replace CSPDarknet with ConvNeXt/EfficientNet)** | Scope creep. The goal is training improvement techniques, not architecture search. Changing the backbone invalidates comparisons and makes ablation results less useful. | Keep CSPDarknet-M as-is. Use DINOv2 distillation to get better features without changing the student backbone. |
| **Multi-scale training / input resolution sweeps** | YOLOX already supports multi-scale. Adding resolution as another ablation axis explodes the experiment matrix. | Fix input resolution (640x640 for COCO, match existing config for basketball). Resolution effects are orthogonal to the 6 improvements. |
| **Custom data augmentation pipelines (Mosaic/MixUp changes)** | Augmentation tuning is a separate research axis. Changing augmentations alongside training techniques confounds the ablation. | Keep existing augmentation pipeline unchanged. If augmentation matters, that is a separate milestone. |
| **Mixed-precision / quantization-aware training** | Optimization concern, not a training technique improvement. Already using AMP via Lightning. | Keep existing AMP setup. Quantization is a deployment milestone, not a training improvement milestone. |
| **Attention mechanisms in the neck (CBAM, SE, etc.)** | Architecture modification, not training technique. Changes parameter count and FLOPs, making fair comparison harder. | The PAFPN neck stays as-is. Training improvements should work regardless of neck architecture. |
| **Auto-tuning / NAS for hyperparameters** | Massive compute cost. The ablation configs A-H already define the experiment plan. | Use fixed hyperparameters from literature/preliminary experiments. Document sensitivities in ablation results. |
| **End-to-end trainable NMS replacement (e.g., learnable NMS)** | The dual-head approach is strictly better -- it needs zero NMS at inference. Learnable NMS adds complexity without the clean export story. | Use the dual-head one-to-many/one-to-one approach from YOLOv10/YOLO26. |
| **Full GFocalV2 with DGQP** | GFocalV2 adds Distribution-Guided Quality Predictor which couples the quality score to the box distribution. Adds complexity beyond what is needed for the 6 planned improvements. | Implement QFL + DFL separately. If needed, DGQP can be added later as an incremental improvement. |

## Feature Dependencies

```
Baseline YOLOX-M (Config A)
    |
    +-- Soft SimOTA / TAL Assignment (Config B)
    |       |
    |       +-- QFL (pairs naturally - soft targets + quality-aware loss)
    |
    +-- DFL Box Regression (Config C)
    |       |
    |       +-- Requires head architecture change (reg output 4 -> 4*(reg_max+1))
    |       +-- Requires new decode_outputs logic
    |       +-- Requires ONNX export update for distribution-to-box
    |
    +-- MAL Cross-Branch Learning (Config D)
    |       |
    |       +-- Depends on: decoupled head (already exists in YOLOX)
    |       +-- Benefits from: soft assignment (Config B) for quality signals
    |
    +-- DINOv2 Distillation (Configs E1-E4)
    |       |
    |       +-- Independent of other features (additive loss term)
    |       +-- Reuses: rfdetr/backbone/dinov2.py for model loading
    |       +-- E1-E4 vary: distillation layers, loss type, projector design
    |
    +-- NMS-Free Dual Head (Configs F1-F4)
    |       |
    |       +-- Benefits from: TAL assignment (Config B) for one-to-many head
    |       +-- F1-F4 vary: matching strategy, consistent metric, head architecture
    |       +-- ONNX export must strip one-to-many head
    |
    +-- Scheduler-Free AdamW (Config G)
    |       |
    |       +-- Independent of all other features
    |       +-- Requires Lightning optimizer hook changes
    |
    +-- Combined Best (Config H)
            |
            +-- All winning features from A-G combined
```

### Critical Path Dependencies

1. **DFL before NMS-Free**: If DFL changes the regression output format, the one-to-one head in dual-head training must also use DFL. Implement DFL first.
2. **Soft Assignment before MAL**: MAL uses quality signals from the assignment. Soft/TAL assignment provides richer signals than hard SimOTA.
3. **DINOv2 distillation is independent**: Can be developed in parallel with other features. Only adds a loss term, does not modify the student architecture.
4. **Scheduler-Free AdamW is independent**: Pure optimizer swap, no interaction with model architecture.
5. **ONNX export must be validated after each feature**: Every feature that changes the head or output format needs an ONNX export test.

## MVP Recommendation

Prioritize (implement first, highest impact-to-complexity ratio):

1. **Scheduler-Free AdamW** -- Drop-in, zero architecture risk, immediate benefit. Eliminates scheduler tuning.
2. **Soft SimOTA / TAL Assignment** -- Moderate complexity, proven ~1% mAP gain. The existing code is already partially soft; completing the TAL alignment metric is a natural next step.
3. **DFL Box Regression** -- High complexity but well-understood (YOLOv8, PP-YOLOE, GFocal all use it). Proven ~0.5-1% mAP gain.
4. **DINOv2 Distillation** -- High impact (potentially 2-3% mAP from rich teacher features), independent development path.

Defer (implement after validating above):

5. **NMS-Free Dual Head** -- High complexity, benefits from having TAL and DFL settled first. ONNX export changes are significant.
6. **MAL** -- Novel/research contribution, highest risk. Should be attempted after the proven techniques are validated so there is a strong baseline to compare against.

## Ablation Config Matrix

| Config | Features Enabled | Purpose |
|--------|-----------------|---------|
| A | Baseline YOLOX-M | Control |
| B | A + Soft TAL Assignment | Measure assignment improvement |
| C | A + DFL | Measure regression improvement |
| D | A + MAL | Measure cross-branch learning |
| E1-E4 | A + DINOv2 Distillation (variants) | Measure distillation impact + design choices |
| F1-F4 | A + NMS-Free Dual Head (variants) | Measure NMS-free impact + design choices |
| G | A + Scheduler-Free AdamW | Measure optimizer impact |
| H | Best of B-G combined | Final model |

**Note on config design**: Each config should add exactly ONE improvement over baseline A, except H which combines winners. This is critical for clean ablation -- if Config C adds DFL AND changes the optimizer, you cannot attribute mAP changes to either one.

## Sources

- [Generalized Focal Loss (QFL + DFL) -- NeurIPS 2020](https://arxiv.org/abs/2006.04388)
- [GFocal implementation -- GitHub](https://github.com/implus/GFocal)
- [DFL loss explained -- YOLOv8 docs](https://yolov8.org/what-is-dfl-loss-in-yolov8/)
- [YOLO Loss Functions: GFL and VFL -- LearnOpenCV](https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/)
- [Label Assignment in YOLO family -- Medium](https://medium.com/@nazari-ehsan/label-assignment-the-hidden-engine-behind-yolos-learning-f6b487d90535)
- [YOLOv10 NMS-Free Dual Assignments](https://arxiv.org/html/2405.14458v1)
- [YOLO26 NMS-Free Architecture](https://learnopencv.com/yolo26-nms-free-inference/)
- [YOLO26 analysis paper](https://arxiv.org/html/2601.12882v1)
- [Schedule-Free AdamW -- Meta Research](https://github.com/facebookresearch/schedule_free)
- [Schedule-Free paper -- "The Road Less Scheduled"](https://arxiv.org/abs/2405.15682)
- [DINOv2 -- Meta AI](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
- [DINOv2 feature distillation for tracking](https://arxiv.org/html/2407.18288v2)
- [MADet: Mutual-Assistance Learning for Detection -- IEEE 2023](https://ieeexplore.ieee.org/abstract/document/10265160)
- [Lightly-Train: DINOv2/v3 distillation framework](https://github.com/lightly-ai/lightly-train)
- [YOLOX official repository](https://github.com/Megvii-BaseDetection/YOLOX)
