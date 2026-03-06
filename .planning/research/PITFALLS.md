# Pitfalls Research

**Domain:** Object detection training improvements (DFL, soft SimOTA, MAL, DINOv2 distillation, NMS-free dual-head) applied to YOLOX-M
**Researched:** 2026-03-05
**Confidence:** HIGH (verified against codebase, papers, community reports)

## Critical Pitfalls

### Pitfall 1: DFL Output Shape Breaks Pretrained Weight Loading and ONNX Export

**What goes wrong:**
DFL changes the regression head output from 4 channels to `4 * (reg_max + 1)` channels (e.g., 4 * 17 = 68 for reg_max=16). This breaks two things simultaneously: (1) pretrained YOLOX-M `reg_preds` weights cannot load because of shape mismatch, and (2) the ONNX export output tensor shape changes from `[batch, num_anchors, 5 + num_classes]` to `[batch, num_anchors, 4*(reg_max+1) + 1 + num_classes]`, which breaks all downstream inference code.

**Why it happens:**
The current `reg_preds` layers are `nn.Conv2d(in, 4, 1, 1, 0)`. DFL requires `nn.Conv2d(in, 4 * (reg_max + 1), 1, 1, 0)`. Developers often implement the DFL head correctly but forget to update: (a) the weight loading logic in `_load_weights()` to skip `reg_preds` like it already skips `cls_preds` on class mismatch, (b) the `get_output_and_grid()` method which hardcodes `n_ch = 5 + self.num_classes`, (c) the `decode_outputs()` method which assumes raw `[cx, cy, w, h]` format, and (d) the `export_onnx()` method and `get_predictions()` which assume a specific output layout.

**How to avoid:**
- Add `reg_preds` to the weight-mismatch skip list in `_load_weights()` (line 342-344 pattern in yolox_lightning.py)
- Implement the DFL softmax-integral decode as a separate method that converts `4*(reg_max+1)` back to 4 coordinates *before* concatenation with obj/cls outputs
- Keep the external output contract identical: `[batch, num_anchors, 5 + num_classes]` in pixel cxcywh. The DFL decode should be internal to the head
- Write an ONNX export round-trip test that verifies output shape matches expectations *before* any other feature is added

**Warning signs:**
- `RuntimeError` on `load_state_dict` with shape mismatch on `reg_preds` keys
- ONNX model output shape differs from expected `[1, 8400, 5+C]`
- Inference scores all near zero (DFL logits interpreted as raw coordinates)

**Phase to address:**
DFL implementation phase (should be first improvement implemented, as it changes the fundamental output structure)

---

### Pitfall 2: DFL Softmax-Integral Produces Wrong Coordinates Due to Scale/Offset Errors

**What goes wrong:**
DFL predicts a probability distribution over `reg_max+1` discrete bins for each of the 4 LTRB (left, top, right, bottom) offsets from the anchor center. The integral is computed as `sum(softmax(logits) * range(0, reg_max+1))`. Common bugs: (1) forgetting to convert YOLOX's native cxcywh format to LTRB before computing DFL targets, (2) not scaling the integral result by stride before converting back to pixel coordinates, (3) using the wrong dimension for softmax (must be over the `reg_max+1` dimension, not the spatial or 4-coordinate dimension).

**Why it happens:**
YOLOX natively uses cxcywh box format with `exp()` decoding (lines 269-270 in yolo_head.py: `output[..., 2:4] = torch.exp(output[..., 2:4]) * stride`). DFL requires LTRB (distance from anchor to each box edge). The format conversion is non-trivial and stride-dependent. The tensor reshape from `[B, 4*(reg_max+1), H, W]` to `[B, H*W, 4, reg_max+1]` is error-prone.

**How to avoid:**
- Implement and unit-test a standalone `dfl_decode(logits, reg_max, stride, grid)` function that: reshapes `[B, 4*17, H, W]` to `[B, H*W, 4, 17]`, applies `F.softmax(dim=-1)`, computes `(softmax @ arange(0, reg_max+1).float())` to get `[B, H*W, 4]` LTRB values, multiplies by stride, then converts LTRB + grid_center to xyxy or cxcywh
- Write a test that creates known boxes, encodes them as DFL targets, decodes them, and verifies reconstruction error < 0.5 pixels
- Keep DFL reg_max=16 (17 bins), which is the established default from GFL/YOLOv8

**Warning signs:**
- Bounding boxes are tiny or enormous after DFL decode
- All boxes cluster at anchor centers (softmax integral producing near-zero offsets)
- AP drops dramatically compared to baseline despite loss converging

**Phase to address:**
DFL implementation phase

---

### Pitfall 3: DINOv2 Teacher Doubles GPU Memory, Crashes Training at Current Batch Sizes

**What goes wrong:**
DINOv2 ViT-B has 86M parameters. Even frozen (no gradients), it must store activations for the forward pass and its output feature maps. With YOLOX-M already using significant GPU memory, adding a frozen ViT-B teacher can push a 24GB GPU past its limit at typical batch sizes (16-32 on 640x640 images). Training crashes with OOM at the exact batch size that worked without the teacher.

**Why it happens:**
Developers calculate only parameter memory (86M * 4 bytes = ~344MB) and conclude it is fine. But the actual overhead includes: (1) ViT intermediate activations during forward pass (~2-4GB depending on image size and patch size), (2) the feature alignment projection layers and their gradients, (3) the distillation loss computation which requires both teacher and student features simultaneously in memory. DINOv2 ViT-B with 14x14 patches on a 640x640 image produces 2048 patch tokens, each 768-dim -- that is a substantial feature map per image.

**How to avoid:**
- Use `torch.no_grad()` context for teacher forward pass (saves activation memory for backprop, but forward activations still exist)
- Use `torch.cuda.amp.autocast()` for the teacher to run in float16
- Implement teacher feature caching: run teacher once per epoch on the dataset and cache features to disk, then load per-batch during training. This trades disk I/O for GPU memory
- Start with half the batch size when enabling distillation, then increase if memory allows
- Consider DINOv2 ViT-S (21M params) instead of ViT-B if memory is tight

**Warning signs:**
- `CUDA out of memory` errors appearing only when distillation is enabled
- Training speed drops > 50% (GPU memory pressure causing swap/fragmentation)
- Validation taking much longer than expected (teacher forward pass during val)

**Phase to address:**
DINOv2 distillation phase (should be one of the later phases, after head modifications are stable)

---

### Pitfall 4: Soft SimOTA Assigns Zero Positives in Early Training, Causing NaN Losses

**What goes wrong:**
Soft SimOTA (like the existing SimOTA in the codebase) uses the model's current predictions to dynamically determine how many anchors to assign to each GT. In early training or after architectural changes (new DFL head, new cls head), predictions are essentially random. The dynamic-k calculation (`topk_ious.sum(1).int()`) can produce k=0 for some GTs when all IoUs are near zero, leading to division by zero in loss normalization (`num_fg = max(num_fg, 1)` saves the division but the model gets no positive supervision for those GTs).

**Why it happens:**
The current code (line 639: `dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)`) clamps to min=1, but with soft labels the cost matrix can become degenerate when both classification and regression predictions are poor. The `sqrt_()` operation on classification predictions (line 514) can amplify instability when values are near zero. Additionally, changing from hard one-hot cls targets to soft targets (IoU-weighted) means the classification cost computation changes, altering the cost matrix landscape.

**How to avoid:**
- Keep the existing SimOTA as the baseline assignment and implement soft label targets as a *post-processing* step on the assignment results, not a change to the assignment algorithm itself
- Add a warmup period (5-10 epochs) where the soft label weight linearly ramps from 0.0 to 1.0
- Monitor `num_fg` per batch -- if it drops below `num_gt * 0.5` consistently, the assignment is failing
- Add gradient clipping (max_norm=10.0) during the first few epochs when assignment is noisy

**Warning signs:**
- `num_fg` logged value dropping to near-zero or staying at exactly 1.0
- Loss spikes or NaN values in the first few epochs
- mAP stays at 0.0 for many epochs before slowly climbing

**Phase to address:**
Soft label assignment phase (should come after DFL is stable, because DFL changes bbox quality which affects SimOTA cost matrix)

---

### Pitfall 5: Dual-Head O2O Branch Produces Weak Supervision Without Consistent Matching Metric

**What goes wrong:**
The naive implementation adds a second YOLOXHead for one-to-one (O2O) matching using Hungarian matching or top-1 selection. But without a *consistent matching metric* between the O2M and O2O heads, the two heads can disagree on what constitutes a good prediction. The O2O head converges slowly, and at inference time (when only the O2O head is used), accuracy is significantly lower than the O2M+NMS baseline.

**Why it happens:**
YOLOv10's key insight is that the matching metric `m = s * p^alpha * IoU^beta` must use consistent hyperparameters between O2M and O2O assignments. If the O2O head uses different cost weighting (e.g., Hungarian matching on IoU alone) while O2M uses SimOTA (which weighs classification and IoU differently), the backbone learns features optimized for O2M's metric, which the O2O head cannot leverage effectively. The performance gap is worse on smaller models (YOLOX-M is mid-size) because features are less discriminative.

**How to avoid:**
- Use the same matching cost formula for both heads: `cost = cls_cost + lambda * iou_cost` with identical weights
- For O2O, use top-1 selection (pick the lowest-cost anchor per GT) rather than full Hungarian matching -- it is simpler, faster, and YOLOv10 showed equivalent performance
- Share the backbone and FPN between heads but give O2O its own cls/reg prediction layers
- Train O2M and O2O losses with equal weight initially, then consider reducing O2O weight if it destabilizes training

**Warning signs:**
- Large gap (> 3 AP) between O2M+NMS evaluation and O2O-only evaluation during training
- O2O head `num_fg` is consistently much lower than O2M head
- Inference latency does not improve (NMS was not the bottleneck)

**Phase to address:**
NMS-free dual-head phase (should be the last improvement, after all other head changes are finalized)

---

### Pitfall 6: EMA Callback State Dict Mismatch After Adding New Head Parameters

**What goes wrong:**
The existing `EMACallback` does a `copy.deepcopy(pl_module.state_dict())` at `on_fit_start`. When new modules are added (DFL projection, O2O head, distillation alignment layers), the EMA state dict gets out of sync with the model state dict if: (1) modules are added dynamically during training (e.g., enabling distillation mid-training), (2) checkpoint resumption loads an old EMA state dict that lacks new keys, or (3) the O2O head is added but EMA should only track the O2M head (since O2O is discarded at inference).

**Why it happens:**
The EMA callback iterates over `self.ema_state_dict` keys (line 67) and assumes they match `model_state.keys()` exactly. Adding new parameters creates keys in the model that have no EMA counterpart. The `load_state_dict` on validation start (line 85) will fail with `strict=True` semantics (which `load_state_dict` defaults to).

**How to avoid:**
- After adding any new module, verify the EMA callback can round-trip: `on_fit_start` -> `on_train_batch_end` -> `on_validation_start` -> `on_validation_end`
- Exclude O2O head parameters from EMA tracking (they are not used at inference)
- Exclude frozen teacher parameters from EMA tracking (they do not change)
- Add a key-matching step in the EMA update that handles missing/extra keys gracefully
- Write an integration test: initialize model with all features enabled, run 2 train steps + 1 val step, verify no crashes

**Warning signs:**
- `RuntimeError: Error(s) in loading state_dict` during validation
- Validation metrics are wrong (EMA applied partial state, some params are random)
- Checkpointing fails because EMA state dict has different keys than model

**Phase to address:**
Infrastructure/scaffolding phase (must be addressed before any feature adds new parameters)

---

### Pitfall 7: Feature Distillation Loss Dominates or Vanishes Due to Scale Mismatch

**What goes wrong:**
The distillation loss (L2 or cosine distance between DINOv2 teacher features and student FPN features) operates in a completely different value range than the detection losses (cls, reg, obj). Without careful balancing, the distillation loss either overwhelms detection losses (student learns to mimic teacher but forgets to detect) or has negligible gradient contribution (wasted compute for no benefit).

**Why it happens:**
DINOv2 features are L2-normalized to unit norm (768-dim vectors). Student FPN features have arbitrary magnitude depending on initialization and training stage. Raw L2 distance between these produces values in a very different range than BCE classification loss or IoU regression loss. Additionally, the feature dimensions differ (DINOv2: 768, YOLOX-M FPN: 192/384/768 depending on level), requiring a projection layer whose initialization also affects scale.

**How to avoid:**
- Use cosine similarity loss (scale-invariant) rather than L2 loss for feature alignment
- Add a learnable 1x1 conv projection from student feature dim to teacher feature dim, initialized with Kaiming/He initialization
- Start distillation loss weight at 0.0 and linearly ramp to target weight over 10-20 epochs
- Log the magnitude of each loss component separately and verify they are within 1-2 orders of magnitude of each other
- Use a stop-gradient on the teacher side (already frozen, but worth being explicit)

**Warning signs:**
- Detection losses (cls, reg, obj) stop decreasing while distillation loss decreases rapidly
- Or: distillation loss stays flat while detection losses decrease normally (distillation having no effect)
- mAP is lower with distillation than without (teacher is hurting, not helping)

**Phase to address:**
DINOv2 distillation phase

---

### Pitfall 8: ONNX Export Breaks When DFL Softmax or Dual-Head Branching Is Not Export-Aware

**What goes wrong:**
The current ONNX export (yolox_lightning.py line 667-726) sets `_export_mode = True` and traces through the model. DFL adds `F.softmax()` and a matrix multiply for the integral decode. The dual-head adds conditional branching (use O2O head at inference, O2M only during training). Both introduce operations that can fail during ONNX tracing: (1) `F.softmax` with a dynamic `reg_max` dimension, (2) conditional branching on `self.training` that ONNX tracer cannot follow, (3) the `arange` tensor used in integral decode must be a buffer registered on the module (not created dynamically) for ONNX export.

**Why it happens:**
ONNX tracing follows a single code path. `if self.training` branches are fine because export always runs in eval mode. But if the O2O vs O2M head selection uses any Python-level branching beyond `self.training`, the tracer will only capture one path. Dynamic tensor creation (`torch.arange(reg_max+1)`) inside forward produces a constant in the ONNX graph, which is correct but only if `reg_max` is truly fixed.

**How to avoid:**
- Register the DFL integral weights as a `nn.Buffer`: `self.register_buffer('dfl_proj', torch.arange(reg_max + 1, dtype=torch.float))`
- Ensure the inference path (eval mode) has zero Python-level branching -- the O2O head forward should be a clean sequential operation
- Add an ONNX export test to CI that: exports, loads with onnxruntime, runs a dummy input, and verifies output shape and value range
- Test ONNX simplification (`onnxsim.simplify`) with the DFL softmax -- some simplifier versions mishandle softmax+matmul patterns

**Warning signs:**
- `torch.onnx.export` raises `TracerWarning` about data-dependent control flow
- Exported ONNX model produces different outputs than PyTorch model on same input
- `onnxsim.simplify` fails or produces incorrect simplified model

**Phase to address:**
Every phase that modifies the head architecture must include an ONNX export verification step

---

### Pitfall 9: MAL Task Alignment Creates Circular Dependencies in Loss Computation

**What goes wrong:**
Mutual Alignment Learning (MAL) uses classification scores to weight regression targets and regression quality (IoU) to weight classification targets. This creates a circular dependency: better classification helps regression, which helps classification. During early training when both are poor, this mutual dependency can amplify noise rather than signal, leading to training instability or convergence to a degenerate solution where the model confidently predicts wrong boxes.

**Why it happens:**
In the existing YOLOX code (line 396), classification targets are already IoU-weighted: `cls_target = F.one_hot(...) * pred_ious_this_matching.unsqueeze(-1)`. MAL extends this by also using classification confidence to weight regression. When both tasks are noisy (early training), the product of two noisy signals is noisier than either alone. Without detaching gradients appropriately, the mutual supervision can create unstable gradient feedback loops.

**How to avoid:**
- Detach (`tensor.detach()`) the cross-task signals: when using cls score to weight reg loss, detach the cls score; when using IoU to weight cls loss, detach the IoU
- Implement MAL as a gradual enhancement: start with standard YOLOX losses for 50% of training, then enable MAL alignment
- Use EMA of alignment weights rather than instantaneous values to smooth out noise
- Test with and without MAL to verify it actually improves AP on your dataset (it may not help on small datasets with few classes)

**Warning signs:**
- Training loss oscillates rather than decreasing smoothly
- Classification and regression losses move in opposite directions (one decreases, other increases)
- Confidence calibration is poor: high-confidence predictions have low IoU

**Phase to address:**
MAL phase (should come after soft SimOTA is stable, as both affect the loss weighting)

---

### Pitfall 10: Toggleable Features Create Combinatorial Testing Burden

**What goes wrong:**
With 6 independently toggleable features (DFL, soft SimOTA, MAL, DINOv2 distillation, dual-head NMS-free, plus existing L1 loss toggle), there are 2^6 = 64 possible configurations. Some combinations are incompatible or interact badly: (a) DFL + dual-head requires both heads to use DFL decode, (b) MAL + soft SimOTA both modify the label assignment, (c) distillation + dual-head means the teacher supervises which head? Testing all combinations is impractical, and untested combinations will be used by mistake.

**Why it happens:**
Each feature is developed and tested in isolation. The config flag approach (Hydra YAML toggles) makes it easy to enable combinations that were never validated together. Integration testing is expensive (each config requires a full training run to verify).

**How to avoid:**
- Define 3-4 "blessed" configurations: baseline (no improvements), recommended (all improvements), and 1-2 intermediate configs
- Document incompatible combinations in config comments and validate them in `__init__` with explicit `ValueError` messages
- Add a "smoke test" that runs 10 training steps for each blessed configuration in CI
- Order features so that each one builds on the previous: DFL -> soft SimOTA -> MAL -> distillation -> dual-head. Do not support arbitrary combinations of later features without earlier ones

**Warning signs:**
- Users report bugs that cannot be reproduced (they were using an untested combination)
- CI passes but production training fails (CI only tests default config)
- "It worked yesterday" -- someone changed a config flag

**Phase to address:**
Infrastructure/scaffolding phase (define the toggle system and blessed configs upfront), with validation added in each feature phase

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoding `reg_max=16` everywhere | Simpler implementation | Cannot experiment with different bin counts | Acceptable permanently -- 16 is the community standard |
| Skipping ONNX export test in CI | Faster CI | Export breaks silently, discovered only at deployment | Never -- export is a hard requirement |
| Using L2 loss instead of cosine for distillation | Simpler code | Scale sensitivity requires careful loss weight tuning | Only during prototyping, switch to cosine before merging |
| Not detaching cross-task signals in MAL | Simpler backward graph | Potential gradient instability, hard to debug | Never -- always detach |
| Testing only the "all features on" config | Minimal test matrix | Regressions in subset configs go undetected | Acceptable for MVP if blessed configs are documented |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| DINOv2 Hub Model | Calling `torch.hub.load('facebookresearch/dinov2', ...)` at every init, hitting network | Cache the model weights locally; load once and reuse. Use `torch.hub.set_dir()` |
| DINOv2 Input Preprocessing | Using YOLOX preprocessing (BGR, unnormalized) for DINOv2 input | DINOv2 expects RGB, ImageNet-normalized input. Must apply separate normalization before teacher forward |
| EMA + New Parameters | EMA callback crashes when model has new keys not in EMA state dict | Add key-matching logic; skip keys not in EMA dict during update |
| Hydra Config + New Toggles | Adding toggle flags without default values breaks existing configs | Always set `enable_dfl: false` (etc.) as defaults in base config |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| DINOv2 teacher forward every batch | Training 2-3x slower than baseline | Cache teacher features per epoch or use smaller ViT-S | Immediately noticeable on first training run |
| SimOTA cost matrix on CPU | Training bottlenecked by assignment | Ensure all cost matrix ops stay on GPU; profile with `torch.cuda.Event` | Visible when num_gt > 20 per image |
| DFL softmax over large reg_max | Marginal slowdown per forward pass but adds up | Keep reg_max=16; avoid reg_max=32+ unless justified by dataset | Noticeable at reg_max > 24 |
| Dual-head doubles head computation | Training 30-40% slower | O2O head can be lighter (fewer conv layers) | Immediately on enabling dual-head |

## "Looks Done But Isn't" Checklist

- [ ] **DFL decode:** Verify decoded boxes match original boxes within 0.5px on synthetic data -- softmax integral is subtle
- [ ] **ONNX export:** Run inference with onnxruntime and compare outputs to PyTorch model (atol=1e-4) -- export can silently produce wrong results
- [ ] **Pretrained weight loading:** Verify that new modules (DFL head, O2O head, projection layers) are properly initialized even when pretrained weights are loaded for old modules
- [ ] **Distillation input normalization:** DINOv2 expects ImageNet-normalized RGB; YOLOX uses unnormalized BGR. Both paths must coexist
- [ ] **O2O head at inference:** Verify the O2M head is truly not executed during inference (not just ignored) -- unnecessary compute wastes latency
- [ ] **Loss logging:** All new loss components (DFL loss, distillation loss, MAL alignment loss) are logged separately to wandb for debugging
- [ ] **Config validation:** Enabling a feature without its dependencies raises a clear error, not a silent wrong result

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| DFL output shape breaks export | LOW | Fix decode to produce standard shape; re-export |
| DINOv2 OOM at training | LOW | Reduce batch size or switch to ViT-S; no code changes needed |
| Soft SimOTA produces NaN | MEDIUM | Add gradient clipping, warmup, and num_fg monitoring; may need to retrain from scratch |
| EMA state dict mismatch | LOW | Update EMA callback key-matching logic; resume from last valid checkpoint |
| MAL circular dependency instability | HIGH | Remove MAL, retrain, then re-add with proper detach(); full retraining needed |
| Combinatorial config failure | MEDIUM | Document and enforce blessed configs; add smoke tests; fix specific combo |
| ONNX export silently wrong | HIGH | Add numerical comparison test; if deployed wrong model, must re-deploy |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| DFL output shape (P1) | DFL phase | ONNX export test passes; pretrained weights load without error |
| DFL coordinate errors (P2) | DFL phase | Synthetic box encode-decode round-trip test passes within 0.5px |
| DINOv2 GPU memory (P3) | Distillation phase | Training completes without OOM at target batch size |
| Soft SimOTA zero positives (P4) | Soft label phase | num_fg > 0.5 * num_gt average over first 5 epochs |
| Dual-head inconsistent matching (P5) | NMS-free phase | O2O AP within 2 AP of O2M+NMS evaluation |
| EMA state dict mismatch (P6) | Infrastructure phase | 2 train steps + 1 val step completes without error for each feature toggle |
| Distillation loss scale (P7) | Distillation phase | All loss components within 2 orders of magnitude of each other |
| ONNX export breakage (P8) | Every head-modifying phase | ONNX export + onnxruntime inference test in CI |
| MAL circular dependency (P9) | MAL phase | Training loss decreases monotonically; cls and reg losses do not diverge |
| Combinatorial configs (P10) | Infrastructure phase | Smoke test for all blessed configurations passes in CI |

## Sources

- [GFL Paper: Generalized Focal Loss](https://arxiv.org/abs/2006.04388) -- DFL bin design and reg_max
- [YOLOv10 Paper: Dual Label Assignment](https://arxiv.org/html/2405.14458v2) -- consistent matching metric, top-1 vs Hungarian
- [YOLOv10 Architecture Deep Dive (Roboflow)](https://blog.roboflow.com/what-is-yolov10/) -- O2M/O2O interaction
- [YOLOX SimOTA Explanation (Medium)](https://gmongaras.medium.com/yolox-explanation-simota-for-dynamic-label-assignment-8fa5ae397f76) -- dynamic-k matching details
- [DINOv2 Engineer's Deep Dive (Lightly)](https://www.lightly.ai/blog/dinov2) -- memory optimization, feature extraction
- [DINOv2 Paper](https://arxiv.org/html/2304.07193v2) -- model sizes, distillation approach
- [Gradient-Guided KD for Object Detectors (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Lan_Gradient-Guided_Knowledge_Distillation_for_Object_Detectors_WACV_2024_paper.pdf) -- feature alignment and gradient stability
- [Mutual Supervision for Dense Object Detection (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Mutual_Supervision_for_Dense_Object_Detection_ICCV_2021_paper.pdf) -- MAL temperature and positive sample control
- [YOLO26 Paper](https://www.arxiv.org/pdf/2509.25164v1) -- DFL removal rationale for export simplicity
- [LearnOpenCV: GFL and VFL Loss](https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/) -- DFL implementation walkthrough
- Current codebase: `yolo_head.py`, `yolox_lightning.py`, `ema.py`, `onnx_export.py` -- verified against actual implementation

---
*Pitfalls research for: DINO-X object detection training improvements*
*Researched: 2026-03-05*
