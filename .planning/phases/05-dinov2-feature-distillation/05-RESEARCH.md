# Phase 5: DINOv2 Feature Distillation - Research

**Researched:** 2026-03-06
**Domain:** Feature-level knowledge distillation from frozen DINOv2 ViT-B/14 teacher to YOLOX-based DINOX student
**Confidence:** HIGH

## Summary

Phase 5 adds a frozen DINOv2-B/14 teacher model that provides feature-level supervision during training. The teacher loads via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` under Apache 2.0 license, produces 768-dim patch embeddings, and is completely excluded from inference and ONNX export. Projector modules (1x1 Conv + BN) map each student FPN feature level's channel count to the teacher's 768-dim embedding space, with bilinear interpolation handling spatial dimension mismatches. The distillation loss (SmoothL1 or MSE) is added to the total loss with a configurable weight lambda.

The key architectural challenge is spatial alignment. The student YOLOPAFPN produces feature maps at strides 8/16/32 (80x80, 40x40, 20x20 for 640x640 input), while DINOv2 with patch_size=14 on a 640x640 input produces 45x45 spatial patches (640//14=45, with 10 pixels cropped). The projector uses `F.interpolate(mode='bilinear')` to resize student features to match teacher spatial dimensions at each level. The teacher extracts features at configurable ViT block indices (default: layers 4, 8, 12 of the 12-block ViT-B), each producing the same 768-dim embeddings at 45x45 spatial resolution.

The ICIP 2025 paper "Improving YOLOv8 for Fast Few-Shot Object Detection by DINOv2 Distillation" (Fourret et al.) validates this exact architecture pattern: 1x1 conv projectors, SmoothL1 loss, frozen DINOv2 teacher, bilinear spatial resize, with all distillation weights removable at inference for zero overhead. Their d2 variant (integrating distillation into existing branches rather than adding a separate branch) achieved the best results, suggesting projectors should tap into the existing FPN feature flow rather than adding a parallel pathway.

**Primary recommendation:** Implement distillation as a self-contained `DistillationModule` (nn.Module) owned by `DINOXLightningModel`, containing the frozen teacher, projectors, and loss computation. The DINOX model and DINOXHead remain unmodified. The Lightning model's forward/training_step orchestrates teacher feature extraction and distillation loss, and the module is excluded from ONNX export via the existing `_export_mode` flag.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.hub (dinov2) | latest | Load DINOv2-B/14 pretrained teacher | Official Meta distribution, Apache 2.0 |
| torch.nn | (existing) | 1x1 Conv2d + BatchNorm2d projectors | Standard PyTorch layers |
| torch.nn.functional | (existing) | `F.interpolate(mode='bilinear')` for spatial alignment, `F.smooth_l1_loss` or `F.mse_loss` for distillation | Standard PyTorch functional API |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hydra | (existing) | Distillation config flags in DINOXConfig and Hydra YAML | Config management |
| lightning | (existing) | Training hook integration, A100 trainer config | Training framework |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SmoothL1 loss | MSE loss | MSE is simpler but more sensitive to outliers; SmoothL1 more robust. Both work. Start with MSE for simplicity, it's the more common choice in feature distillation literature |
| Cosine similarity loss | SmoothL1/MSE | Cosine is scale-invariant but requires careful normalization; L2/SmoothL1 is simpler to implement and debug |

**No new pip/pixi dependencies required.** DINOv2 loads via `torch.hub` which downloads the model weights at runtime.

## Architecture Patterns

### Distillation Module Structure

```
DINOXLightningModel (modified)
  model: DINOX                    # UNCHANGED - backbone + head
  distillation: DistillationModule | None  # NEW - only when enable_distillation=True
    teacher: nn.Module            # Frozen DINOv2-B/14 (all requires_grad=False)
    projectors: nn.ModuleList     # One per FPN level: Conv2d(C_student, 768, 1) + BN
    teacher_layer_indices: list   # e.g., [4, 8, 12] (0-indexed ViT block indices)
    distill_weight: float         # lambda_distill for loss weighting
```

### Key Design Decisions

**1. DistillationModule as separate nn.Module (not inside DINOXHead):**
- DINOXHead remains a pure detection head, no teacher dependency
- Clean separation: detection forward path is unmodified
- ONNX export naturally excludes DistillationModule since the export path goes through `self.model` only
- Teacher parameters never enter the optimizer (frozen + separate from `self.model`)

**2. Teacher feature extraction flow:**
```python
# In DistillationModule.forward():
teacher_features = self.teacher.get_intermediate_layers(
    images, n=self.teacher_layer_indices, reshape=True
)
# Returns tuple of (B, 768, H_t, W_t) tensors, one per requested layer
# For 640x640 input: H_t = W_t = 45 (640 // 14)
```

**3. Projector + spatial alignment flow:**
```python
# For each FPN level k and corresponding teacher layer:
student_feat = fpn_outputs[k]  # (B, C_k, H_s, W_s)
projected = self.projectors[k](student_feat)  # (B, 768, H_s, W_s)
# Spatially align to teacher dimensions:
projected = F.interpolate(projected, size=(H_t, W_t), mode='bilinear', align_corners=False)
teacher_feat = teacher_features[k]  # (B, 768, H_t, W_t)
loss_k = F.mse_loss(projected, teacher_feat.detach())
```

**4. Loss integration:**
```
L_total = L_detection + lambda_distill * sum(L_level for each FPN level)
```

### Recommended Module Placement

```
src/object_detection_training/models/dinox/
  distillation.py          # NEW: DistillationModule class
  config.py                # MODIFIED: add distillation config fields
src/object_detection_training/models/
  dinox_lightning.py        # MODIFIED: create/use DistillationModule, add distill loss
src/object_detection_training/conf/
  models/dinox_base.yaml    # MODIFIED: add distillation defaults (off)
  trainer/gpu_a100.yaml     # NEW: A100-specific trainer config
tests/
  test_dinox_distillation.py  # NEW: distillation unit tests
```

### Student FPN Feature Dimensions (YOLOX-M, width=0.75, 640x640 input)

| FPN Level | Stride | Spatial Size | Channels (int(C * 0.75)) |
|-----------|--------|-------------|--------------------------|
| P3 (pan_out2) | 8 | 80x80 | 192 |
| P4 (pan_out1) | 16 | 40x40 | 384 |
| P5 (pan_out0) | 32 | 20x20 | 768 |

### DINOv2 ViT-B/14 Teacher Output Dimensions (640x640 input)

| Layer Index | Spatial Size | Embedding Dim |
|-------------|-------------|---------------|
| Block 4 | 45x45 | 768 |
| Block 8 | 45x45 | 768 |
| Block 12 (last) | 45x45 | 768 |

Note: ViT-B/14 has 12 transformer blocks (0-indexed: 0-11). The `get_intermediate_layers` method accepts 0-indexed block indices. Layer 12 means the output after the last block (index 11), but by convention `n=[4, 8, 12]` in the method means taking blocks at those indices. Since ViT-B only has 12 blocks (0-11), the default should be `[3, 7, 11]` (0-indexed) to get layers 4, 8, 12 in 1-indexed terms.

### Projector Specifications

| FPN Level | Input Channels | Output Channels | Spatial In | Spatial Out (bilinear) |
|-----------|---------------|----------------|-----------|----------------------|
| P3 | 192 | 768 | 80x80 | 45x45 |
| P4 | 384 | 768 | 40x40 | 45x45 |
| P5 | 768 | 768 | 20x20 | 45x45 |

Each projector: `nn.Sequential(nn.Conv2d(C_in, 768, 1, bias=False), nn.BatchNorm2d(768))`

### ONNX Export Exclusion Pattern

The existing codebase already handles this naturally:

```python
# In DINOXLightningModel.forward():
if self._export_mode:
    result = self.model(images)  # Only self.model is traced, not self.distillation
    return result
```

The `export_onnx` method calls `self.set_export_mode(True)` then traces `self` with a dummy input. Since the export-mode forward only calls `self.model(images)`, the distillation module (teacher + projectors) is never traced and excluded from ONNX.

### Anti-Patterns to Avoid

- **Putting teacher inside DINOXHead:** This would couple detection logic with distillation, making ONNX export fragile and the head harder to test independently.
- **Including projectors in optimizer param groups without the teacher:** The teacher must have `requires_grad=False` on ALL parameters. Double-check that `configure_optimizers` only picks up student params. Since projectors are in `self.distillation` (not `self.model`), they need explicit inclusion in optimizer param groups.
- **Forgetting to handle the teacher's input preprocessing:** DINOv2 expects ImageNet-normalized input (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) in RGB. The YOLOX pipeline feeds raw 0-255 BGR. The DistillationModule must normalize and channel-swap before feeding to the teacher.
- **Computing teacher features with gradients:** Always wrap teacher forward in `torch.no_grad()` to save memory. Even though parameters are frozen, intermediate activations still consume memory if autograd tracks them.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DINOv2 model loading | Custom ViT implementation | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` | Official weights, tested, maintained |
| Intermediate layer extraction | Manual hook registration | `model.get_intermediate_layers(x, n=[3,7,11], reshape=True)` | Built-in method handles CLS token removal and spatial reshape |
| Spatial interpolation | Custom resize layers | `F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)` | Standard, differentiable, ONNX-compatible |
| Position embedding interpolation | Manual pos-embed resize | DINOv2 handles this internally via `interpolate_pos_encoding` | Built into the model's forward pass |

**Key insight:** DINOv2's `get_intermediate_layers` method is specifically designed for this use case. It handles CLS token separation, optional reshape to spatial format, and optional layer normalization. Using it directly avoids re-implementing ViT internals.

## Common Pitfalls

### Pitfall 1: Input Preprocessing Mismatch
**What goes wrong:** DINOv2 produces garbage features because it receives raw 0-255 BGR pixels instead of ImageNet-normalized RGB.
**Why it happens:** YOLOX's pipeline skips normalization (raw 0-255 float values). DINOv2 was trained with ImageNet normalization.
**How to avoid:** In DistillationModule, normalize and convert before teacher forward:
```python
# Convert BGR (YOLOX convention) to RGB, then normalize
images_rgb = images[:, [2, 1, 0], :, :]  # BGR -> RGB
images_norm = (images_rgb / 255.0 - mean) / std  # ImageNet normalize
```
**Warning signs:** Distillation loss stays constant/high and never decreases.

### Pitfall 2: 640 Not Divisible by Patch Size 14
**What goes wrong:** DINOv2 silently crops input to 630x630 (45*14=630), wasting 10 pixels per dimension.
**Why it happens:** ViT patch embedding uses integer division: `640 // 14 = 45` patches, covering only 630 pixels.
**How to avoid:** This is actually fine -- DINOv2's `interpolate_pos_encoding` handles it, and the 10-pixel crop has negligible impact. Just be aware that teacher spatial output is 45x45, not 46x46. Document the expected spatial dimensions clearly.
**Warning signs:** None -- this works correctly, just needs awareness for dimension calculations.

### Pitfall 3: Teacher Gradients Consuming GPU Memory
**What goes wrong:** OOM despite teacher being "frozen" because autograd still tracks intermediate activations.
**Why it happens:** `requires_grad=False` prevents gradient updates but not activation storage. PyTorch still builds computation graph through frozen modules unless `torch.no_grad()` is used.
**How to avoid:** Always wrap teacher forward in `torch.no_grad()`:
```python
with torch.no_grad():
    teacher_features = self.teacher.get_intermediate_layers(...)
```
**Warning signs:** GPU memory usage 2-3x higher than expected; OOM on A100-40GB with reasonable batch sizes.

### Pitfall 4: Projector Parameters Missing from Optimizer
**What goes wrong:** Projectors never learn because they're not in any optimizer param group.
**Why it happens:** The existing `configure_optimizers` in DINOXLightningModel iterates over `self.model.named_modules()`, but projectors live in `self.distillation`.
**How to avoid:** Modify `configure_optimizers` to include projector parameters in the weight-decay param group (pg1). Explicitly add: `for m in self.distillation.projectors.modules(): ...`
**Warning signs:** Distillation loss never decreases from initial value.

### Pitfall 5: Forgetting to Detach Teacher Features
**What goes wrong:** Gradient flows backward through teacher despite frozen params, wasting compute.
**Why it happens:** Even with `requires_grad=False`, if teacher forward happens inside `torch.no_grad()` this is handled. But if someone removes the no_grad context, gradients flow through.
**How to avoid:** Belt-and-suspenders: use both `torch.no_grad()` wrapper AND `.detach()` on teacher outputs.

### Pitfall 6: EMA Callback Including Distillation Parameters
**What goes wrong:** EMA shadow copies teacher parameters (86M extra params), or crashes on state dict mismatch when distillation is toggled.
**Why it happens:** EMA callback iterates all model parameters.
**How to avoid:** EMA should only shadow `self.model` parameters, not `self.distillation`. The existing EMA callback already operates on `model.model` (the DINOX nn.Module), so this should work naturally since distillation is a sibling attribute on the Lightning module.

## Code Examples

### DistillationModule Implementation Pattern

```python
# Source: Derived from DINOv2 official API + ICIP 2025 Fourret et al.
class DistillationModule(nn.Module):
    def __init__(
        self,
        student_channels: list[int],  # e.g., [192, 384, 768] for YOLOX-M
        teacher_embed_dim: int = 768,
        teacher_layer_indices: list[int] | None = None,  # 0-indexed
        distill_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if teacher_layer_indices is None:
            teacher_layer_indices = [3, 7, 11]  # layers 4, 8, 12 (1-indexed)
        self.teacher_layer_indices = teacher_layer_indices
        self.distill_weight = distill_weight

        # Load frozen teacher
        self.teacher = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True
        )
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # ImageNet normalization constants
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Projectors: one per FPN level
        assert len(student_channels) == len(teacher_layer_indices)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, teacher_embed_dim, 1, bias=False),
                nn.BatchNorm2d(teacher_embed_dim),
            )
            for c in student_channels
        ])

    def forward(
        self,
        images: torch.Tensor,          # Raw 0-255 BGR from YOLOX pipeline
        student_features: list[torch.Tensor],  # FPN outputs [P3, P4, P5]
    ) -> torch.Tensor:
        # Preprocess for teacher: BGR->RGB, normalize
        images_rgb = images[:, [2, 1, 0], :, :]
        images_norm = (images_rgb / 255.0 - self.mean) / self.std

        # Teacher forward (no grad for memory efficiency)
        with torch.no_grad():
            teacher_features = self.teacher.get_intermediate_layers(
                images_norm,
                n=self.teacher_layer_indices,
                reshape=True,  # Returns (B, D, H, W) format
            )

        # Compute per-level distillation loss
        total_loss = torch.tensor(0.0, device=images.device)
        for k, (student_feat, teacher_feat) in enumerate(
            zip(student_features, teacher_features, strict=True)
        ):
            projected = self.projectors[k](student_feat)
            # Spatial alignment via bilinear interpolation
            target_h, target_w = teacher_feat.shape[2], teacher_feat.shape[3]
            projected = F.interpolate(
                projected, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
            total_loss = total_loss + F.mse_loss(projected, teacher_feat.detach())

        return total_loss
```

### Lightning Integration Pattern

```python
# In DINOXLightningModel.__init__():
if enable_distillation:
    student_channels = [int(c * width) for c in in_channels]
    self.distillation = DistillationModule(
        student_channels=student_channels,
        teacher_layer_indices=distill_layer_indices,
        distill_weight=distill_weight,
    )
else:
    self.distillation = None

# In DINOXLightningModel.forward():
if self._export_mode:
    return self.model(images)  # Distillation excluded

# In DINOXLightningModel.training_step():
outputs = self(images, targets)
loss = outputs["total_loss"]
if self.distillation is not None:
    fpn_features = self.model.backbone(images_bgr)  # Need FPN outputs
    distill_loss = self.distillation(images, list(fpn_features))
    loss = loss + self.distillation.distill_weight * distill_loss
    self.log("train/distill_loss", distill_loss, ...)
```

### FPN Feature Extraction Pattern

Note: The current DINOX.forward() calls `self.backbone(x)` and passes results to the head, but does not expose FPN features externally. Two options:

**Option A (preferred): Pass images to distillation module separately, let it call backbone internally.**
Problem: This runs the backbone TWICE (once for detection, once for distillation).

**Option B (better): Modify DINOX.forward() to also return FPN features during training.**
```python
# In DINOX.forward():
fpn_outs = self.backbone(x)
if targets is not None:
    outputs = ...  # existing loss computation
    outputs["fpn_features"] = fpn_outs  # NEW: expose for distillation
    return outputs
```
This is cleaner -- backbone runs once, FPN features used by both head and distillation.

### A100 Trainer Config

```yaml
# src/object_detection_training/conf/trainer/gpu_a100.yaml
_target_: lightning.Trainer
max_epochs: 300
precision: 16-mixed  # bf16-mixed also works on A100
accelerator: gpu
devices: 1
strategy: auto
val_check_interval: 1.0
log_every_n_steps: 50
enable_checkpointing: true
enable_progress_bar: true
enable_model_summary: true
gradient_clip_val: 5.0
accumulate_grad_batches: 1  # A100 has enough memory for larger batch
num_sanity_val_steps: 2
```

Batch size recommendation for A100-80GB with distillation:
- Student YOLOX-M: ~26M params
- Teacher DINOv2-B/14: ~86M params (frozen, no grad storage)
- With mixed precision + no_grad teacher: batch_size=32-48 should fit
- Without mixed precision: batch_size=16-24
- The ICIP 2025 paper used batch_size=16 on single A100 for fine-tuning

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CC BY-NC 4.0 license | Apache 2.0 license | Aug 2023 | DINOv2 now freely usable for commercial training |
| Separate distillation branch | Integrate into existing branches | 2025 (ICIP) | d2 approach (projector on existing features) outperforms separate branch |
| Feature mimicking with L2 | SmoothL1 or MSE both viable | Ongoing | SmoothL1 more robust to outliers, MSE simpler; both produce comparable results |
| DINOv1 | DINOv2 | 2023 | Better features, register tokens reduce artifacts, ViT-B/14 is sweet spot for teacher size |

**Note on DINOv3:** DINOv3 was released in 2025 but is under a different license (not confirmed Apache 2.0). Stick with DINOv2 per requirements.

## Memory Budget Analysis

### Per-Component GPU Memory (fp16/mixed precision, 640x640 input)

| Component | Parameters | Grad Storage | Activation Memory | Total Estimate |
|-----------|-----------|-------------|-------------------|----------------|
| YOLOX-M backbone+FPN | ~15M | ~30MB | ~500MB/img | ~1GB at batch=16 |
| DINOXHead | ~11M | ~22MB | ~300MB/img | ~600MB at batch=16 |
| DINOv2-B/14 teacher (frozen) | 86M | 0 (frozen) | ~200MB/img (no_grad) | ~3.5GB at batch=16 |
| Projectors (3x) | ~0.8M | ~1.6MB | ~50MB/img | ~100MB at batch=16 |
| **Total** | **~113M** | **~54MB** | | **~5.2GB at batch=16** |

These are rough estimates. On A100-80GB, batch_size=32 with mixed precision should be comfortable (~10-12GB). On A100-40GB, batch_size=16 is safe.

## DINOXConfig Changes

Add these fields to the existing `DINOXConfig`:

```python
# Phase 5: Distillation
enable_distillation: bool = False
distill_weight: float = 0.5        # lambda_distill
distill_layer_indices: list[int] = [3, 7, 11]  # 0-indexed ViT block indices
distill_teacher: str = "dinov2_vitb14"  # torch.hub model name
```

## Open Questions

1. **Optimal distill_weight (lambda_distill)**
   - What we know: The ICIP 2025 paper doesn't report lambda tuning details. Common range is 0.1-1.0.
   - What's unclear: Best value for COCO full training (vs few-shot fine-tuning in the paper).
   - Recommendation: Default to 0.5, expose as config parameter for ablation.

2. **Layer index mapping: 0-indexed vs 1-indexed**
   - What we know: `get_intermediate_layers` uses 0-indexed block indices when a list is passed. ViT-B has 12 blocks (0-11).
   - What's unclear: The requirement says "layers 4, 8, 12" which in 1-indexed terms means blocks [3, 7, 11] in 0-indexed.
   - Recommendation: Use [3, 7, 11] as defaults, document the 0-indexed convention. Validate with unit test that output shapes are correct.

3. **Whether to normalize projected features before loss**
   - What we know: Some distillation methods L2-normalize both student and teacher features before computing loss. The ICIP 2025 paper uses unnormalized SmoothL1.
   - What's unclear: Whether normalization improves training stability for our architecture.
   - Recommendation: Start without normalization (simpler), make it a config option for future experimentation.

4. **FPN feature exposure pattern**
   - What we know: DINOX.forward() currently doesn't return FPN features.
   - Recommendation: Add `fpn_features` key to the training outputs dict from DINOX.forward(). This is a minimal change.

## Sources

### Primary (HIGH confidence)
- [facebookresearch/dinov2 GitHub](https://github.com/facebookresearch/dinov2) - Model architecture, hub API, get_intermediate_layers signature, license
- [dinov2/models/vision_transformer.py](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) - get_intermediate_layers implementation details, interpolate_pos_encoding behavior
- [dinov2/hub/backbones.py](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py) - torch.hub loading, patch_size=14 confirmation
- Existing codebase: DINOX model, DINOXHead, DINOXLightningModel, YOLOPAFPN - verified via source code reading

### Secondary (MEDIUM confidence)
- [ICIP 2025: Improving YOLOv8 for Fast Few-Shot Object Detection by DINOv2 Distillation](https://ieeexplore.ieee.org/document/11084724/) (Fourret et al.) - Projector architecture (1x1 conv), SmoothL1 loss, frozen teacher, spatial resize, batch sizes on A100
- [Meta announcement on DINOv2 relicensing to Apache 2.0](https://ai.meta.com/blog/dinov2-facet-computer-vision-fairness-evaluation/) - License confirmation
- [DINOv2 MODEL_CARD.md](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) - ViT-B/14 specs: 86M params, 768 embed_dim

### Tertiary (LOW confidence)
- GPU memory estimates are rough calculations based on parameter counts and typical activation sizes, not benchmarked

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - torch.hub API and DINOv2 architecture are well-documented and stable
- Architecture: HIGH - Pattern validated by ICIP 2025 paper and consistent with codebase structure
- Pitfalls: HIGH - Input preprocessing, memory management, and ONNX exclusion patterns are well-understood
- Memory budget: MEDIUM - Estimates not benchmarked on actual hardware, but conservative

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (DINOv2 API is stable; DINOv3 may warrant revisiting if license changes)
