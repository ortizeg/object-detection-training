# Phase 1: DFL Foundation and Infrastructure - Research

**Researched:** 2026-03-05
**Domain:** DFL box regression, DINO-X infrastructure, YOLOX-M extension
**Confidence:** HIGH

## Summary

Phase 1 establishes the foundation for all DINO-X improvements: a new `DINOXHead` with Distribution Focal Loss (DFL) box regression, `DINOXConfig` Pydantic model for feature flag validation, `DINOXLightningModel` for training integration, and Hydra configs for Phase 1 ablations. The ONNX export must bake the DFL distribution-to-point conversion into the graph so the downstream inference pipeline remains unchanged.

The codebase follows a clear pattern: `BaseDetectionModel` (Lightning module) -> model-specific Lightning class -> inner `nn.Module` (backbone + neck + head). YOLOX uses `YOLOXLightningModel` -> `YOLOX` -> `YOLOPAFPN` + `YOLOXHead`. DINO-X must follow this same pattern: `DINOXLightningModel` -> `DINOX` -> `YOLOPAFPN` + `DINOXHead`. Model variants are registered with Hydra via the `@register` decorator, and configs chain through `defaults` inheritance (variant -> base -> model base).

**Primary recommendation:** Create `DINOXHead` as a fresh class (not subclass of `YOLOXHead`) that reuses the same building blocks (`BaseConv`, `DWConv`) and duplicates the structural pattern but changes `reg_preds` output from 4 channels to `4 * (reg_max + 1)` channels. When `use_dfl=False`, skip the DFL integral and use standard YOLOX-style decode (exp-based), making it functionally equivalent to `YOLOXHead`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Claude decides whether to subclass YOLOXHead or create a fresh class that reuses internals -- pick the approach that best fits the codebase
- When DFL is disabled (use_dfl=False), DINOXHead must be functionally equivalent to YOLOXHead (same architecture and behavior, small numerical differences OK -- not required to be bit-identical)
- Must support loading pretrained YOLOX-M checkpoint weights where architecturally compatible (backbone, neck, shared conv layers)
- DINOXLightningModel goes in a new file: `dinox_lightning.py` -- clean separation from existing yolox_lightning.py
- Claude decides how DINOXConfig integrates with Hydra (nested under model config vs separate config group) -- follow existing codebase patterns
- Strict Pydantic validation for invalid flag combinations (e.g., use_mal=True without use_soft_labels=True) -- fail fast with clear errors
- Only create Phase 1 Hydra configs now (dinox_baseline.yaml, dinox_dfl.yaml) -- add others as phases implement their components
- Claude decides whether to enforce blessed configs only or allow arbitrary flag combinations via CLI overrides
- Claude determines the exact ONNX output format by reading the existing export and eval code
- DFL distribution-to-point conversion MUST be baked into the ONNX graph (softmax + weighted sum inside the model) -- inference code stays unchanged
- Claude decides whether to include an automated ONNX round-trip test in CI or as a manual script
- ONNX export must support dynamic batch size
- Models/yolox/ files: minor interface additions OK (new methods, making internal methods public) but no behavior changes
- Shared utilities (utils/boxes.py, callbacks/onnx_export.py): can add new functions/methods but don't modify existing signatures or behavior
- All existing YOLOX/RFDETR tests must continue passing unchanged -- DINO-X is purely additive
- Modify task_manager.py to route DINO-X configs -- single `pixi run train` entry point for all models

### Claude's Discretion
- DINOXHead inheritance/composition strategy
- Hydra config group structure (nested vs separate)
- Blessed configs vs arbitrary flag combinations
- ONNX round-trip test location (CI vs manual script)
- Exact ONNX output format (determined by reading existing code)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | existing | DFL module, head implementation | Already in project |
| Pydantic | >=2.0 | DINOXConfig validation | Already used in project (`pyproject.toml` constraint) |
| Hydra | existing | Config composition for ablation variants | Already used for all models |
| Lightning | existing | DINOXLightningModel training integration | Already used for all models |
| ONNX | existing | Export with DFL baked into graph | Already used, opset 17 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| onnxsim | existing | Simplify exported ONNX model | During export (already integrated) |
| onnxruntime | existing | ONNX round-trip validation | Testing DFL bake-in correctness |

**No new dependencies needed.** All required libraries are already in the project.

## Architecture Patterns

### Recommended Project Structure
```
src/object_detection_training/
  models/
    dinox/
      __init__.py               # Exports DINOXHead, DFLModule, DINOXConfig
      config.py                 # DINOXConfig (Pydantic)
      dinox.py                  # DINOX nn.Module (backbone + neck + head)
      dinox_head.py             # DINOXHead (decoupled head with DFL)
      dfl.py                    # DFLModule (distribution -> coords integral)
    dinox_lightning.py          # DINOXLightningModel (extends BaseDetectionModel)
  conf/
    models/
      dinox_base.yaml           # Base DINO-X config (extends base.yaml)
      dinox_m.yaml              # DINO-X Medium variant (extends dinox_base)
      dinox_m_baseline.yaml     # Phase 1: use_dfl=false (YOLOX-M equivalent)
      dinox_m_dfl.yaml          # Phase 1: use_dfl=true (DFL enabled)
tests/
  test_dinox_config.py          # DINOXConfig validation tests
  test_dinox_head.py            # DINOXHead forward/loss tests
  test_dfl_module.py            # DFL integral + loss tests
  test_dinox_parameterization.py # Hydra config tests (follows test_yolox_parameterization.py pattern)
  test_dinox_onnx.py            # ONNX export round-trip test
```

### Pattern 1: Fresh DINOXHead Class (Recommended)

**What:** Create `DINOXHead` as a new `nn.Module` that follows the same decoupled-head pattern as `YOLOXHead` but with DFL regression output. Not a subclass -- a peer class that reuses building blocks.

**When to use:** Always for Phase 1. The regression output changes from 4 channels to `4 * (reg_max + 1)` channels, which is a fundamental architectural difference.

**Why not subclass:** `YOLOXHead.__init__` hardcodes `reg_preds` with output channels of 4. Subclassing would require overriding `__init__`, `forward`, `get_output_and_grid`, `decode_outputs`, and `get_losses` -- essentially everything. A fresh class is cleaner and more maintainable.

**Key design:**
```python
# dinox/dinox_head.py
class DINOXHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        strides: list[int] = [8, 16, 32],
        in_channels: list[int] = [256, 512, 1024],
        act: str = "silu",
        depthwise: bool = False,
        use_dfl: bool = False,
        reg_max: int = 16,
    ):
        # stems, cls_convs, reg_convs, cls_preds, obj_preds
        # -- identical to YOLOXHead

        # reg_preds output channels change based on use_dfl:
        reg_out = 4 * (reg_max + 1) if use_dfl else 4
        for i in range(len(in_channels)):
            self.reg_preds.append(
                nn.Conv2d(int(256 * width), reg_out, 1, 1, 0)
            )

        # DFL module (only used when use_dfl=True)
        if use_dfl:
            self.dfl = DFLModule(reg_max=reg_max)
```

**Weight loading compatibility:** When `use_dfl=False`, `reg_preds` has output channels of 4, matching YOLOX-M checkpoint weights exactly. When `use_dfl=True`, `reg_preds` weights from YOLOX-M checkpoint will be skipped (shape mismatch: 4 vs 68), similar to how `cls_preds` are skipped when `num_classes` differs. The existing `_load_weights` pattern handles this gracefully.

### Pattern 2: DFLModule (Distribution-to-Point Integral)

**What:** A small `nn.Module` that converts `4 * (reg_max + 1)` distribution logits into 4 continuous LTRB (left, top, right, bottom) distances via softmax + weighted sum.

**Why a module:** Needs a registered buffer (`project = arange(0, reg_max + 1)`) for the integral weights. Being a module ensures this buffer moves with the model to the correct device and is included in the ONNX graph.

```python
# dinox/dfl.py
class DFLModule(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project",
            torch.arange(0, reg_max + 1, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert distribution logits to point estimates.

        Args:
            x: [..., 4 * (reg_max + 1)] distribution logits
        Returns:
            [..., 4] continuous LTRB distances
        """
        shape = x.shape[:-1]  # preserve leading dims
        x = x.reshape(*shape, 4, self.reg_max + 1)
        x = F.softmax(x, dim=-1)
        x = (x * self.project).sum(dim=-1)
        return x
```

### Pattern 3: LTRB-to-CXCYWH Box Decoding

**What:** DFL produces LTRB (left, top, right, bottom) distances from anchor points. These must be converted to CXCYWH for compatibility with existing YOLOX output format.

**Why this matters:** YOLOX uses `(cx + grid) * stride` and `exp(wh) * stride` decoding. DFL uses distance-from-anchor format: `anchor_x - left, anchor_y - top, anchor_x + right, anchor_y + bottom`. The decode path diverges based on `use_dfl`.

```python
# In DINOXHead.decode_outputs / forward (inference mode):
if self.use_dfl:
    # reg_output: [B, 4*(reg_max+1), H, W] -> DFLModule -> [B, 4, H, W] LTRB
    ltrb = self.dfl(reg_output)  # distances from anchor in stride units
    # Convert LTRB to CXCYWH in pixel coordinates:
    cx = (anchor_x + (ltrb[..., 2] - ltrb[..., 0]) / 2) * stride
    cy = (anchor_y + (ltrb[..., 3] - ltrb[..., 1]) / 2) * stride
    w = (ltrb[..., 0] + ltrb[..., 2]) * stride
    h = (ltrb[..., 1] + ltrb[..., 3]) * stride
else:
    # Standard YOLOX decode
    cx = (output[..., 0] + grid_x) * stride
    cy = (output[..., 1] + grid_y) * stride
    w = exp(output[..., 2]) * stride
    h = exp(output[..., 3]) * stride
```

### Pattern 4: DINOXConfig Pydantic Model

**What:** A Pydantic `BaseModel` that validates all DINO-X feature flags. Phase 1 defines the full schema but only implements `use_dfl` and `reg_max`. Other flags exist with safe defaults (all `False`).

**Integration with Hydra:** Follow the existing pattern -- config values flow through Hydra YAML into `DINOXLightningModel.__init__` as keyword arguments. `DINOXConfig` is constructed inside `__init__` from those kwargs. This matches how existing models work (they don't use a separate config object in Hydra, but validate internally).

```python
# dinox/config.py
from pydantic import BaseModel, field_validator, model_validator

class DINOXConfig(BaseModel):
    """Validates all DINO-X improvement flags."""
    model_config = {"frozen": True}  # immutable after creation

    # Phase 1: DFL
    use_dfl: bool = False
    reg_max: int = 16
    dfl_loss_weight: float = 0.25

    # Future phases (safe defaults)
    use_soft_labels: bool = False
    soft_label_gamma: float = 2.0
    use_mal: bool = False
    mal_gamma: float = 1.5
    use_dual_head: bool = False
    enable_distillation: bool = False
    use_scheduler_free: bool = False
    assigner: str = "simota"
    iou_loss_type: str = "iou"

    @field_validator("reg_max")
    @classmethod
    def validate_reg_max(cls, v: int) -> int:
        if v < 1 or v > 32:
            raise ValueError(f"reg_max must be in [1, 32], got {v}")
        return v

    @model_validator(mode="after")
    def validate_combinations(self) -> "DINOXConfig":
        if self.use_mal and not self.use_soft_labels:
            raise ValueError(
                "use_mal=True requires use_soft_labels=True "
                "(MAL needs IoU-weighted soft targets)"
            )
        return self
```

### Pattern 5: Model Registration with Hydra

**What:** Follow the existing `@register` decorator pattern for DINO-X model variants.

```python
# In dinox_lightning.py
@register(name="DINOXMBaseline")
class DINOXMBaselineModel(DINOXLightningModel):
    _checkpoint_name = "yolox_m.pth"
    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        kwargs.setdefault("use_dfl", False)  # YOLOX-M equivalent
        super().__init__(**kwargs)

@register(name="DINOXMDFL")
class DINOXMDFLModel(DINOXLightningModel):
    _checkpoint_name = "yolox_m.pth"
    def __init__(self, **kwargs):
        kwargs.pop("variant", None)
        kwargs.setdefault("checkpoint_name", self._checkpoint_name)
        kwargs.setdefault("use_dfl", True)
        super().__init__(**kwargs)
```

### Pattern 6: ONNX Output Format

**What:** The YOLOX ONNX export produces a single tensor `[batch, num_anchors, 5 + num_classes]` with columns `[cx, cy, w, h, obj_conf, cls_0, cls_1, ...]` in pixel coordinates. The `YOLOXPostProcessor` expects this exact format.

**DINO-X must match this format.** The DFL integral must be computed inside the model graph so the ONNX output is `[B, N, 5 + num_classes]` -- not the raw `[B, N, 4*(reg_max+1) + 1 + num_classes]` distribution.

The export uses:
- Input name: `"input"`, dynamic axis `{0: "batch_size"}`
- Output name: `"output"`, dynamic axis `{0: "batch_size"}`
- `torch.onnx.export` with opset 17

**DFL ONNX compatibility:** `F.softmax`, tensor multiplication, and `sum` are all standard ONNX ops. The `DFLModule.forward` will trace cleanly. The `register_buffer("project")` tensor becomes a constant in the ONNX graph. Verified: these ops are supported in opset 17.

### Anti-Patterns to Avoid

- **Subclassing YOLOXHead:** Would require overriding nearly every method. Use composition with shared building blocks instead.
- **Inheriting from YOLOXLightningModel:** Same problem -- `__init__` constructs a `YOLOXHead`. Extend `BaseDetectionModel` directly.
- **Modifying `yolox/yolo_head.py`:** Third-party code. Create new files instead. Constraint: no behavior changes to existing files.
- **Passing DINOXConfig through Hydra as a structured config:** Too complex. Pass individual parameters through Hydra YAML (matching existing pattern), construct DINOXConfig internally.
- **Adding DFL as a post-processing step outside the model:** Breaks ONNX export requirement. DFL integral must be inside `nn.Module.forward()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DFL loss function | Custom cross-entropy over bins | `F.cross_entropy` with interpolated targets (GFL paper pattern) | The DFL loss is simply weighted cross-entropy between adjacent discrete bins. The implementation is 5 lines (see Code Examples). |
| ONNX export infrastructure | Custom ONNX export logic | Existing `export_onnx` method on `BaseDetectionModel` + override in `DINOXLightningModel` | Already handles dynamic axes, simplification, and output path management. |
| Weight loading with shape mismatches | Custom weight filter | Copy/reuse `YOLOXLightningModel._load_weights` pattern | Already handles class-dimension mismatches, shape mismatches, and logging. |
| Model registration with Hydra | Manual ConfigStore entries | `@register` decorator from `utils/hydra.py` | Already handles target path, group inference, and ConfigStore registration. |
| EMA integration | Custom EMA for new model | Existing `EMACallback` | Works on full `state_dict` -- automatically handles new parameters. No changes needed. |

## Common Pitfalls

### Pitfall 1: DFL LTRB vs YOLOX CXCYWH Coordinate Mismatch
**What goes wrong:** DFL produces LTRB distances from anchor points, but the existing loss computation (SimOTA, IoU loss) expects CXCYWH format. Feeding LTRB into CXCYWH-expecting code produces garbage IoU values and no training signal.
**Why it happens:** YOLOX internally works in CXCYWH. DFL papers (GFL, YOLOv8, YOLO26) use LTRB. The conversion must happen at the right point in the pipeline.
**How to avoid:** Convert DFL LTRB output to CXCYWH immediately after the integral, before passing to SimOTA assignment or IoU loss. The training `get_losses` and inference `decode_outputs` paths must both do this conversion.
**Warning signs:** IoU loss is always near 1.0 (or NaN), SimOTA assigns zero foregrounds, training loss does not decrease.

### Pitfall 2: DFL Target Computation for CXCYWH Ground Truth
**What goes wrong:** The DFL loss expects target distances in LTRB format relative to anchor points (in stride units). Ground truth boxes are in pixel CXCYWH format. Incorrect conversion produces wrong DFL targets.
**Why it happens:** Need to convert GT CXCYWH to LTRB relative to each assigned anchor: `left = (anchor_x * stride - (gt_cx - gt_w/2)) / stride`, etc. Missing the stride division or anchor offset breaks the discrete bin assignment.
**How to avoid:** Write a dedicated `bbox_to_ltrb_target(gt_cxcywh, anchor_points, strides)` utility function. Unit test it independently. Verify that `ltrb_to_cxcywh(ltrb_target)` recovers the original GT box.
**Warning signs:** DFL loss is very high or NaN. Distribution targets fall outside `[0, reg_max]` range.

### Pitfall 3: reg_preds Weight Loading from YOLOX-M Checkpoint
**What goes wrong:** When `use_dfl=True`, `reg_preds` output channels change from 4 to `4 * (reg_max + 1) = 68`. YOLOX-M checkpoint `reg_preds` weights have shape `[4, C, 1, 1]`. Loading will fail unless handled.
**Why it happens:** Shape mismatch between pretrained weights and model architecture.
**How to avoid:** The existing `_load_weights` pattern in `YOLOXLightningModel` already skips mismatched shapes with logging. Replicate this pattern in `DINOXLightningModel._load_weights`. Log that `reg_preds` weights were skipped. Initialize `reg_preds` with proper bias (zero or small values for DFL distributions).
**Warning signs:** `RuntimeError: size mismatch` during weight loading, or model training with uninitialized `reg_preds` (random initial predictions, very high initial loss).

### Pitfall 4: DFL Module Buffer Not in ONNX Graph
**What goes wrong:** The `project` buffer (`arange(0, reg_max+1)`) is not included in the ONNX graph because it was created as a regular tensor instead of a registered buffer.
**Why it happens:** Using `self.project = torch.arange(...)` instead of `self.register_buffer("project", torch.arange(...))`.
**How to avoid:** Always use `register_buffer` for the integral weights. Verify with an ONNX round-trip test that the integral computation is present in the exported graph.
**Warning signs:** ONNX export succeeds but inference produces wrong box coordinates (raw distribution logits instead of integrated coordinates).

### Pitfall 5: Forgetting to Update n_ch in get_output_and_grid
**What goes wrong:** `YOLOXHead.get_output_and_grid` calculates `n_ch = 5 + self.num_classes` to reshape the concatenated output. With DFL, the channel count changes to `4 * (reg_max + 1) + 1 + num_classes` during training.
**Why it happens:** The reshape operation depends on knowing the exact channel count.
**How to avoid:** Compute `n_ch` dynamically: `reg_channels = 4 * (self.reg_max + 1) if self.use_dfl else 4; n_ch = reg_channels + 1 + self.num_classes`.
**Warning signs:** `RuntimeError: shape '[B, 1, n_ch, H, W]' is invalid for input of size X`.

### Pitfall 6: EMA Callback Compatibility with Dynamic Parameters
**What goes wrong:** If future phases add modules dynamically (e.g., distillation projectors added after EMA initialization), the EMA state dict will be missing those keys.
**Why it happens:** `EMACallback.on_fit_start` copies `pl_module.state_dict()` once. New parameters added later are not tracked.
**How to avoid:** For Phase 1, this is not an issue (all parameters exist at init time). For future phases, the EMA callback may need a `_sync_new_keys` method. Flag this as a known concern for Phase 5.
**Warning signs:** `RuntimeError: unexpected key` when restoring EMA weights after adding distillation modules.

## Code Examples

### DFL Loss Function (from GFL paper, verified via mmdetection)
```python
# Source: https://mmdetection.readthedocs.io/en/v2.9.0/_modules/mmdet/models/losses/gfocal_loss.html
def distribution_focal_loss(
    pred: torch.Tensor,    # [N, reg_max + 1] logits
    target: torch.Tensor,  # [N] continuous target distances
) -> torch.Tensor:
    """DFL loss: cross-entropy between adjacent bins, weighted by distance.

    Target value 3.7 produces:
      - loss at bin 3 weighted by 0.3 (distance to right bin)
      - loss at bin 4 weighted by 0.7 (distance to left bin)
    """
    dis_left = target.long()              # floor bin index
    dis_right = dis_left + 1              # ceil bin index
    weight_left = dis_right.float() - target   # 1 - fractional part
    weight_right = target - dis_left.float()   # fractional part
    loss = (
        F.cross_entropy(pred, dis_left, reduction="none") * weight_left
        + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )
    return loss
```

### DFL Integral (Distribution to Point)
```python
# Source: GFL paper, verified in mmdetection and ultralytics
class DFLModule(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project",
            torch.arange(0, reg_max + 1, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., 4 * (reg_max + 1)]
        # Output: [..., 4] continuous LTRB distances
        *batch_dims, _ = x.shape
        x = x.reshape(*batch_dims, 4, self.reg_max + 1)
        x = F.softmax(x, dim=-1)
        x = (x * self.project).sum(dim=-1)
        return x
```

### LTRB to CXCYWH Conversion
```python
def ltrb_to_cxcywh(
    ltrb: torch.Tensor,       # [N, 4] left, top, right, bottom distances
    anchor_points: torch.Tensor,  # [N, 2] anchor (x, y) in grid units
    strides: torch.Tensor,    # [N] stride per anchor
) -> torch.Tensor:
    """Convert LTRB distances from anchors to pixel CXCYWH boxes."""
    l, t, r, b = ltrb.unbind(dim=-1)
    cx = (anchor_points[:, 0] + (r - l) / 2) * strides
    cy = (anchor_points[:, 1] + (b - t) / 2) * strides
    w = (l + r) * strides
    h = (t + b) * strides
    return torch.stack([cx, cy, w, h], dim=-1)
```

### CXCYWH GT to LTRB Target for DFL
```python
def cxcywh_to_ltrb_target(
    gt_cxcywh: torch.Tensor,     # [N, 4] pixel CXCYWH ground truth
    anchor_points: torch.Tensor,  # [N, 2] anchor (x, y) in grid units
    strides: torch.Tensor,        # [N] stride per anchor
    reg_max: int = 16,
) -> torch.Tensor:
    """Convert pixel CXCYWH GT boxes to LTRB targets in stride units.

    Targets are clamped to [0, reg_max] for valid DFL bin assignment.
    """
    # GT box edges in pixel coords
    x1 = gt_cxcywh[:, 0] - gt_cxcywh[:, 2] / 2
    y1 = gt_cxcywh[:, 1] - gt_cxcywh[:, 3] / 2
    x2 = gt_cxcywh[:, 0] + gt_cxcywh[:, 2] / 2
    y2 = gt_cxcywh[:, 1] + gt_cxcywh[:, 3] / 2

    # Anchor positions in pixel coords
    ax = anchor_points[:, 0] * strides
    ay = anchor_points[:, 1] * strides

    # LTRB distances in stride units
    left = (ax - x1) / strides
    top = (ay - y1) / strides
    right = (x2 - ax) / strides
    bottom = (y2 - ay) / strides

    ltrb = torch.stack([left, top, right, bottom], dim=-1)
    return ltrb.clamp(min=0, max=reg_max)
```

### Hydra Config Chain (following existing yolox_m.yaml pattern)
```yaml
# conf/models/dinox_base.yaml
defaults:
  - base
  - _self_

# Inherits from base.yaml: num_classes, input_height/width, learning_rate, etc.
learning_rate: 1e-3
weight_decay: 5e-4
warmup_epochs: 5
download_pretrained: true
pretrain_weights: null
freeze_backbone_epochs: 0
l1_loss_epoch: 0
iou_loss_type: iou

# Architecture (same defaults as yolox_base)
depth: 0.33
width: 0.50
depthwise: false
in_channels:
  - 256
  - 512
  - 1024

# DFL settings
use_dfl: false
reg_max: 16
dfl_loss_weight: 0.25

# YOLOX uses raw 0-255 float values
image_mean: null
image_std: null
```

```yaml
# conf/models/dinox_m_baseline.yaml
defaults:
  - DINOXMBaseline
  - dinox_base
  - _self_

depth: 0.67
width: 0.75
depthwise: false
checkpoint_name: yolox_m.pth
use_dfl: false
```

```yaml
# conf/models/dinox_m_dfl.yaml
defaults:
  - DINOXMDFL
  - dinox_base
  - _self_

depth: 0.67
width: 0.75
depthwise: false
checkpoint_name: yolox_m.pth
use_dfl: true
reg_max: 16
dfl_loss_weight: 0.25
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct 4-value box regression (YOLOX) | Distribution-based regression with DFL (YOLOv8, YOLO26, GFocal) | 2020 (GFL paper), widely adopted 2023+ | +1-2% mAP, better localization uncertainty modeling |
| exp-based box decode: `exp(pred) * stride` | LTRB distance from anchor via soft-argmax integral | GFL 2020, standardized in YOLOv8 2023 | More stable gradients, naturally bounded predictions |
| Hard one-hot classification targets | Soft IoU-weighted targets | GFL 2020, refined in RTMDet 2022 | Better alignment between classification and localization quality |
| `reg_max=7` (early GFL) | `reg_max=16` (standard) | YOLOv8 2023 | 17 bins per edge, good balance of precision vs parameter count |

**Standard `reg_max` value:** 16 is the standard used by YOLOv8, YOLO26, and recommended in the GFL paper for general detection. This means 17 bins per edge direction (0 to 16), producing `4 * 17 = 68` output values per anchor for regression.

## Open Questions

1. **DFL bias initialization for `reg_preds`**
   - What we know: YOLOX initializes `cls_preds` bias with `-log((1-p)/p)` for prior probability. For DFL `reg_preds`, the initialization should encourage a peaked distribution centered at the middle of the range.
   - What's unclear: Exact initialization strategy. YOLOv8 uses default initialization (zeros). GFL paper does not specify.
   - Recommendation: Use default initialization (zero bias, kaiming normal for weights). The DFL softmax naturally produces a uniform distribution initially, which is a reasonable starting point. Monitor first few training iterations to verify loss decreases.

2. **DFL loss weight (lambda_dfl)**
   - What we know: The combined loss is `L_GIoU + lambda_dfl * L_DFL`. YOLOv8 uses 1.5 for DFL weight. GFL paper uses 0.25.
   - What's unclear: Best weight for this codebase / dataset.
   - Recommendation: Start with 0.25 (GFL paper default). This is a hyperparameter that can be tuned via Hydra config (`dfl_loss_weight`).

3. **Whether `use_dfl=False` path should share code with YOLOXHead or duplicate**
   - What we know: When `use_dfl=False`, behavior must be functionally equivalent to YOLOXHead.
   - What's unclear: Whether to branch inside DINOXHead methods or create truly separate code paths.
   - Recommendation: Use if/else branching within DINOXHead methods. This is simpler than maintaining two separate implementations and makes the equivalence testable.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `yolox/yolo_head.py` (YOLOXHead architecture, 680 lines)
- Existing codebase: `models/yolox_lightning.py` (Lightning integration pattern, weight loading)
- Existing codebase: `models/base.py` (BaseDetectionModel interface, ONNX export)
- Existing codebase: `utils/hydra.py` (`@register` decorator pattern)
- Existing codebase: `inference/postprocess.py` (YOLOXPostProcessor, ONNX output format)
- [MMDetection DFL implementation](https://mmdetection.readthedocs.io/en/v2.9.0/_modules/mmdet/models/losses/gfocal_loss.html) - verified DFL loss formula

### Secondary (MEDIUM confidence)
- [GFL Paper (NeurIPS 2020)](https://arxiv.org/abs/2006.04388) - DFL theory, reg_max=16 standard
- [LearnOpenCV: GFL and VFL Loss](https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/) - DFL implementation walkthrough
- `.planning/research/ARCHITECTURE.md` - Prior codebase architecture research

### Tertiary (LOW confidence)
- [Ultralytics DFL discussion](https://github.com/ultralytics/ultralytics/issues/6596) - community DFL implementation notes

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in project, no new dependencies
- Architecture: HIGH - patterns directly observed from existing codebase code
- DFL implementation: HIGH - verified against GFL paper and MMDetection reference implementation
- ONNX compatibility: HIGH - verified that softmax, matmul, sum are standard ONNX ops in opset 17
- Pitfalls: MEDIUM - based on codebase analysis and DFL implementation experience from GFL/YOLOv8

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable domain, well-understood techniques)
