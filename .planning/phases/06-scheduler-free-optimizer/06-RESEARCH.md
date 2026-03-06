# Phase 6: Scheduler-Free Optimizer - Research

**Researched:** 2026-03-06
**Domain:** Schedule-free optimization with PyTorch Lightning integration
**Confidence:** HIGH

## Summary

Phase 6 integrates Facebook Research's `schedulefree` library (specifically `AdamWScheduleFree`) as an alternative optimizer option in the DINOXLightningModel. The core challenge is not the optimizer itself (it is a well-tested drop-in `torch.optim.Optimizer` subclass) but the **train/eval mode switching** required by the schedule-free approach: the optimizer maintains two parameter sequences (y for gradient evaluation, x for loss evaluation) and must be explicitly told which mode it is in. PyTorch Lightning does not natively handle this, so we need to add lifecycle hooks.

The existing codebase already has the `use_scheduler_free` flag in `DINOXConfig` (defaulting to `False`) and the `configure_optimizers()` method in `DINOXLightningModel` currently creates SGD with warmup+cosine schedule. The integration requires: (1) conditionally constructing `AdamWScheduleFree` instead of SGD when the flag is set, (2) adding Lightning hooks for train/eval mode switching, (3) handling checkpoint saving correctly, and (4) returning no LR scheduler when using schedule-free (the optimizer handles warmup internally via its `warmup_steps` parameter).

**Primary recommendation:** Use `AdamWScheduleFree` from `schedulefree>=1.4` with Lightning hooks `on_train_epoch_start`, `on_validation_model_eval`, and `on_validation_model_train` for mode switching. Use a custom callback or `on_save_checkpoint` for checkpoint safety.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| schedulefree | >=1.4,<2.0 | Schedule-free AdamW optimizer | Official Facebook Research implementation; 1.4 adds RAdam, 1.3 fixed weight decay during warmup |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| lightning | (existing) | Training framework | Already in project; provides the hooks we need |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| AdamWScheduleFree | RAdamScheduleFree | RAdam eliminates warmup entirely; AdamW is better understood and the phase spec says "AdamW" |
| schedulefree pip package | Copy optimizer code | Don't -- the library is small but actively maintained with bug fixes |

**Installation:**
```bash
/Users/ortizeg/.pixi/bin/pixi add schedulefree
```

## Architecture Patterns

### Integration Point: configure_optimizers()

The existing `DINOXLightningModel.configure_optimizers()` (lines 578-648 in `dinox_lightning.py`) creates SGD + warmup + cosine schedule. The scheduler-free path must:

1. Build param groups identically (pg0/pg1/pg2 with same weight decay rules)
2. Construct `AdamWScheduleFree` instead of SGD
3. Return `{"optimizer": optimizer}` with **no lr_scheduler key** (or an empty/dummy one)

**Key insight:** The `OptimizerConfig` TypedDict in `types.py` currently requires `lr_scheduler`. This needs to be made optional (use `total=False` or a Union type) or the scheduler-free path returns a dummy scheduler.

### Pattern 1: Optimizer Mode Switching via Lightning Hooks

**What:** Schedule-free optimizers maintain two parameter buffers. `optimizer.train()` exposes the gradient-evaluation parameters; `optimizer.eval()` exposes the evaluation/checkpoint parameters.

**When to use:** Whenever the model switches between training and evaluation.

**Example:**
```python
# Source: https://github.com/Lightning-AI/pytorch-lightning/discussions/19759
class DINOXLightningModel(BaseDetectionModel):

    def on_train_epoch_start(self) -> None:
        """Handle epoch-based schedule changes + scheduler-free mode."""
        super().on_train_epoch_start()  # existing logic
        if self._use_scheduler_free:
            for opt in self.trainer.optimizers:
                if hasattr(opt, "train"):
                    opt.train()

    def on_validation_model_eval(self) -> None:
        """Switch optimizer to eval mode for validation."""
        super().on_validation_model_eval()
        if self._use_scheduler_free:
            for opt in self.trainer.optimizers:
                if hasattr(opt, "eval"):
                    opt.eval()

    def on_validation_model_train(self) -> None:
        """Switch optimizer back to train mode after validation."""
        super().on_validation_model_train()
        if self._use_scheduler_free:
            for opt in self.trainer.optimizers:
                if hasattr(opt, "train"):
                    opt.train()
```

### Pattern 2: Checkpoint Safety

**What:** Optimizer must be in eval mode when saving checkpoints so that the "x" (evaluation) parameters are stored, not the "y" (gradient) parameters.

**When to use:** When `use_scheduler_free=True` and model checkpoints are being saved.

**Example:**
```python
# Source: https://github.com/Lightning-AI/pytorch-lightning/discussions/19759
def on_save_checkpoint(self, checkpoint: dict) -> None:
    """Ensure scheduler-free optimizer is in eval mode for checkpoint."""
    if self._use_scheduler_free:
        for opt in self.trainer.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()
```

**Important caveat:** Lightning reads optimizer state *before* `on_save_checkpoint` executes. A safer approach is a custom `Callback` that overrides `on_save_checkpoint` on the Trainer level, or a subclassed `ModelCheckpoint`. However, for this project the simplest approach is to override `on_save_checkpoint` in the LightningModule -- this works because Lightning calls `pl_module.on_save_checkpoint(checkpoint)` after populating the checkpoint dict but the optimizer `.eval()` call modifies parameters in-place.

### Pattern 3: OptimizerConfig Return Type

**What:** When using scheduler-free, no LR scheduler is needed (warmup is handled by `warmup_steps` parameter).

**Example:**
```python
def configure_optimizers(self):
    if self._use_scheduler_free:
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(
            pg0,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,  # pg0 has no decay
            warmup_steps=warmup_steps,
        )
        optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})
        optimizer.add_param_group({"params": pg2})
        return {"optimizer": optimizer}
    else:
        # existing SGD + scheduler code
        ...
```

### Recommended File Changes

```
src/object_detection_training/
  models/
    dinox_lightning.py       # Modify: configure_optimizers(), add hooks
    dinox/config.py          # Already has use_scheduler_free flag
  types.py                   # Modify: make lr_scheduler optional in OptimizerConfig
  callbacks/
    scheduler_free.py        # NEW (optional): callback for checkpoint safety
tests/
  test_dinox_scheduler_free.py  # NEW: test optimizer creation, hook behavior
src/object_detection_training/conf/models/
  dinox_m_e6.yaml            # NEW: experiment config with scheduler-free enabled
```

### Anti-Patterns to Avoid
- **Using on_train_epoch_start/on_validation_epoch_start for mode switching:** Use `on_validation_model_eval` and `on_validation_model_train` instead -- these fire at the correct time relative to `model.train()`/`model.eval()` calls.
- **Adding a dummy LR scheduler:** Don't create a no-op scheduler just to satisfy the TypedDict. Fix the type to allow optimizer-only returns.
- **Wrapping the optimizer in ScheduleFreeWrapper:** Use `AdamWScheduleFree` directly. The wrapper is marked "experimental" in the library.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schedule-free optimization | Custom warmup-only optimizer | `schedulefree.AdamWScheduleFree` | Implements the paper's algorithm correctly with foreach optimization |
| Built-in warmup | Manual LR warmup logic | `AdamWScheduleFree(warmup_steps=N)` | The optimizer handles warmup internally; external schedulers conflict |

**Key insight:** The whole point of schedule-free is eliminating the LR schedule. Do not add a scheduler on top of the schedule-free optimizer.

## Common Pitfalls

### Pitfall 1: Forgetting optimizer.train()/eval() Calls
**What goes wrong:** Model evaluates/saves with gradient-evaluation parameters (y) instead of the proper evaluation parameters (x), leading to degraded validation metrics and corrupt checkpoints.
**Why it happens:** Standard PyTorch optimizers don't have train/eval modes, so it's easy to forget.
**How to avoid:** Add the hooks described above. Test by verifying that validation loss is computed with optimizer in eval mode.
**Warning signs:** Validation metrics are significantly worse than expected; checkpoint loading produces different results than end-of-training evaluation.

### Pitfall 2: Conflicting LR Schedulers
**What goes wrong:** Adding an external LR scheduler on top of AdamWScheduleFree causes the learning rate to be modified twice, leading to training instability.
**Why it happens:** The existing code always creates a scheduler in configure_optimizers().
**How to avoid:** When `use_scheduler_free=True`, return only `{"optimizer": optimizer}` with no `lr_scheduler` key.
**Warning signs:** Unexpected learning rate curves in W&B/TensorBoard logs.

### Pitfall 3: OptimizerConfig TypedDict Mismatch
**What goes wrong:** Returning `{"optimizer": optimizer}` without `lr_scheduler` fails mypy because `OptimizerConfig` requires it.
**Why it happens:** The TypedDict was defined with `lr_scheduler` as a required key.
**How to avoid:** Either (a) change `OptimizerConfig` to use `total=False` for `lr_scheduler`, or (b) create a separate `OptimizerOnlyConfig` TypedDict, or (c) use a Union return type. Option (a) is simplest.
**Warning signs:** mypy errors on the configure_optimizers return.

### Pitfall 4: BatchNorm Statistics Misalignment
**What goes wrong:** When the optimizer switches to eval mode, the model's BatchNorm layers may have running statistics computed at the "y" point rather than the "x" point.
**Why it happens:** Schedule-free optimizers update parameters at a different point than standard optimizers; BN running stats track the wrong point.
**How to avoid:** For this project, this is LOW risk because (a) Lightning calls `model.eval()` which sets BN to use running stats, and (b) the schedule-free README suggests this matters most for very long training. Monitor validation metrics. If there's a gap, add a BN statistics refresh step before validation.
**Warning signs:** Validation loss/mAP is consistently worse than training metrics suggest.

### Pitfall 5: EMA Callback Interaction
**What goes wrong:** The project has an `EMACallback` that swaps model parameters during validation. If both EMA and scheduler-free optimizer are active, the parameter-swapping sequences could conflict.
**Why it happens:** Both EMA and scheduler-free maintain shadow parameters and need to swap them at train/eval boundaries.
**How to avoid:** Document that EMA and scheduler-free should not be used simultaneously, OR ensure the hook ordering is correct (optimizer.eval() before EMA swap). For Phase 6, recommend disabling EMA when using scheduler-free.
**Warning signs:** Parameter values oscillating between steps; NaN losses.

### Pitfall 6: Parameter Group Weight Decay
**What goes wrong:** `AdamWScheduleFree` applies weight decay differently than SGD. Passing `weight_decay` to the constructor applies it to ALL parameter groups.
**Why it happens:** The existing code uses `add_param_group` with per-group weight_decay.
**How to avoid:** Set `weight_decay=0` in the constructor (for pg0 which has no decay), then set per-group weight_decay when adding pg1 and pg2 via `add_param_group`.
**Warning signs:** Unexpected parameter norm drift.

## Code Examples

### Creating AdamWScheduleFree with Parameter Groups
```python
# Source: https://github.com/facebookresearch/schedule_free (README + source)
from schedulefree import AdamWScheduleFree

# Parameter groups (same as existing SGD setup)
pg0: list[nn.Parameter] = []  # BN weights - no decay
pg1: list[nn.Parameter] = []  # Other weights - with decay
pg2: list[nn.Parameter] = []  # Biases - no decay

for k, v in self.model.named_modules():
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight)
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)

# Calculate warmup steps
warmup_steps = int(total_steps * (self.warmup_epochs / max(1, max_epochs)))
warmup_steps = max(1, warmup_steps)

optimizer = AdamWScheduleFree(
    pg0,
    lr=self.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,       # pg0: BN weights, no decay
    warmup_steps=warmup_steps,
)
optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})
optimizer.add_param_group({"params": pg2, "weight_decay": 0.0})
```

### AdamWScheduleFree Constructor Signature
```python
# Source: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py
class AdamWScheduleFree(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.0025,                    # default LR (higher than standard AdamW)
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,               # built-in linear warmup
        r=0.0,                        # polynomial weighting power
        weight_lr_power=2.0,
        foreach=None,                 # auto-detected
    ):
```

### Lightning Hook Integration
```python
# Source: https://github.com/Lightning-AI/pytorch-lightning/discussions/19759
# Verified pattern for schedule-free optimizer mode switching

def on_train_epoch_start(self) -> None:
    # ... existing epoch logic ...
    if self._use_scheduler_free:
        for opt in self.trainer.optimizers:
            if hasattr(opt, "train"):
                opt.train()

def on_validation_model_eval(self) -> None:
    super().on_validation_model_eval()
    if self._use_scheduler_free:
        for opt in self.trainer.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()

def on_validation_model_train(self) -> None:
    super().on_validation_model_train()
    if self._use_scheduler_free:
        for opt in self.trainer.optimizers:
            if hasattr(opt, "train"):
                opt.train()

def on_save_checkpoint(self, checkpoint: dict) -> None:
    if self._use_scheduler_free:
        for opt in self.trainer.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()
```

### Test Pattern: Verify Optimizer Creation
```python
# Pattern matching existing test_dinox_config.py style
import pytest
from schedulefree import AdamWScheduleFree

def test_scheduler_free_optimizer_created():
    """AdamWScheduleFree is created when use_scheduler_free=True."""
    model = DINOXLightningModel(
        num_classes=2,
        download_pretrained=False,
        use_scheduler_free=True,
    )
    # Need trainer context for configure_optimizers
    trainer = L.Trainer(max_epochs=10, fast_dev_run=True)
    trainer.model = model
    config = model.configure_optimizers()
    assert isinstance(config["optimizer"], AdamWScheduleFree)
    assert "lr_scheduler" not in config

def test_scheduler_free_optimizer_has_warmup():
    """AdamWScheduleFree uses internal warmup, not external scheduler."""
    model = DINOXLightningModel(
        num_classes=2,
        download_pretrained=False,
        use_scheduler_free=True,
        warmup_epochs=5,
    )
    # ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual LR schedule tuning (cosine, step, etc.) | Schedule-free optimizer (no schedule needed) | 2024 (paper + library) | Eliminates one major hyperparameter axis |
| Separate warmup scheduler | Built-in `warmup_steps` parameter | schedulefree 1.0+ | Warmup is part of the optimizer, not a separate scheduler |
| schedulefree 1.2 weight decay during warmup | Fixed in 1.3 to match PyTorch AdamW | 2024 | Weight decay now consistent with standard AdamW |

**Deprecated/outdated:**
- `ScheduleFreeWrapper` (experimental): Use `AdamWScheduleFree` directly instead
- schedulefree <1.3: Had inconsistent weight decay during warmup

## Open Questions

1. **Learning rate scaling for AdamW vs SGD**
   - What we know: The schedulefree README suggests AdamW schedule-free may need 1x-10x larger LR than schedule-based AdamW. The current LR (1e-3) is set for SGD.
   - What's unclear: What is the right LR for AdamWScheduleFree with this specific model architecture?
   - Recommendation: Start with the default 0.0025 from schedulefree, or use the existing 1e-3. The smoke test (OPT-03) will validate convergence.

2. **EMA + Schedule-Free interaction**
   - What we know: Both maintain shadow parameters. Both swap at train/eval boundaries.
   - What's unclear: Whether they can coexist safely with correct hook ordering.
   - Recommendation: For Phase 6, document that EMA should be disabled when using schedule-free. Investigate interaction in a future phase if needed.

3. **OptimizerConfig return type for no-scheduler case**
   - What we know: Lightning `configure_optimizers()` can return just an optimizer (no dict needed), or a dict with only `"optimizer"` key. The project's TypedDict requires `lr_scheduler`.
   - What's unclear: Whether changing the TypedDict will break other code paths.
   - Recommendation: Make `lr_scheduler` optional in the TypedDict (`total=False` on `LRSchedulerConfig` fields, or create a Union). This is a safe change since Lightning accepts both forms.

## Sources

### Primary (HIGH confidence)
- [facebookresearch/schedule_free GitHub](https://github.com/facebookresearch/schedule_free) - README, optimizer API, train/eval mode requirements
- [PyPI schedulefree 1.4.1](https://pypi.org/project/schedulefree/) - Latest version (1.4.1, released 2024-03-24), Python >=3.4
- [AdamWScheduleFree source](https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py) - Constructor signature and defaults

### Secondary (MEDIUM confidence)
- [Lightning Discussion #19759](https://github.com/Lightning-AI/pytorch-lightning/discussions/19759) - Community-verified hook patterns for schedule-free integration with Lightning
- [Lightning Optimization docs](https://lightning.ai/docs/pytorch/stable/common/optimization.html) - configure_optimizers return type flexibility

### Tertiary (LOW confidence)
- BatchNorm statistics refresh recommendation (from README, but impact is use-case dependent)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - schedulefree is the only maintained implementation of the paper; API verified from source
- Architecture: HIGH - Hook patterns verified from Lightning discussion with multiple confirmations; existing codebase structure is clear
- Pitfalls: HIGH - train/eval mode switching is well-documented as the primary concern; TypedDict issue verified from source code

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable library, slow-moving API)
