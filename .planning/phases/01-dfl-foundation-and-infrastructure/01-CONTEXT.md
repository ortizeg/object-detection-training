# Phase 1: DFL Foundation and Infrastructure - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

New DINOXHead with DFL box regression that trains end-to-end through the existing pipeline, exports to ONNX with the same output contract, and all improvement flags off by default reproducing standard YOLOX-M behavior. Includes DINOXConfig Pydantic model, Hydra configs for Phase 1 ablations, DINOXLightningModel, and EMA callback compatibility.

</domain>

<decisions>
## Implementation Decisions

### DINOXHead Design
- Claude decides whether to subclass YOLOXHead or create a fresh class that reuses internals — pick the approach that best fits the codebase
- When DFL is disabled (use_dfl=False), DINOXHead must be functionally equivalent to YOLOXHead (same architecture and behavior, small numerical differences OK — not required to be bit-identical)
- Must support loading pretrained YOLOX-M checkpoint weights where architecturally compatible (backbone, neck, shared conv layers)
- DINOXLightningModel goes in a new file: `dinox_lightning.py` — clean separation from existing yolox_lightning.py

### Config & Toggling Strategy
- Claude decides how DINOXConfig integrates with Hydra (nested under model config vs separate config group) — follow existing codebase patterns
- Strict Pydantic validation for invalid flag combinations (e.g., use_mal=True without use_soft_labels=True) — fail fast with clear errors
- Only create Phase 1 Hydra configs now (dinox_baseline.yaml, dinox_dfl.yaml) — add others as phases implement their components
- Claude decides whether to enforce blessed configs only or allow arbitrary flag combinations via CLI overrides

### ONNX Export Compatibility
- Claude determines the exact ONNX output format by reading the existing export and eval code
- DFL distribution-to-point conversion MUST be baked into the ONNX graph (softmax + weighted sum inside the model) — inference code stays unchanged
- Claude decides whether to include an automated ONNX round-trip test in CI or as a manual script
- ONNX export must support dynamic batch size

### Existing Code Boundaries
- Models/yolox/ files: minor interface additions OK (new methods, making internal methods public) but no behavior changes
- Shared utilities (utils/boxes.py, callbacks/onnx_export.py): can add new functions/methods but don't modify existing signatures or behavior
- All existing YOLOX/RFDETR tests must continue passing unchanged — DINO-X is purely additive
- Modify task_manager.py to route DINO-X configs — single `pixi run train` entry point for all models

### Claude's Discretion
- DINOXHead inheritance/composition strategy
- Hydra config group structure (nested vs separate)
- Blessed configs vs arbitrary flag combinations
- ONNX round-trip test location (CI vs manual script)
- Exact ONNX output format (determined by reading existing code)

</decisions>

<specifics>
## Specific Ideas

- DFL distribution→point conversion baked into ONNX graph so downstream inference pipeline is completely unchanged
- Weight loading from YOLOX-M checkpoints is required for transfer learning / faster convergence
- PRD specifies reg_max=16 (17 bins per edge) as standard value from GFL paper

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-dfl-foundation-and-infrastructure*
*Context gathered: 2026-03-05*
