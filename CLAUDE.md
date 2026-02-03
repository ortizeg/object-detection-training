# Object Detection Training Project

## Project Overview

This is an object detection training framework built with PyTorch Lightning, Hydra, and specialized for training models like RFDETR and YOLOX. The project focuses on training object detectors with modern architectures and efficient configuration management.

**Author**: Enrique G. Ortiz (ortizeg@gmail.com)
**Version**: 0.0.1
**Python Version**: 3.11

## Key Technologies

- **PyTorch Lightning**: Training framework
- **Hydra**: Configuration management
- **Pixi**: Package and environment management (replaces conda/pip)
- **RFDETR**: RT-DETR based object detection model
- **YOLOX**: YOLO-based object detection model
- **Weights & Biases (wandb)**: Experiment tracking
- **TensorBoard**: Training visualization
- **ONNX**: Model export and deployment

## Project Structure

```
.
├── src/object_detection_training/    # Main source code
│   ├── models/                       # Model implementations
│   │   ├── yolox/                   # YOLOX model family
│   │   ├── rfdetr/                  # RFDETR model family
│   │   └── yolox_lightning.py       # Lightning module for YOLOX
│   ├── callbacks/                    # Lightning callbacks
│   │   ├── ema.py                   # Exponential Moving Average
│   │   ├── onnx_export.py          # ONNX export callback
│   │   ├── plotting.py             # Training plots
│   │   ├── visualization.py        # Visualizations
│   │   ├── model_info.py          # Model information
│   │   └── statistics.py          # Training statistics
│   ├── metrics/                      # Custom metrics
│   │   └── curves.py               # Precision-recall curves
│   ├── utils/                        # Utility functions
│   │   ├── boxes.py                # Bounding box utilities
│   │   ├── plotting.py            # Plotting utilities
│   │   ├── hydra.py              # Hydra configuration helpers
│   │   ├── seed.py               # Random seed management
│   │   └── json_utils.py         # JSON utilities
│   └── tasks.py                     # Task definitions
├── conf/                            # Hydra configuration files
│   ├── train.yaml                  # Main training config
│   ├── train_basketball_rfdetr.yaml # Basketball dataset RFDETR config
│   ├── train_yolox.yaml           # YOLOX training config
│   ├── models/                     # Model configurations
│   ├── data/                       # Dataset configurations
│   ├── callbacks/                  # Callback configurations
│   ├── task/                       # Task configurations
│   ├── logging/                    # Logging configurations
│   └── trainer/                    # Trainer configurations
├── tests/                           # Unit tests
├── scripts/                         # Helper scripts
│   └── dev-install.sh             # Development installation script
├── pixi.toml                       # Pixi project configuration
├── pixi.lock                       # Pixi lock file
├── pyproject.toml                  # Python project metadata
├── Dockerfile                      # Docker configuration
└── .pre-commit-config.yaml        # Pre-commit hooks

```

## Development Workflow

### Environment Setup

This project uses **Pixi** (not conda/pip) for all dependency management:

```bash
# Install dependencies
./scripts/dev-install.sh

# Or manually with pixi
pixi install
```

### Running Tasks

All tasks are managed through Pixi:

```bash
# Training
pixi run train                              # Default training
pixi run train -- training.epochs=50        # Override Hydra config

# Testing
pixi run test                               # Run pytest

# Code Quality
pixi run format                             # Format with ruff
pixi run lint                               # Lint with ruff
```

### Configuration with Hydra

The project uses Hydra for hierarchical configuration management. All configs are in `conf/`:

- Modify `conf/train.yaml` for general training settings
- Override from command line: `pixi run train -- model.learning_rate=0.001`
- Use config groups for different models, datasets, etc.

## Key Components

### Models

1. **YOLOX** (`src/object_detection_training/models/yolox/`)
   - Anchor-free YOLO variant
   - Lightning module wrapper in `yolox_lightning.py`

2. **RFDETR** (`src/object_detection_training/models/rfdetr/`)
   - RT-DETR based transformer detector
   - Specialized for real-time detection

### Callbacks

Located in `src/object_detection_training/callbacks/`:
- **EMA**: Exponential moving average for model weights
- **ONNX Export**: Automatic model export to ONNX format
- **Plotting**: Training curves and metrics visualization
- **Visualization**: Detection result visualization
- **Model Info**: Model architecture and parameter statistics

### Experiment Tracking

- **Weights & Biases**: Configure via `conf/logging/`
- **TensorBoard**: Built-in Lightning integration

## Code Style & Quality

- **Formatter**: Ruff (Black-compatible)
- **Linter**: Ruff (multi-rule linting)
- **Type Checking**: MyPy (strict mode, with exceptions for third-party model code)
- **Line Length**: 88 characters
- **Python Version**: 3.11

### Import Organization
```python
# Standard library
import os

# Third-party
import torch
from lightning import LightningModule

# First-party (this project)
from object_detection_training.utils.boxes import box_iou
```

## Important Notes for AI Assistants

### When Making Changes

1. **Always use Pixi**: Don't suggest `pip install` or `conda`. Use `pixi add <package>`
2. **Respect Hydra configs**: Changes to training behavior should go in YAML configs, not hardcoded
3. **Type annotations**: This project uses strict typing (MyPy). Always add type hints
4. **Format code**: Run `pixi run format` before committing
5. **Tests**: Add tests in `tests/` for new functionality

### Model Code Exceptions

Files in these directories have relaxed type checking:
- `src/object_detection_training/models/yolox/`
- `src/object_detection_training/models/rfdetr/`

These are third-party model implementations with existing code style.

### Dependencies

Key constraints:
- `numpy<2.0.0`: Compatibility with existing models
- `pydantic>=2.0`: Modern Pydantic API
- `onnxscript>=0.5.6,<0.6`: Specific ONNX version requirements

### CUDA Support

- Linux/Windows: CUDA 12.1
- macOS: MPS (Metal Performance Shaders)

## Common Patterns

### Adding a New Model

1. Create model class in `src/object_detection_training/models/`
2. Create Lightning module wrapper if needed
3. Add Hydra config in `conf/models/`
4. Update task configuration

### Adding a New Dataset

1. Create dataset class (use PyTorch Dataset API)
2. Add Hydra config in `conf/data/`
3. Update data module configuration

### Adding a Callback

1. Implement callback in `src/object_detection_training/callbacks/`
2. Add config in `conf/callbacks/`
3. Register in training config

## Entry Points

- **Main training script**: `src/object_detection_training/tasks.py`
- **CLI command**: `train-detect` (installed via `pip install -e .`)

## Output Structure

Training outputs go to `outputs/` (Hydra-managed):
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/              # Hydra config snapshots
        ├── checkpoints/         # Model checkpoints
        └── logs/               # Training logs
```
