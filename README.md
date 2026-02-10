[![Test Suite](https://github.com/ortizeg/object-detection-training/actions/workflows/test.yml/badge.svg)](https://github.com/ortizeg/object-detection-training/actions/workflows/test.yml)
[![Lint & Format](https://github.com/ortizeg/object-detection-training/actions/workflows/lint.yml/badge.svg)](https://github.com/ortizeg/object-detection-training/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ortizeg/object-detection-training/blob/main/LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

# Object Detection Training

Training framework for object detection models (RFDETR and YOLOX) built with PyTorch Lightning, Hydra, and Pixi.

## Prerequisites

- **Pixi**: A package management tool. Install it from [pixi.sh](https://pixi.sh).
- **CUDA 12.1**: Required for GPU acceleration (linux/windows) or appropriate drivers for macOS (MPS).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ortizeg/object-detection-training.git
    cd object-detection-training
    ```

2.  **Install dependencies:**
    Pixi will automatically handle environment setup and dependency installation when you run any task.
    ```bash
    ./scripts/dev-install.sh
    ```

## Usage

This project uses `pixi` to manage tasks.

### Training

To start the training process:

```bash
pixi run train
```
This runs `src/object_detection_training/task_manager.py`.
You can customize the training configuration by modifying `src/object_detection_training/conf/train.yaml` or passing overrides to Hydra:
```bash
pixi run train -- training.epochs=50
```

### Testing

To run the unit tests:

```bash
pixi run test
```
This executes `pytest`.

### Code Quality

To format the code using `ruff`:
```bash
pixi run format
```

To lint the code using `ruff`:
```bash
pixi run lint
```

To type-check using `mypy`:
```bash
pixi run typecheck
```

## Docker Build

The project includes Docker support for running training jobs on GCP Vertex AI.

### Cloud Build (Recommended)

Cloud Build runs the Docker build on GCP infrastructure, which is much faster than building locally on ARM Macs (no cross-compilation, no large image upload over home internet).

```bash
# Submit build to Cloud Build
./scripts/cloud-build.sh
# or
pixi run build

# Preview the command without submitting
./scripts/cloud-build.sh --dry-run
```

**What happens:**
1. Your code is uploaded as a ~50MB tarball to Cloud Build
2. Cloud Build pulls the previous `:latest` image for layer caching
3. Builds the AMD64 image natively (2-5 min with cache)
4. Pushes to Artifact Registry with both `:SHORT_SHA` and `:latest` tags

**Prerequisites:**
- `gcloud` CLI configured with your project: `gcloud config set project <PROJECT_ID>`
- Cloud Build API enabled: `gcloud services enable cloudbuild.googleapis.com`
- Artifact Registry repository exists (created separately)

### Local Build

For testing Docker builds locally without pushing:

```bash
./scripts/build-docker.sh --local
# or
pixi run build-local
```

To build and push from your local machine (slower on ARM Macs):

```bash
./scripts/build-docker.sh
```

### Runtime Secrets

The Docker image does not contain secrets. `WANDB_API_KEY` and other credentials must be injected at runtime via environment variables (e.g., in Vertex AI job specs).

## Project Structure

```
.
├── src/object_detection_training/    # Main source code
│   ├── models/                       # Model implementations
│   │   ├── yolox/                   # YOLOX model family
│   │   ├── rfdetr/                  # RFDETR model family
│   │   ├── yolox_lightning.py       # Lightning module for YOLOX
│   │   └── rfdetr_lightning.py      # Lightning module for RFDETR
│   ├── callbacks/                    # Lightning callbacks
│   │   ├── ema.py                   # Exponential Moving Average
│   │   ├── onnx_export.py          # ONNX export callback
│   │   ├── plotting.py             # Training plots
│   │   ├── visualization.py        # Visualizations
│   │   ├── model_info.py          # Model information
│   │   └── statistics.py          # Training statistics
│   ├── data/                         # Data modules and datasets
│   │   ├── detection_dataset.py    # Base detection dataset
│   │   ├── dataset_stats.py        # Dataset statistics
│   │   └── base.py                 # Base data module
│   ├── metrics/                      # Custom metrics
│   │   └── curves.py               # Precision-recall curves
│   ├── utils/                        # Utility functions
│   │   ├── boxes.py                # Bounding box utilities
│   │   ├── plotting.py            # Plotting utilities
│   │   ├── hydra.py              # Hydra configuration helpers
│   │   ├── seed.py               # Random seed management
│   │   └── json_utils.py         # JSON utilities
│   ├── conf/                        # Hydra configuration files
│   │   ├── train.yaml              # Main training config
│   │   ├── models/                 # Model configurations
│   │   ├── data/                   # Dataset configurations
│   │   ├── callbacks/              # Callback configurations
│   │   └── trainer/                # Trainer configurations
│   ├── tasks.py                     # Task definitions
│   ├── task_manager.py              # CLI entry point
│   └── types.py                     # Shared type definitions
├── tests/                           # Unit tests
├── docs/                            # Documentation (MkDocs)
├── scripts/                         # Helper scripts
├── pixi.toml                       # Pixi project configuration
├── pyproject.toml                  # Python project metadata
├── Dockerfile                      # Docker configuration
└── .pre-commit-config.yaml        # Pre-commit hooks
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
