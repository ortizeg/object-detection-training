[![Test Suite](https://github.com/ortizeg/object-detection-training/actions/workflows/test.yml/badge.svg)](https://github.com/ortizeg/object-detection-training/actions/workflows/test.yml)
[![Lint & Format](https://github.com/ortizeg/object-detection-training/actions/workflows/lint.yml/badge.svg)](https://github.com/ortizeg/object-detection-training/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ortizeg/object-detection-training/blob/main/LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

# Object Detection Training

End-to-end object detection framework for training, exporting, and annotating with modern architectures and vision-language models.

**Train** detectors (RFDETR, YOLOX) with PyTorch Lightning and Hydra config management. **Export** to ONNX for production inference. **Annotate** datasets at scale using VLM-powered labeling with Google Gemini.

## Features

- **Multi-architecture training** &mdash; RFDETR (transformer-based) and YOLOX (anchor-free) with a unified Lightning training loop
- **Hydra configuration** &mdash; hierarchical YAML configs with full CLI override support for models, datasets, callbacks, and trainers
- **ONNX export & inference** &mdash; one-command checkpoint-to-ONNX conversion with built-in inference pipeline and pluggable post-processors
- **VLM annotation** &mdash; generate detection annotations from images using Gemini, with structured JSON output, retry logic, and configurable class taxonomies
- **Experiment tracking** &mdash; Weights & Biases and TensorBoard integration
- **Production-ready** &mdash; Docker builds for GCP Vertex AI, pre-commit hooks, strict MyPy typing, 280+ unit tests

## Quick Start

### Prerequisites

- [Pixi](https://pixi.sh) for environment and dependency management
- CUDA 12.1 (Linux/Windows) or MPS (macOS) for GPU acceleration

### Install & Run

```bash
git clone https://github.com/ortizeg/object-detection-training.git
cd object-detection-training
./scripts/dev-install.sh

# Train a detector
pixi run train

# Export to ONNX
pixi run task_manager task=onnx_export task.checkpoint_path=model.ckpt

# Run VLM annotation with Gemini
export GOOGLE_API_KEY=<your-key>
pixi run task_manager task=basketball_gemini task.image_dir=/path/to/images

# Override any config from CLI
pixi run train -- training.epochs=50 model.learning_rate=0.001
```

### Code Quality

```bash
pixi run format       # Ruff formatter
pixi run lint         # Ruff linter
pixi run typecheck    # MyPy strict mode
pixi run test         # Pytest suite
```

## Architecture

```
src/object_detection_training/
  models/          RFDETR and YOLOX with Lightning module wrappers
  inference/       ONNX runtime and Gemini VLM inference engines
  callbacks/       EMA, ONNX export, plotting, visualization
  data/            COCO data modules with SQLite-cached datasets
  tasks/           Training, ONNX export, ONNX inference, VLM annotation
  conf/            Hydra YAML configs (models, data, callbacks, tasks)
  schemas/         Pydantic models for detections and annotations
```

## Docker

```bash
pixi run build         # Cloud Build on GCP (fast, cached)
pixi run build-local   # Local Docker build
```

The image targets GCP Vertex AI. Secrets (`WANDB_API_KEY`, etc.) are injected at runtime via environment variables.

## License

Apache License 2.0 &mdash; see [LICENSE](LICENSE).
