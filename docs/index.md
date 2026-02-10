# Object Detection Training

Training framework for object detection models (**RFDETR** and **YOLOX**) built with PyTorch Lightning, Hydra, and Pixi.

## Features

- **Multiple architectures** -- RFDETR (transformer-based) and YOLOX (anchor-free YOLO)
- **Hydra configuration** -- Hierarchical configs with command-line overrides
- **PyTorch Lightning** -- Scalable training with callbacks, logging, and checkpointing
- **Experiment tracking** -- Weights & Biases and TensorBoard integration
- **ONNX export** -- Automatic model export for deployment
- **Dataset statistics** -- Built-in analysis and visualization of detection datasets

## Quick Start

```bash
# Install dependencies
pixi install

# Run training with default config
pixi run train

# Override configuration
pixi run train -- model=yolox_s training.epochs=50
```

See the [Getting Started](getting-started.md) guide for detailed setup instructions.
