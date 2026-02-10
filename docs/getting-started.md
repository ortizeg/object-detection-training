# Getting Started

## Prerequisites

- [Pixi](https://pixi.sh) package manager
- CUDA 12.1 (Linux/Windows) or macOS with MPS support

## Installation

```bash
git clone https://github.com/ortizeg/object-detection-training.git
cd object-detection-training
./scripts/dev-install.sh
```

## First Training Run

```bash
pixi run train
```

This uses the default configuration at `src/object_detection_training/conf/train.yaml`, which trains an RFDETR-Small model.

### Override Configuration

Pass Hydra overrides after `--`:

```bash
# Change model
pixi run train -- models=yolox_s

# Change training parameters
pixi run train -- trainer.max_epochs=50 data.batch_size=16
```

## Running Tests

```bash
pixi run test
```

## Code Quality

```bash
pixi run format      # Auto-format
pixi run lint        # Lint
pixi run typecheck   # Type check
```
