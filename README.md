# Object Detection Training

This project is designed for training an object detector using `rfnet` and `hydra`. It leverages `pixi` for dependency management and environment handling.

## Prerequisites

- **Pixi**: A package management tool. Install it from [pixi.sh](https://pixi.sh).
- **CUDA 12.1**: Required for GPU acceleration (linux/windows) or appropriate drivers for macOS (MPS).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
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
This runs `src/object_detection_training/train.py`.
You can customize the training configuration by modifying `conf/config.yaml` or passing overrides to Hydra:
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

To format the code using `black`:
```bash
pixi run format
```

To lint the code using `flake8`:
```bash
pixi run lint
```

## Project Structure

- `src/`: Source code for the project.
- `conf/`: Hydra configuration files.
- `tests/`: Unit tests.
- `pixi.toml`: Project configuration and dependencies.
- `scripts/`: Helper scripts.
- `outputs/`: Training outputs and logs.
