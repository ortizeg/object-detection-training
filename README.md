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

To format the code using `black`:
```bash
pixi run format
```

To lint the code using `ruff`:
```bash
pixi run lint
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

- `src/`: Source code for the project.
- `src/object_detection_training/conf/`: Hydra configuration files.
- `tests/`: Unit tests.
- `pixi.toml`: Project configuration and dependencies.
- `scripts/`: Helper scripts.
- `outputs/`: Training outputs and logs.
