FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install curl and other basics needed for pixi install
# libgl1 and libglib2.0-0 are often needed for OpenCV and other ML libraries
RUN apt-get update && apt-get install -y curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Add pixi to PATH
ENV PATH="/root/.pixi/bin:$PATH"

# WandB API Key
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY

# Override CUDA version for pixi to allow installation of cuda-dependent packages
# even if the build environment doesn't have a GPU attached.
ENV CONDA_OVERRIDE_CUDA=12.1

COPY pixi.toml pixi.lock* pyproject.toml ./

# Create a dummy project structure to allow pixi (and flit) to install dependencies
# without invalidating the cache when source code changes.
RUN mkdir -p src/object_detection_training && touch src/object_detection_training/__init__.py

# Install dependencies (only production environment)
RUN pixi install --environment prod

# Copy source code
COPY . .
# If we have a lot of code, we might want to be more selective, but this follows standard practice.
# .dockerignore should be used to exclude unrelated files.

# Set entrypoint to run the training task defined in pixi.toml
ENTRYPOINT ["pixi", "run", "train"]
