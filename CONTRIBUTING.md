# Contributing

Thank you for your interest in contributing to the Object Detection Training project!

## Development Environment Setup

This project uses [Pixi](https://pixi.sh) for dependency management. **Do not use pip or conda directly.**

```bash
# Clone the repository
git clone https://github.com/ortizeg/object-detection-training.git
cd object-detection-training

# Install all dependencies (including dev tools)
./scripts/dev-install.sh

# Or manually
pixi install
```

## Code Quality Standards

### Formatting

We use **Ruff** (Black-compatible) for formatting with a line length of 88 characters:

```bash
pixi run format        # Auto-format all files
pixi run format-check  # Check formatting without modifying
```

### Linting

Ruff is configured with multiple rule sets (pycodestyle, pyflakes, isort, bugbear, bandit, etc.):

```bash
pixi run lint
```

### Type Checking

MyPy is configured in strict mode. All new code must include type annotations:

```bash
pixi run typecheck
```

**Exception:** Files under `src/object_detection_training/models/yolox/` and `src/object_detection_training/models/rfdetr/` have relaxed type checking since they contain third-party model implementations.

## Running Tests

```bash
pixi run test          # Run all tests
pixi run test-cov      # Run with coverage report
```

Tests should:
- Not require a GPU (use CPU tensors)
- Not make network calls (mock external services)
- Use `matplotlib.use("Agg")` for any plotting tests
- Follow existing style: `from __future__ import annotations`

## Pre-commit Hooks

Set up pre-commit hooks to catch issues before they reach CI:

```bash
pixi run precommit     # Run all hooks on all files
```

Hooks include trailing whitespace removal, end-of-file fixing, YAML validation, large file checks, and Ruff formatting/linting.

## Git Workflow

1. **Branch from `develop`** for new features or fixes
2. Use descriptive branch names: `feat/add-new-model`, `fix/ema-state-dict`
3. Write clear commit messages following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `ci:` for CI changes
   - `build:` for build system changes
4. Open a PR targeting `develop`
5. Ensure all CI checks pass before requesting review

## Adding New Components

### Adding a New Model

1. Create model class in `src/object_detection_training/models/`
2. Create a Lightning module wrapper (extend `BaseDetectionModel`)
3. Add Hydra config in `src/object_detection_training/conf/models/`
4. Use `@register` decorator to register with ConfigStore
5. Add tests in `tests/`

### Adding a New Dataset

1. Create a dataset class extending `DetectionDataset` in `src/object_detection_training/data/`
2. Implement `load_annotations()` and `_load_image()` abstract methods
3. Add Hydra config in `src/object_detection_training/conf/data/`
4. Add tests in `tests/`

### Adding a Callback

1. Implement callback in `src/object_detection_training/callbacks/`
2. Add config in `src/object_detection_training/conf/callbacks/`
3. Register in the main training config
4. Add tests in `tests/`

## Docker

### Cloud Build (Recommended)

```bash
pixi run build
```

### Local Build

```bash
pixi run build-local
```

See the [README](README.md#docker-build) for full Docker documentation.
