"""Shared type definitions for the object detection training framework.

Provides TypedDicts and type aliases used across models, callbacks, data,
and utilities.  Import from here instead of re-defining inline.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import torch


# ---------------------------------------------------------------------------
# Detection target dict (labels, boxes, etc.)
# ---------------------------------------------------------------------------
class DetectionTarget(TypedDict, total=False):
    """Target dict produced by detection datasets and consumed by models."""

    boxes: torch.Tensor  # [N, 4]
    labels: torch.Tensor  # [N]
    image_id: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor
    orig_size: torch.Tensor
    size: torch.Tensor


# ---------------------------------------------------------------------------
# Batch type returned by collate functions / consumed by *_step methods
# ---------------------------------------------------------------------------
DetectionBatch = tuple[torch.Tensor, list[DetectionTarget]]

# ---------------------------------------------------------------------------
# Model forward outputs (loss dicts, prediction dicts, etc.)
# ---------------------------------------------------------------------------
ModelOutputs = dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Prediction dict returned by get_predictions()
# ---------------------------------------------------------------------------
class DetectionPrediction(TypedDict):
    """Per-image prediction dict."""

    boxes: torch.Tensor  # [N, 4] xyxy
    scores: torch.Tensor  # [N]
    labels: torch.Tensor  # [N]


# ---------------------------------------------------------------------------
# Visualization sample collected by VisualizationCallback
# ---------------------------------------------------------------------------
class VisualizationSample(TypedDict):
    image: torch.Tensor
    target: DetectionTarget
    image_id: int


# ---------------------------------------------------------------------------
# Model statistics dict
# ---------------------------------------------------------------------------
class ModelStats(TypedDict, total=False):
    total_params: int
    trainable_params: int
    flops: int
    model_size_mb: float
    inference_time_ms: float
    fps: float
    input_shape: list[int]
    model_class: str
    num_classes: int | None


# ---------------------------------------------------------------------------
# Callback state dicts
# ---------------------------------------------------------------------------
class EMAState(TypedDict, total=False):
    ema_state_dict: dict[str, torch.Tensor]
    step_count: int
    decay: float


class ModelInfoState(TypedDict):
    model_info: dict[str, int | float | str | list[int] | None]


class ONNXExportState(TypedDict):
    exported_checkpoints: list[str]


# ---------------------------------------------------------------------------
# Detection curves
# ---------------------------------------------------------------------------
class CurveData(TypedDict):
    precision: npt.NDArray[np.floating[Any]]
    recall: npt.NDArray[np.floating[Any]]
    scores: npt.NDArray[np.floating[Any]]
    f1: npt.NDArray[np.floating[Any]]


DetectionCurves = dict[int | str, CurveData]

# ---------------------------------------------------------------------------
# Numpy alias
# ---------------------------------------------------------------------------
NDArrayFloat = npt.NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Lightning optimizer config
# ---------------------------------------------------------------------------
class LRSchedulerConfig(TypedDict, total=False):
    scheduler: torch.optim.lr_scheduler.LRScheduler
    interval: str
    frequency: int


class OptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRSchedulerConfig
