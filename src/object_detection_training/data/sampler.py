"""Online class-balanced sampling for object detection training.

Provides weighted random sampling to re-balance class representation
at the image level. Each image's weight is determined by its rarest
(highest-weight) annotation class.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from loguru import logger
from pydantic import BaseModel, model_validator
from torch.utils.data import WeightedRandomSampler


class SamplerConfig(BaseModel, frozen=True):
    """Configuration for online class-balanced sampling.

    Modes:
        disabled: Standard shuffle, no reweighting.
        auto: Inverse-frequency weights computed from annotations.
        manual: User-provided per-class weights.
    """

    mode: Literal["disabled", "auto", "manual"] = "disabled"
    class_weights: dict[str, float] | None = None
    num_samples: int | None = None
    replacement: bool = True

    @model_validator(mode="after")
    def _validate_manual_weights(self) -> SamplerConfig:
        if self.mode == "manual" and not self.class_weights:
            raise ValueError("class_weights must be provided when mode='manual'")
        return self


def build_weighted_sampler(
    config: SamplerConfig,
    annotations_df: pd.DataFrame,
    image_ids: list[int],
    class_names: list[str],
) -> WeightedRandomSampler | None:
    """Build a WeightedRandomSampler based on class frequency.

    Args:
        config: Sampler configuration.
        annotations_df: DataFrame with 'image_id' and 'category_name' columns.
        image_ids: Ordered list of image IDs (index-aligned with dataset).
        class_names: List of all class names in the dataset.

    Returns:
        WeightedRandomSampler when enabled, None when disabled.

    Raises:
        ValueError: If manual weights are missing classes.
    """
    if config.mode == "disabled":
        return None

    # Compute per-class weight
    if config.mode == "auto":
        class_counts = annotations_df.groupby("category_name").size()
        total = class_counts.sum()
        class_weight_map: dict[str, float] = {}
        for name in class_names:
            count = class_counts.get(name, 0)
            class_weight_map[name] = total / count if count > 0 else 0.0
        # Normalize so weights sum to len(class_names)
        weight_sum = sum(class_weight_map.values())
        if weight_sum > 0:
            scale = len(class_names) / weight_sum
            class_weight_map = {k: v * scale for k, v in class_weight_map.items()}
    else:
        # manual mode — class_weights guaranteed by SamplerConfig validator
        manual_weights = config.class_weights
        if manual_weights is None:
            raise ValueError("class_weights must be provided when mode='manual'")
        missing = set(class_names) - set(manual_weights.keys())
        if missing:
            raise ValueError(f"Manual class_weights missing classes: {sorted(missing)}")
        class_weight_map = dict(manual_weights)

    logger.info(f"Sampler class weights: {class_weight_map}")

    # Compute per-image weight = max weight of its annotations
    # Images with no annotations get median weight
    image_id_set = set(image_ids)
    annotated = annotations_df[annotations_df["image_id"].isin(image_id_set)]
    ann_weights = annotated["category_name"].map(class_weight_map)
    per_image_max = ann_weights.groupby(annotated["image_id"]).max()

    all_weights = list(per_image_max.values)
    median_weight = float(pd.Series(all_weights).median()) if all_weights else 1.0

    image_weights: list[float] = []
    for img_id in image_ids:
        if img_id in per_image_max.index:
            image_weights.append(float(per_image_max[img_id]))
        else:
            image_weights.append(median_weight)

    num_samples = (
        config.num_samples if config.num_samples is not None else len(image_ids)
    )

    logger.info(
        f"OnlineSampler: mode={config.mode}, num_samples={num_samples}, "
        f"replacement={config.replacement}"
    )

    return WeightedRandomSampler(
        weights=image_weights,
        num_samples=num_samples,
        replacement=config.replacement,
    )
