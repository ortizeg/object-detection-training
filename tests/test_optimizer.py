"""Tests for optimizer configuration."""

from __future__ import annotations

import torch

from object_detection_training.models.yolox_lightning import YOLOXSModel


def test_yolox_optimizer_config():
    """Verify YOLOX optimizer configuration."""
    model = YOLOXSModel(
        num_classes=80,
        learning_rate=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
    )

    # Manually call configure_optimizers since we're not running a full trainer
    # We need to simulate the trainer's estimated stepping if we use total_steps
    # But for now let's just check the optimizer structure returned

    # Mock trainer for LR scheduler which might need estimated_stepping_batches
    model.trainer = type(
        "Trainer", (), {"estimated_stepping_batches": 1000, "max_epochs": 100}
    )()

    optimizer_config = model.configure_optimizers()

    # Check if it returns a dictionary or just the optimizer/list
    if isinstance(optimizer_config, dict):
        optimizer = optimizer_config["optimizer"]
        scheduler_config = optimizer_config["lr_scheduler"]
    else:
        # It could be (optimizer, scheduler) or just optimizer or list of them
        # Implementation plan says we will use dict format for LR scheduler config
        # But let's handle tuple just in case
        if isinstance(optimizer_config, tuple):
            optimizer = optimizer_config[0]
            scheduler_config = (
                optimizer_config[1] if len(optimizer_config) > 1 else None
            )
        else:
            optimizer = optimizer_config
            scheduler_config = None

    # 1. Check Optimizer Type
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["nesterov"] is True

    # 2. Check Parameter Groups
    # We expect 2 groups: one with weight decay, one without (bias/norm)
    param_groups = optimizer.param_groups
    assert len(param_groups) == 2

    # Identify groups by weight_decay
    decay_group = [g for g in param_groups if g["weight_decay"] > 0]
    no_decay_group = [g for g in param_groups if g["weight_decay"] == 0]

    assert len(decay_group) == 1
    assert len(no_decay_group) == 1

    assert decay_group[0]["weight_decay"] == 0.0005
    assert no_decay_group[0]["weight_decay"] == 0.0

    # 3. Check Scheduler
    if scheduler_config:
        scheduler = scheduler_config["scheduler"]
        # It should be a composed scheduler (SequentialLR) or similar
        # Since we use warmup + cosine, it's likely SequentialLR or chained

        # Just check it exists for now and is a valid scheduler
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
