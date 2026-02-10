# Configuration

The project uses [Hydra](https://hydra.cc/) for hierarchical configuration management.

## Config Structure

All configs live in `src/object_detection_training/conf/`:

```
conf/
├── train.yaml          # Main training config (composes all groups)
├── models/             # Model configs (rfdetr_small, yolox_s, etc.)
├── data/               # Dataset configs
├── callbacks/          # Callback configs
├── trainer/            # Trainer configs
├── logging/            # Logger configs (wandb, tensorboard)
└── task/               # Task configs
```

## Config Groups

Switch between configurations using Hydra overrides:

```bash
# Use YOLOX instead of RFDETR
pixi run train -- models=yolox_s

# Change dataset
pixi run train -- data=custom_coco
```

## Command-line Overrides

Override any config value from the command line:

```bash
pixi run train -- trainer.max_epochs=100 data.batch_size=32
```

## The @register Decorator

Models, tasks, and data modules are registered with Hydra's ConfigStore via the `@register` decorator:

```python
from object_detection_training.utils.hydra import register

# Explicit group and name
@register(group="task")
class TrainTask(BaseTask):
    ...

# Group and name inferred from module path and class name
@register
class COCODataModule(L.LightningDataModule):
    ...

# Custom name (group inferred as "models" from module path)
@register(name="RFDETRNano")
class RFDETRNanoModel(RFDETRLightningModel):
    ...
```

Any extra `**kwargs` passed to `@register` become default values in the Hydra config node. This makes the class available as a Hydra config group option.

## Output Directory

Hydra manages output directories automatically:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/           # Config snapshots
        ├── checkpoints/      # Model checkpoints
        └── logs/             # Training logs
```
