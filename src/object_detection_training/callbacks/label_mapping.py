from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback

from object_detection_training.schemas.label_mapping import LabelMapping


class LabelMappingCallback(Callback):
    """Callback to save label mapping to JSON at the start of training."""

    def __init__(self, output_dir: str | Path = "labels_mapping.json"):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save label mapping when training starts."""
        # Use getattr to avoid MyPy "Trainer has no attribute datamodule" error
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        # Try to get class names from datamodule
        # Assuming datamodule exposes `class_names` property or attribute
        if hasattr(datamodule, "class_names"):
            class_names = datamodule.class_names
        elif hasattr(datamodule, "classes"):
            class_names = datamodule.classes
        elif hasattr(datamodule, "train_dataset") and hasattr(
            datamodule.train_dataset, "classes"
        ):
            class_names = datamodule.train_dataset.classes
        else:
            # Fallback or warning if classes not found
            return

        num_classes = len(class_names)

        # Create mapping (assuming contiguous 0-indexed IDs for now)
        id_to_name = dict(enumerate(class_names))

        # Determine output path (relative to trainer's default root dir if not absolute)
        if not self.output_dir.is_absolute():
            save_path = Path(trainer.default_root_dir) / self.output_dir
        else:
            save_path = self.output_dir

        # Construct the LabelMapping model
        # Note: The user's example had specific fields like name_to_original_id etc.
        # Ideally we'd get these from the dataset metadata if available.
        # For COCO/standard datasets, original IDs might differ.
        # Here we construct a best-effort mapping based on contiguous indices.

        # If dataset has category_id mapping (like COCO), try to use it
        name_to_original_id = {}
        original_id_to_contiguous_id = {}

        # Check if train_dataset has specific id mapping (e.g. CocoDetection)
        # This part depends on the specific dataset implementation details.
        # For now, we'll map 1:1 if no other info, assuming 1-based original IDs
        # For now, we'll map 1:1 if no other info, assuming 1-based original IDs
        # (often seen), or 0-based. But if we don't know original IDs, we might just
        # map name -> ID.
        # Let's inspect COCO dataset attribute structure if possible in future steps.
        # For now, separate mapping logic.
        # If we can't find original IDs easily, we map contiguous ID + 1 or just same.

        name_to_original_id = {name: i for i, name in enumerate(class_names)}
        original_id_to_contiguous_id = {i: i for i in range(len(class_names))}

        label_mapping = LabelMapping(
            num_classes=num_classes,
            class_names=list(class_names),
            id_to_name=id_to_name,
            name_to_original_id=name_to_original_id,
            original_id_to_contiguous_id=original_id_to_contiguous_id,
        )

        label_mapping.save_json(save_path)
