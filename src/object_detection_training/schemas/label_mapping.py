from pathlib import Path

from pydantic import BaseModel, Field


class LabelMapping(BaseModel):
    """Schema for label mapping configuration."""

    num_classes: int = Field(..., description="Total number of classes")
    class_names: list[str] = Field(..., description="List of class names")
    id_to_name: dict[int, str] = Field(
        ..., description="Mapping from contiguous ID to class name"
    )
    name_to_original_id: dict[str, int] = Field(
        ..., description="Mapping from class name to original (dataset) ID"
    )
    original_id_to_contiguous_id: dict[int, int] = Field(
        ..., description="Mapping from original (dataset) ID to contiguous ID"
    )

    @classmethod
    def from_json(cls, path: Path | str) -> "LabelMapping":
        """Load label mapping from a JSON file."""
        with open(path) as f:
            return cls.model_validate_json(f.read())

    def save_json(self, path: Path | str) -> None:
        """Save label mapping to a JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
