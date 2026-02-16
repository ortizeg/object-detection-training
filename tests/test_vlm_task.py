from unittest.mock import MagicMock, patch

from object_detection_training.schemas.detection import BoundingBox, Detection
from object_detection_training.tasks.vlm_annotation_task import VLMAnnotationTask


def test_vlm_task_initialization(tmp_path):
    task = VLMAnnotationTask(
        name="test_vlm",
        image_dir=tmp_path / "images",
        output_dir=tmp_path / "output",
        model_name="gemini-2.5-pro",
        classes=["cat", "dog"],
    )
    assert task.model_name == "gemini-2.5-pro"
    assert task.classes == ["cat", "dog"]


@patch(
    "object_detection_training.inference.gemini_inferencer.GeminiInferencer",
)
@patch("object_detection_training.io.image.ImageLoader")
def test_vlm_task_run(mock_loader_cls, mock_inferencer_cls, tmp_path):
    """Full task run with mocked inferencer and image loader."""
    # Mock image path
    mock_image_path = MagicMock()
    mock_image_path.is_file.return_value = True
    mock_image_path.suffix = ".jpg"

    # Mock ImageLoader
    mock_loader = mock_loader_cls.return_value
    mock_loader.read.return_value = MagicMock()
    mock_loader.width = 100
    mock_loader.height = 100
    mock_loader.filename = "test.jpg"

    # Mock GeminiInferencer.predict
    mock_inferencer = mock_inferencer_cls.return_value
    mock_inferencer.predict.return_value = [
        Detection(
            bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
            confidence=1.0,
            class_id=0,
        )
    ]

    output_dir = tmp_path / "output"
    task = VLMAnnotationTask(
        name="test_vlm",
        image_dir=str(tmp_path / "images"),
        output_dir=output_dir,
        model_name="gemini-2.5-pro",
        classes=["cat"],
    )

    with patch("pathlib.Path.iterdir", return_value=[mock_image_path]):
        result = task.run()

    assert output_dir.exists()
    assert (output_dir / "test.json").exists()
    assert result["num_images"] == "1"
    mock_inferencer.predict.assert_called_once()
