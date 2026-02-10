"""Tests for task abstractions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from object_detection_training.tasks import BaseTask, TrainTask


class TestBaseTask:
    """Tests for BaseTask abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseTask is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseTask(name="test")  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        """Concrete subclass can be instantiated."""

        class ConcreteTask(BaseTask):
            def run(self) -> dict[str, str | None]:
                return {"result": "done"}

        task = ConcreteTask(name="test_task")
        assert task.name == "test_task"

    def test_call_invokes_run(self) -> None:
        """__call__ delegates to run()."""

        class ConcreteTask(BaseTask):
            def run(self) -> dict[str, str | None]:
                return {"result": "done"}

        task = ConcreteTask(name="test_task")
        result = task()
        assert result == {"result": "done"}

    def test_default_output_dir_is_none(self) -> None:
        """Default output_dir is None."""

        class ConcreteTask(BaseTask):
            def run(self) -> dict[str, str | None]:
                return {}

        task = ConcreteTask(name="test")
        assert task.output_dir is None

    def test_custom_output_dir(self) -> None:
        """Output dir can be set."""

        class ConcreteTask(BaseTask):
            def run(self) -> dict[str, str | None]:
                return {}

        task = ConcreteTask(name="test", output_dir=Path("/custom"))
        assert task.output_dir == Path("/custom")


class TestTrainTask:
    """Tests for TrainTask instantiation and validation."""

    def _make_train_task(self, **kwargs: object) -> TrainTask:
        """Helper to create a TrainTask with mock Lightning objects."""
        import lightning as L

        defaults: dict[str, object] = {
            "name": "train",
            "model": MagicMock(spec=L.LightningModule),
            "data": MagicMock(spec=L.LightningDataModule),
            "trainer": MagicMock(spec=L.Trainer),
        }
        defaults.update(kwargs)
        return TrainTask(**defaults)  # type: ignore[arg-type]

    def test_default_output_dir(self) -> None:
        """Output dir defaults to 'outputs' via model_validator."""
        task = self._make_train_task()
        assert task.output_dir == Path("outputs")

    def test_custom_output_dir(self) -> None:
        """Custom output dir is preserved."""
        task = self._make_train_task(output_dir=Path("/custom"))
        assert task.output_dir == Path("/custom")

    def test_default_seed(self) -> None:
        """Default seed is 42."""
        task = self._make_train_task()
        assert task.seed == 42

    @patch("object_detection_training.utils.seed.seed_everything")
    def test_run_calls_trainer_fit(self, mock_seed: MagicMock) -> None:
        """run() calls trainer.fit with model and data."""
        import lightning as L

        model = MagicMock(spec=L.LightningModule)
        model.download_pretrained = False
        model.pretrain_weights = None
        data = MagicMock(spec=L.LightningDataModule)
        data.setup = MagicMock()
        data.test_dataloader = MagicMock(return_value=None)

        trainer = MagicMock(spec=L.Trainer)
        trainer.checkpoint_callback = None

        task = self._make_train_task(model=model, data=data, trainer=trainer)
        result = task.run()

        trainer.fit.assert_called_once()
        assert "output_dir" in result
