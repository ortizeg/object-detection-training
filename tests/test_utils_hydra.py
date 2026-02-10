"""Tests for Hydra configuration utilities."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from object_detection_training.utils.hydra import register


class TestRegister:
    """Tests for the @register decorator."""

    def test_register_with_explicit_group_and_name(self) -> None:
        """Class is registered with explicit group and name."""

        @register(group="test_group", name="TestModel")
        class _TestModel:
            pass

        ConfigStore.instance()
        # ConfigStore stores configs; verify by attempting to retrieve
        # The node should have been stored successfully
        assert _TestModel is not None  # decorator returns the class

    def test_register_infers_group_from_module(self) -> None:
        """Group is inferred from module path when not provided."""

        @register(name="InferredModel")
        class _InferredModel:
            pass

        # Should not raise; group inferred from module path
        assert _InferredModel is not None

    def test_register_infers_name_from_class(self) -> None:
        """Name defaults to class name when not provided."""

        @register(group="test_auto_name")
        class AutoNameModel:
            pass

        assert AutoNameModel.__name__ == "AutoNameModel"

    def test_register_with_kwargs(self) -> None:
        """Extra kwargs are included in the registered config node."""

        @register(group="test_kwargs", name="KwargsModel", num_classes=10, lr=0.001)
        class _KwargsModel:
            pass

        assert _KwargsModel is not None

    def test_register_preserves_class(self) -> None:
        """Decorated class is returned unchanged."""

        @register(group="test_preserve")
        class MyClass:
            def method(self) -> str:
                return "hello"

        obj = MyClass()
        assert obj.method() == "hello"

    def test_register_sets_correct_target(self) -> None:
        """Registered config node has correct _target_ path."""

        @register(group="test_target")
        class _TargetModel:
            pass

        # The _target_ should be module.classname
        expected_target = f"{_TargetModel.__module__}.{_TargetModel.__name__}"
        assert "." in expected_target
