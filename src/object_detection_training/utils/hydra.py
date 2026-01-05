from dataclasses import dataclass, make_dataclass
from typing import Any, Optional, Type, Union

from hydra.core.config_store import ConfigStore


def register(
    cls: Optional[Type[Any]] = None, *, group: Optional[str] = None
) -> Union[Type[Any], Any]:
    """
    Decorator to register a class with Hydra's ConfigStore.

    Automatically creates a configuration dataclass with the correct `_target_`
    pointing to the decorated class and registers it in the ConfigStore.

    If `group` is not provided, it tries to infer it from the module path:
    - `src.models.xyz` -> group="model"
    - `src.datasets.xyz` -> group="dataset"
    - Defaults to "model" if no known keyword is found.

    Arguments:
        cls: The class to register.
        group: The ConfigStore group. If None, inference is attempted.
    """

    def _process_class(target_cls: Type[Any]) -> Type[Any]:
        nonlocal group

        # Determine the target path (module + class name)
        # Assuming the class is defined in a module that is importable
        target_path = f"{target_cls.__module__}.{target_cls.__name__}"

        # Use class name as the config name
        config_name = target_cls.__name__

        # Infer group if not provided
        if group is None:
            module_parts = target_cls.__module__.split(".")
            group = module_parts[-2]

        # Create the configuration dataclass dynamically
        config_cls_name = f"{target_cls.__name__}Config"
        ConfigClass = make_dataclass(
            config_cls_name,
            [("_target_", str, target_path)],
            bases=(),
            namespace={"__module__": target_cls.__module__},
        )
        # Apply @dataclass decorator
        ConfigClass = dataclass(ConfigClass)

        # Register in ConfigStore
        cs = ConfigStore.instance()
        cs.store(group=group, name=config_name, node=ConfigClass)

        return target_cls

    if cls is None:
        return _process_class
    return _process_class(cls)
