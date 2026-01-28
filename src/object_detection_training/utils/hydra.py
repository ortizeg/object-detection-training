from typing import Any, List, Optional, Type, Union

import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig


def register(
    cls: Optional[Type[Any]] = None,
    *,
    group: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Union[Type[Any], Any]:
    """
    Decorator to register a class with Hydra's ConfigStore.

    Automatically creates a configuration entry with the correct `_target_`
    pointing to the decorated class and registers it in the ConfigStore.

    If `group` is not provided, it tries to infer it from the module path
    by taking the second to last element of the module path.

    Arguments:
        cls: The class to register.
        group: The ConfigStore group. If None, inference is attempted.
        name: The name for the config. Defaults to class name.
        **kwargs: Default values for the configuration node.
    """

    def _process_class(target_cls: Type[Any]) -> Type[Any]:
        nonlocal group, name

        # Determine the target path (module + class name)
        target_path = f"{target_cls.__module__}.{target_cls.__name__}"

        # Use provided name or class name as the config name
        config_name = name or target_cls.__name__

        # Infer group if not provided
        if group is None:
            module_parts = target_cls.__module__.split(".")
            group = module_parts[-2]

        # Register in ConfigStore
        cs = ConfigStore.instance()
        logger.debug(
            f"Registering {target_cls.__name__} as '{config_name}' in group '{group}'"
        )
        # Build the configuration node
        node = {"_target_": target_path}
        node.update(kwargs)
        cs.store(group=group, name=config_name, node=node)

        return target_cls

    if cls is None:
        return _process_class
    return _process_class(cls)


def instantiate_datamodule(cfg: DictConfig) -> Any:
    """
    Instantiate a Lightning DataModule from Hydra config.

    Args:
        cfg: Hydra configuration for the datamodule.

    Returns:
        Instantiated LightningDataModule.
    """
    logger.debug(f"Instantiating DataModule: {cfg.get('_target_', 'unknown')}")
    datamodule = hydra.utils.instantiate(cfg)
    logger.info(f"DataModule instantiated: {type(datamodule).__name__}")
    return datamodule


def instantiate_model(
    cfg: DictConfig, checkpoint_path: Optional[str] = None, **kwargs: Any
) -> Any:
    """
    Instantiate a Lightning Module from Hydra config.

    Args:
        cfg: Hydra configuration for the model.
        checkpoint_path: Optional path to a checkpoint to load.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        Instantiated LightningModule.
    """
    logger.debug(f"Instantiating Model: {cfg.get('_target_', 'unknown')}")

    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        # If model class supports from_checkpoint, use it
        model_cls = hydra.utils.get_class(cfg._target_)
        if hasattr(model_cls, "load_from_checkpoint"):
            model = model_cls.load_from_checkpoint(checkpoint_path)
        else:
            model = hydra.utils.instantiate(
                cfg, pretrain_weights=checkpoint_path, **kwargs
            )
    else:
        model = hydra.utils.instantiate(cfg, **kwargs)

    logger.info(f"Model instantiated: {type(model).__name__}")
    return model


def instantiate_callbacks(cfg: DictConfig) -> List[Any]:
    """
    Instantiate a list of callbacks from Hydra config.

    Args:
        cfg: Hydra configuration containing callback definitions.

    Returns:
        List of instantiated callbacks.
    """
    callbacks = []

    if cfg is None:
        logger.debug("No callbacks configured")
        return callbacks

    for name, callback_cfg in cfg.items():
        if callback_cfg is not None and "_target_" in callback_cfg:
            logger.debug(f"Instantiating callback: {name}")
            callback = hydra.utils.instantiate(callback_cfg)
            callbacks.append(callback)
            logger.info(f"Callback instantiated: {type(callback).__name__}")

    return callbacks


def instantiate_loggers(cfg: DictConfig) -> List[Any]:
    """
    Instantiate a list of loggers from Hydra config.

    Args:
        cfg: Hydra configuration containing logger definitions.

    Returns:
        List of instantiated loggers.
    """
    loggers = []

    if cfg is None:
        logger.debug("No loggers configured")
        return loggers

    for name, logger_cfg in cfg.items():
        if logger_cfg is not None and "_target_" in logger_cfg:
            logger.debug(f"Instantiating logger: {name}")
            log_instance = hydra.utils.instantiate(logger_cfg)
            loggers.append(log_instance)
            logger.info(f"Logger instantiated: {type(log_instance).__name__}")

    return loggers


def instantiate_trainer(
    cfg: DictConfig,
    callbacks: Optional[List[Any]] = None,
    loggers: Optional[List[Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a PyTorch Lightning Trainer from Hydra config.

    Args:
        cfg: Hydra configuration for the trainer.
        callbacks: Optional list of callbacks to add.
        loggers: Optional list of loggers to add.
        **kwargs: Additional keyword arguments to pass to the Trainer.

    Returns:
        Instantiated Trainer.
    """
    import lightning as L

    logger.debug("Instantiating Trainer")

    trainer_kwargs = dict(cfg) if cfg else {}

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    if loggers:
        trainer_kwargs["logger"] = loggers

    trainer_kwargs.update(kwargs)

    # Remove _target_ if present (we instantiate Trainer directly)
    trainer_kwargs.pop("_target_", None)

    trainer = L.Trainer(**trainer_kwargs)
    logger.info(f"Trainer instantiated with {len(callbacks or [])} callbacks")

    return trainer
