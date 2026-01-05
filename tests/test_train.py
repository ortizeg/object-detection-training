import sys
from os.path import dirname, join

import hydra

# Add src to path to import train script if needed, though we primarily test config here
# Add src to path to import train script if needed, though we primarily test config here
sys.path.append(join(dirname(__file__), "../src"))

# Register model configs by importing the wrappers
import object_detection_training.models.rfdetr_wrappers as _  # noqa: F401, E402


def test_hydra_configuration():
    # Verify that we can load the configuration using Hydra
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="train")

        # Check basic config structure
        assert "training" in cfg
        assert "models" in cfg

        # Check default values
        assert cfg.training.batch_size == 8
        assert (
            cfg.models._target_
            == "object_detection_training.models.rfdetr_wrappers.RFDETRSmall"
        )
