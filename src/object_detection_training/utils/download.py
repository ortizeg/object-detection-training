"""
Utility functions for downloading files.
"""

import urllib.request
from pathlib import Path

from loguru import logger


def download_checkpoint(url: str, destination: Path) -> Path:
    """Download a checkpoint file if it doesn't exist."""
    destination = Path(destination)
    if destination.exists():
        logger.info(f"Checkpoint already exists: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint from {url}")

    try:
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Checkpoint downloaded to {destination}")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise

    return destination
