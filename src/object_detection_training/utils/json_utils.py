"""
JSON utilities for object detection training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file using orjson for fast parsing.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data as a dictionary.
    """
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_json(data: dict[str, Any], path: Path, indent: bool = True) -> None:
    """Save data to a JSON file using orjson.

    Args:
        data: Dictionary to save.
        path: Path to save the file to.
        indent: Whether to pretty-print with 2-space indentation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    options = orjson.OPT_INDENT_2 if indent else 0
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=options))
