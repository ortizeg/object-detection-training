"""Tests for JSON utilities."""

from __future__ import annotations

from pathlib import Path

from object_detection_training.utils.json_utils import load_json, save_json


class TestSaveJson:
    """Tests for save_json function."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Data survives a save/load roundtrip."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        path = tmp_path / "a" / "b" / "c" / "test.json"
        data = {"hello": "world"}
        save_json(data, path)
        assert path.exists()
        loaded = load_json(path)
        assert loaded == data

    def test_indent_option(self, tmp_path: Path) -> None:
        """Indented output is larger than compact output."""
        data = {"key": "value", "list": [1, 2, 3]}
        indented_path = tmp_path / "indented.json"
        compact_path = tmp_path / "compact.json"

        save_json(data, indented_path, indent=True)
        save_json(data, compact_path, indent=False)

        assert indented_path.stat().st_size > compact_path.stat().st_size

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Existing file is overwritten."""
        path = tmp_path / "test.json"
        save_json({"first": True}, path)
        save_json({"second": True}, path)
        loaded = load_json(path)
        assert loaded == {"second": True}


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_returns_dict(self, tmp_path: Path) -> None:
        """Loaded JSON is a Python dict."""
        path = tmp_path / "test.json"
        save_json({"x": 1}, path)
        result = load_json(path)
        assert isinstance(result, dict)
        assert result["x"] == 1
