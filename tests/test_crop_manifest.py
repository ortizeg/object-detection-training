"""Tests for crop manifest schemas and RLE encode/decode."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from object_detection_training.schemas.crop_manifest import (
    CropMetadata,
    RLEMask,
    decode_rle,
    encode_rle,
)


class TestRLEMask:
    """Tests for RLE encode/decode roundtrips."""

    def test_roundtrip_random(self) -> None:
        """Random binary mask survives encode → decode roundtrip."""
        rng = np.random.default_rng(42)
        mask = rng.random((64, 48)) > 0.5
        rle = encode_rle(mask)
        recovered = decode_rle(rle)
        np.testing.assert_array_equal(recovered, mask)

    def test_all_zeros(self) -> None:
        """All-zero mask roundtrip."""
        mask = np.zeros((32, 32), dtype=bool)
        rle = encode_rle(mask)
        recovered = decode_rle(rle)
        np.testing.assert_array_equal(recovered, mask)
        assert rle.counts == [32 * 32]

    def test_all_ones(self) -> None:
        """All-one mask roundtrip."""
        mask = np.ones((16, 16), dtype=bool)
        rle = encode_rle(mask)
        recovered = decode_rle(rle)
        np.testing.assert_array_equal(recovered, mask)
        # Starts with 0-length bg run, then full fg run
        assert rle.counts == [0, 16 * 16]

    def test_checkerboard(self) -> None:
        """Alternating checkerboard pattern in 1D sense."""
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, :] = True
        mask[2, :] = True
        rle = encode_rle(mask)
        recovered = decode_rle(rle)
        np.testing.assert_array_equal(recovered, mask)

    def test_dimensions_preserved(self) -> None:
        """RLE preserves height and width."""
        mask = np.zeros((100, 200), dtype=bool)
        rle = encode_rle(mask)
        assert rle.height == 100
        assert rle.width == 200
        recovered = decode_rle(rle)
        assert recovered.shape == (100, 200)

    def test_frozen(self) -> None:
        """RLEMask is immutable."""
        rle = RLEMask(counts=[10, 5], height=3, width=5)
        with pytest.raises(ValidationError):
            rle.height = 99  # type: ignore[misc]


class TestCropMetadata:
    """Tests for CropMetadata schema."""

    def _make_metadata(self) -> CropMetadata:
        return CropMetadata(
            filename="12345.png",
            mask=RLEMask(counts=[10, 20, 30], height=8, width=10),
            bbox_x=10.0,
            bbox_y=20.0,
            bbox_w=50.0,
            bbox_h=60.0,
            source_image="frame_001.jpg",
            annotation_id=12345,
            category_name="ball",
        )

    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """CropMetadata survives save → load roundtrip."""
        meta = self._make_metadata()
        path = tmp_path / "test_meta.json"  # type: ignore[operator]
        meta.save_json(path)
        loaded = CropMetadata.from_json(path)
        assert loaded == meta

    def test_frozen(self) -> None:
        """CropMetadata is immutable."""
        meta = self._make_metadata()
        with pytest.raises(ValidationError):
            meta.annotation_id = 999  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        """All fields are accessible."""
        meta = self._make_metadata()
        assert meta.filename == "12345.png"
        assert meta.category_name == "ball"
        assert meta.bbox_w == 50.0
        assert meta.mask.height == 8
