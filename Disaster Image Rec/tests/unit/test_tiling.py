"""Unit tests for image tiling."""

import numpy as np
import pytest

from xbd_damage_assessment.data.tiling import (
    tile_image,
    create_tiles_with_overlap,
    pad_to_tile_size,
    reconstruct_from_tiles,
)


def test_tile_image_no_overlap():
    """Test tiling without overlap."""
    image = np.random.rand(1024, 1024, 3)
    tiles, tile_infos = tile_image(image, tile_size=512, overlap=0)

    # Should create 2x2 grid = 4 tiles
    assert len(tiles) == 4
    assert len(tile_infos) == 4

    # Check tile shapes
    for tile in tiles:
        assert tile.shape == (512, 512, 3)


def test_tile_image_with_overlap():
    """Test tiling with overlap."""
    image = np.random.rand(1024, 1024, 3)
    tiles, tile_infos = tile_image(image, tile_size=512, overlap=64)

    # With overlap, should create more tiles
    assert len(tiles) > 4
    assert len(tile_infos) == len(tiles)


def test_tile_image_2d():
    """Test tiling 2D image (grayscale)."""
    image = np.random.rand(512, 512)
    tiles, tile_infos = tile_image(image, tile_size=256, overlap=0)

    assert len(tiles) == 4
    for tile in tiles:
        assert tile.shape == (256, 256)


def test_tile_image_invalid_overlap():
    """Test error when overlap >= tile_size."""
    image = np.random.rand(512, 512, 3)

    with pytest.raises(ValueError):
        tile_image(image, tile_size=256, overlap=256)

    with pytest.raises(ValueError):
        tile_image(image, tile_size=256, overlap=300)


def test_create_tiles_with_overlap():
    """Test creating tiles with both image and mask."""
    image = np.random.rand(512, 512, 3)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

    image_tiles, mask_tiles, tile_infos = create_tiles_with_overlap(
        image, mask, tile_size=256, overlap=32
    )

    assert len(image_tiles) == len(mask_tiles)
    assert len(image_tiles) == len(tile_infos)


def test_create_tiles_with_overlap_no_mask():
    """Test creating tiles without mask."""
    image = np.random.rand(512, 512, 3)

    image_tiles, mask_tiles, tile_infos = create_tiles_with_overlap(
        image, mask=None, tile_size=256, overlap=0
    )

    assert len(image_tiles) == len(mask_tiles)
    assert all(m is None for m in mask_tiles)


def test_create_tiles_with_filtering():
    """Test filtering tiles by building content."""
    image = np.random.rand(512, 512, 3)

    # Create mask with buildings only in top-left
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:256, :256] = 1  # Only top-left has buildings

    image_tiles, mask_tiles, tile_infos = create_tiles_with_overlap(
        image, mask, tile_size=256, overlap=0, min_building_pixels=1000
    )

    # Should filter out tiles without sufficient buildings
    assert len(image_tiles) < 4


def test_pad_to_tile_size():
    """Test padding image to tile size."""
    image = np.random.rand(500, 700, 3)
    padded = pad_to_tile_size(image, tile_size=256)

    # Should pad to 512x768 (next multiples of 256)
    assert padded.shape == (512, 768, 3)


def test_pad_to_tile_size_already_divisible():
    """Test padding when image is already divisible."""
    image = np.random.rand(512, 512, 3)
    padded = pad_to_tile_size(image, tile_size=256)

    # Should not change
    assert padded.shape == (512, 512, 3)
    assert np.array_equal(padded, image)


def test_reconstruct_from_tiles_no_overlap():
    """Test reconstructing image from non-overlapping tiles."""
    original = np.random.rand(512, 512, 3)

    # Tile
    tiles, tile_infos = tile_image(original, tile_size=256, overlap=0)

    # Reconstruct
    reconstructed = reconstruct_from_tiles(
        tiles, tile_infos, original_shape=(512, 512), blend_overlap=True
    )

    assert reconstructed.shape == original.shape
    # Should be very close to original (allowing for floating point errors)
    np.testing.assert_allclose(reconstructed, original, atol=1e-5)


def test_tile_info_properties():
    """Test TileInfo properties."""
    from xbd_damage_assessment.data.tiling import TileInfo

    info = TileInfo(
        tile_id=0,
        row=0,
        col=0,
        y_start=0,
        y_end=256,
        x_start=0,
        x_end=256,
        original_shape=(512, 512),
    )

    assert info.height == 256
    assert info.width == 256
    assert "TileInfo" in repr(info)
