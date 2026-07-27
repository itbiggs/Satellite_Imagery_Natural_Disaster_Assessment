"""Unit tests for polygon rasterization."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from xbd_damage_assessment.data.rasterize import (
    rasterize_building_masks,
    rasterize_damage_masks,
    create_multi_class_mask,
)


def test_rasterize_building_masks_single_polygon():
    """Test rasterizing a single building polygon."""
    # Create a square polygon
    polygon = Polygon([(10, 10), (50, 10), (50, 50), (10, 50)])
    polygons = [polygon]

    mask = rasterize_building_masks(polygons, image_shape=(100, 100))

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.sum(mask) > 0  # Should have some building pixels
    assert np.all(np.isin(mask, [0, 1]))  # Binary mask


def test_rasterize_building_masks_multiple_polygons():
    """Test rasterizing multiple building polygons."""
    polygons = [
        Polygon([(10, 10), (30, 10), (30, 30), (10, 30)]),
        Polygon([(60, 60), (80, 60), (80, 80), (60, 80)]),
    ]

    mask = rasterize_building_masks(polygons, image_shape=(100, 100))

    assert mask.shape == (100, 100)
    assert np.sum(mask) > 0


def test_rasterize_building_masks_empty():
    """Test rasterizing with no polygons."""
    mask = rasterize_building_masks([], image_shape=(100, 100))

    assert mask.shape == (100, 100)
    assert np.sum(mask) == 0  # Should be all zeros


def test_rasterize_damage_masks():
    """Test rasterizing damage masks."""
    polygons = [
        Polygon([(10, 10), (30, 10), (30, 30), (10, 30)]),  # No damage
        Polygon([(60, 60), (80, 60), (80, 80), (60, 80)]),  # Destroyed
    ]
    damage_classes = [0, 3]

    mask = rasterize_damage_masks(
        polygons, damage_classes, image_shape=(100, 100), num_classes=4
    )

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.all(np.isin(mask, [0, 1, 2, 3]))  # 4-class mask


def test_rasterize_damage_masks_mismatch():
    """Test error when polygon and damage class counts don't match."""
    polygons = [Polygon([(10, 10), (30, 10), (30, 30), (10, 30)])]
    damage_classes = [0, 1]  # Mismatch!

    with pytest.raises(ValueError):
        rasterize_damage_masks(polygons, damage_classes, image_shape=(100, 100))


def test_create_multi_class_mask():
    """Test combining binary and damage masks."""
    binary_mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]], dtype=np.uint8)

    damage_mask = np.array([[0, 0, 2, 2], [0, 1, 2, 3], [0, 0, 0, 1]], dtype=np.uint8)

    combined = create_multi_class_mask(binary_mask, damage_mask)

    assert combined.shape == binary_mask.shape
    # Background should be 0
    assert combined[0, 0] == 0
    assert combined[0, 1] == 0
    # Building pixels should have damage classes
    assert combined[0, 2] == 2
    assert combined[1, 3] == 3
