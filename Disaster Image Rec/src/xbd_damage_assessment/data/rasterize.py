"""
Polygon rasterization utilities.

Convert building polygons from xBD labels into pixel-level segmentation masks.
"""

from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
import logging

logger = logging.getLogger(__name__)


def rasterize_building_masks(
    polygons: List[Polygon],
    image_shape: Tuple[int, int],
    transform: rasterio.Affine = None,
) -> np.ndarray:
    """
    Rasterize building polygons into a binary segmentation mask.

    Args:
        polygons: List of Shapely Polygon objects representing building footprints
        image_shape: (height, width) of the output mask
        transform: Optional affine transform for geo-referencing. If None, assumes
                  pixel coordinates match polygon coordinates (typical for xBD)

    Returns:
        Binary mask of shape (height, width) where 1 = building, 0 = background
    """
    height, width = image_shape

    # Default transform: identity mapping (pixel coordinates)
    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)

    # Create binary mask
    mask = np.zeros((height, width), dtype=np.uint8)

    if not polygons:
        logger.warning("No polygons provided for rasterization")
        return mask

    # Rasterize: assign value 1 to all pixels inside any polygon
    shapes = [(polygon, 1) for polygon in polygons if polygon.is_valid]

    if not shapes:
        logger.warning("No valid polygons to rasterize")
        return mask

    try:
        mask = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,  # Include pixels touched by polygon edges
        )
    except Exception as e:
        logger.error(f"Error during rasterization: {e}")
        raise

    building_pixel_count = np.sum(mask)
    total_pixels = height * width
    coverage = building_pixel_count / total_pixels * 100

    logger.debug(
        f"Rasterized {len(polygons)} buildings: "
        f"{building_pixel_count}/{total_pixels} pixels ({coverage:.2f}% coverage)"
    )

    return mask


def rasterize_damage_masks(
    polygons: List[Polygon],
    damage_classes: List[int],
    image_shape: Tuple[int, int],
    transform: rasterio.Affine = None,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Rasterize building polygons with damage class labels into a multi-class mask.

    Args:
        polygons: List of Shapely Polygon objects
        damage_classes: List of damage class IDs (0=no-damage, 1=minor, 2=major, 3=destroyed)
        image_shape: (height, width) of output mask
        transform: Optional affine transform
        num_classes: Number of damage classes (default 4 for xBD)

    Returns:
        Multi-class mask of shape (height, width) where pixel values are damage class IDs (0-3)
        Background pixels (non-building) are set to 0 (no-damage class)
    """
    height, width = image_shape

    if len(polygons) != len(damage_classes):
        raise ValueError(
            f"Mismatch: {len(polygons)} polygons but {len(damage_classes)} damage classes"
        )

    # Default transform
    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)

    # Initialize mask with background (0 = no-damage)
    mask = np.zeros((height, width), dtype=np.uint8)

    if not polygons:
        logger.warning("No polygons provided for damage rasterization")
        return mask

    # Group polygons by damage class for efficient rasterization
    # Process in order: no-damage (0) -> destroyed (3)
    # This ensures higher damage classes overwrite lower ones in overlapping regions
    for damage_class in range(num_classes):
        shapes = [
            (polygon, damage_class)
            for polygon, dc in zip(polygons, damage_classes)
            if dc == damage_class and polygon.is_valid
        ]

        if not shapes:
            continue

        try:
            # Rasterize this damage class
            class_mask = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,  # Background remains 0
                dtype=np.uint8,
                all_touched=True,
            )

            # Update main mask where class_mask is non-zero
            mask[class_mask > 0] = damage_class

        except Exception as e:
            logger.error(f"Error rasterizing damage class {damage_class}: {e}")
            raise

    # Log damage distribution
    class_counts = {i: np.sum(mask == i) for i in range(num_classes)}
    logger.debug(f"Damage mask class distribution: {class_counts}")

    return mask


def create_multi_class_mask(
    binary_mask: np.ndarray, damage_mask: np.ndarray
) -> np.ndarray:
    """
    Combine binary building mask with damage classification mask.

    Useful when you have separate building localization and damage predictions.

    Args:
        binary_mask: Binary mask (0=background, 1=building)
        damage_mask: Damage class mask (0-3 for buildings, 0 for background)

    Returns:
        Combined mask where background=0, buildings have damage class 1-4
        (shifted by +1 to distinguish from background)
    """
    # Create output mask
    combined = np.zeros_like(binary_mask, dtype=np.uint8)

    # Only keep damage labels inside buildings
    building_pixels = binary_mask == 1
    combined[building_pixels] = damage_mask[building_pixels]

    return combined
