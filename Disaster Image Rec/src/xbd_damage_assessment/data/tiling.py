"""
Image tiling utilities.

Split large satellite images into smaller patches for model training.
Handles overlapping tiles to ensure edge buildings are properly captured.
"""

from typing import List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TileInfo:
    """Container for tile metadata."""

    def __init__(
        self,
        tile_id: int,
        row: int,
        col: int,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        original_shape: Tuple[int, int],
    ):
        self.tile_id = tile_id
        self.row = row
        self.col = col
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end
        self.original_shape = original_shape

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    def __repr__(self) -> str:
        return (
            f"TileInfo(id={self.tile_id}, row={self.row}, col={self.col}, "
            f"bounds=({self.y_start}:{self.y_end}, {self.x_start}:{self.x_end}))"
        )


def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 0,
    min_tile_size: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[TileInfo]]:
    """
    Tile an image into smaller patches with optional overlap.

    Args:
        image: Input image of shape (H, W, C) or (H, W)
        tile_size: Size of square tiles (default 512x512)
        overlap: Overlap between adjacent tiles in pixels (default 0)
        min_tile_size: Minimum tile size to keep. If None, defaults to tile_size // 2.
                      Tiles smaller than this are discarded.

    Returns:
        Tuple of:
            - tiles: List of image tiles
            - tile_infos: List of TileInfo objects with metadata for each tile

    Example:
        >>> image = np.random.rand(1024, 1024, 3)
        >>> tiles, infos = tile_image(image, tile_size=512, overlap=64)
        >>> print(f"Generated {len(tiles)} tiles")
    """
    if min_tile_size is None:
        min_tile_size = tile_size // 2

    # Get image dimensions
    if image.ndim == 2:
        height, width = image.shape
        has_channels = False
    elif image.ndim == 3:
        height, width, channels = image.shape
        has_channels = True
    else:
        raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")

    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than tile_size ({tile_size})")

    tiles = []
    tile_infos = []
    tile_id = 0

    # Calculate tile grid
    rows = []
    cols = []

    # Row boundaries
    y = 0
    row_idx = 0
    while y < height:
        y_end = min(y + tile_size, height)
        if y_end - y >= min_tile_size:
            rows.append((row_idx, y, y_end))
            row_idx += 1
        y += stride

    # Column boundaries
    x = 0
    col_idx = 0
    while x < width:
        x_end = min(x + tile_size, width)
        if x_end - x >= min_tile_size:
            cols.append((col_idx, x, x_end))
            col_idx += 1
        x += stride

    # Extract tiles
    for row_idx, y_start, y_end in rows:
        for col_idx, x_start, x_end in cols:
            # Extract tile
            if has_channels:
                tile = image[y_start:y_end, x_start:x_end, :]
            else:
                tile = image[y_start:y_end, x_start:x_end]

            # Create tile info
            info = TileInfo(
                tile_id=tile_id,
                row=row_idx,
                col=col_idx,
                y_start=y_start,
                y_end=y_end,
                x_start=x_start,
                x_end=x_end,
                original_shape=(height, width),
            )

            tiles.append(tile)
            tile_infos.append(info)
            tile_id += 1

    logger.info(
        f"Tiled image ({height}x{width}) into {len(tiles)} tiles "
        f"(size={tile_size}, overlap={overlap}, grid={len(rows)}x{len(cols)})"
    )

    return tiles, tile_infos


def create_tiles_with_overlap(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    tile_size: int = 512,
    overlap: int = 64,
    min_building_pixels: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[TileInfo]]:
    """
    Create overlapping tiles from image and optional mask, with filtering.

    This is the main function for preprocessing xBD data. It tiles both
    the imagery and corresponding masks, and optionally filters out
    tiles with insufficient building content.

    Args:
        image: Input image (H, W, C)
        mask: Optional segmentation mask (H, W). If None, returns None masks.
        tile_size: Size of square tiles
        overlap: Overlap between tiles in pixels
        min_building_pixels: Minimum number of building pixels (mask > 0) required
                           to keep a tile. Set to 0 to keep all tiles.

    Returns:
        Tuple of:
            - image_tiles: List of image tiles
            - mask_tiles: List of mask tiles (or list of None if mask is None)
            - tile_infos: List of TileInfo objects
    """
    # Tile image
    image_tiles, tile_infos = tile_image(
        image, tile_size=tile_size, overlap=overlap, min_tile_size=tile_size // 2
    )

    # Tile mask if provided
    if mask is not None:
        mask_tiles, _ = tile_image(
            mask, tile_size=tile_size, overlap=overlap, min_tile_size=tile_size // 2
        )

        if len(image_tiles) != len(mask_tiles):
            raise RuntimeError(
                f"Tile count mismatch: {len(image_tiles)} image tiles "
                f"vs {len(mask_tiles)} mask tiles"
            )
    else:
        mask_tiles = [None] * len(image_tiles)

    # Filter tiles by building content if requested
    if min_building_pixels > 0 and mask is not None:
        filtered_images = []
        filtered_masks = []
        filtered_infos = []

        for img_tile, mask_tile, info in zip(image_tiles, mask_tiles, tile_infos):
            building_pixels = np.sum(mask_tile > 0)

            if building_pixels >= min_building_pixels:
                filtered_images.append(img_tile)
                filtered_masks.append(mask_tile)
                filtered_infos.append(info)

        num_filtered = len(image_tiles) - len(filtered_images)
        logger.info(
            f"Filtered {num_filtered}/{len(image_tiles)} tiles with "
            f"< {min_building_pixels} building pixels"
        )

        return filtered_images, filtered_masks, filtered_infos
    else:
        return image_tiles, mask_tiles, tile_infos


def pad_to_tile_size(image: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Pad an image to be evenly divisible by tile_size.

    Args:
        image: Input image (H, W) or (H, W, C)
        tile_size: Target tile size

    Returns:
        Padded image
    """
    height, width = image.shape[:2]

    # Calculate padding needed
    pad_height = (tile_size - height % tile_size) % tile_size
    pad_width = (tile_size - width % tile_size) % tile_size

    if pad_height == 0 and pad_width == 0:
        return image

    # Pad at bottom and right
    if image.ndim == 2:
        padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode="reflect")
    else:
        padded = np.pad(
            image, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect"
        )

    logger.debug(f"Padded image from {image.shape} to {padded.shape}")

    return padded


def reconstruct_from_tiles(
    tiles: List[np.ndarray],
    tile_infos: List[TileInfo],
    original_shape: Tuple[int, int],
    blend_overlap: bool = True,
) -> np.ndarray:
    """
    Reconstruct full image from overlapping tiles.

    Args:
        tiles: List of tile images
        tile_infos: Corresponding TileInfo objects
        original_shape: (height, width) of original image
        blend_overlap: If True, average overlapping regions. Otherwise, use max.

    Returns:
        Reconstructed image of original_shape
    """
    height, width = original_shape

    # Determine output shape
    if tiles[0].ndim == 3:
        channels = tiles[0].shape[2]
        output = np.zeros((height, width, channels), dtype=tiles[0].dtype)
        counts = np.zeros((height, width, 1), dtype=np.float32)
    else:
        output = np.zeros((height, width), dtype=tiles[0].dtype)
        counts = np.zeros((height, width), dtype=np.float32)

    # Place tiles
    for tile, info in zip(tiles, tile_infos):
        output[info.y_start : info.y_end, info.x_start : info.x_end] += tile
        counts[info.y_start : info.y_end, info.x_start : info.x_end] += 1

    # Average overlapping regions
    if blend_overlap:
        counts = np.maximum(counts, 1)  # Avoid division by zero
        output = (output / counts).astype(tiles[0].dtype)

    logger.info(f"Reconstructed image of shape {output.shape} from {len(tiles)} tiles")

    return output
