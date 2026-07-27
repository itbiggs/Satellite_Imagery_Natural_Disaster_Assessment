"""
Main preprocessing script for xBD dataset.

Converts raw xBD data into model-ready format:
1. Parse JSON labels
2. Rasterize building polygons to masks
3. Tile large images into smaller patches
4. Split into train/val/test sets
5. Save processed data
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random
import json
import numpy as np
import cv2
from tqdm import tqdm
import yaml

from .label_parser import xBDLabelParser
from .rasterize import rasterize_building_masks, rasterize_damage_masks
from .tiling import create_tiles_with_overlap
from ..utils.io import ensure_dir

logger = logging.getLogger(__name__)


class xBDPreprocessor:
    """Preprocessor for xBD disaster damage dataset."""

    def __init__(self, config: Dict):
        self.config = config
        self.label_parser = xBDLabelParser(
            damage_class_map=config["damage_classes"]["class_mapping"]
        )

        # Setup directories
        self.raw_root = Path(config["data"]["raw_root"])
        self.processed_root = Path(config["data"]["processed_root"])
        self.interim_root = Path(config["data"]["interim_root"])

        ensure_dir(self.processed_root)
        ensure_dir(self.interim_root)

    def process_all(self) -> None:
        """Run complete preprocessing pipeline."""
        logger.info("Starting xBD preprocessing pipeline...")

        # Discover raw data files
        disaster_ids = self._discover_disasters()
        logger.info(f"Found {len(disaster_ids)} disaster events")

        # Split into train/val/test
        splits = self._create_splits(disaster_ids)

        # Process each split
        for split_name, split_disasters in splits.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {split_name} split ({len(split_disasters)} disasters)")
            logger.info(f"{'='*60}\n")

            self._process_split(split_name, split_disasters)

        logger.info("\nPreprocessing complete!")
        self._print_statistics()

    def _discover_disasters(self) -> List[str]:
        """Discover all disaster IDs in the raw data."""
        train_dir = self.raw_root / "train" / "images"

        if not train_dir.exists():
            raise FileNotFoundError(
                f"Raw data not found at {train_dir}. "
                f"Please download xBD dataset and extract to {self.raw_root}"
            )

        # Find all pre-disaster images
        pre_images = sorted(train_dir.glob("*_pre_disaster.png"))

        # Extract disaster IDs (everything before _pre_disaster)
        disaster_ids = [img.stem.replace("_pre_disaster", "") for img in pre_images]

        logger.info(f"Discovered {len(disaster_ids)} disaster events")
        return disaster_ids

    def _create_splits(self, disaster_ids: List[str]) -> Dict[str, List[str]]:
        """Split disasters into train/val/test sets."""
        random.seed(self.config["split"]["random_seed"])
        random.shuffle(disaster_ids)

        train_ratio = self.config["split"]["train_ratio"]
        val_ratio = self.config["split"]["val_ratio"]

        n_total = len(disaster_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": disaster_ids[:n_train],
            "val": disaster_ids[n_train : n_train + n_val],
            "test": disaster_ids[n_train + n_val :],
        }

        logger.info(
            f"Split: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        # Save split metadata
        split_file = self.processed_root / "splits.json"
        with open(split_file, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Saved split info to {split_file}")

        return splits

    def _process_split(self, split_name: str, disaster_ids: List[str]) -> None:
        """Process all disasters in a split."""
        # Create output directories
        split_dir = self.processed_root / split_name
        images_dir = ensure_dir(split_dir / "images")
        masks_building_dir = ensure_dir(split_dir / "masks_building")
        masks_damage_dir = ensure_dir(split_dir / "masks_damage")

        tile_counter = 0
        tile_list = []

        for disaster_id in tqdm(disaster_ids, desc=f"Processing {split_name}"):
            try:
                tiles = self._process_disaster(disaster_id)
                tile_counter = self._save_tiles(
                    tiles, images_dir, masks_building_dir, masks_damage_dir, tile_counter
                )
                tile_list.extend([t["tile_id"] for t in tiles])

            except Exception as e:
                logger.error(f"Error processing {disaster_id}: {e}")
                continue

        # Save tile list
        tile_list_file = split_dir / "tile_list.txt"
        with open(tile_list_file, "w") as f:
            f.write("\n".join(tile_list))

        logger.info(f"Saved {len(tile_list)} tiles to {split_dir}")

    def _process_disaster(self, disaster_id: str) -> List[Dict]:
        """Process a single disaster event."""
        # Load images
        images_dir = self.raw_root / "train" / "images"
        labels_dir = self.raw_root / "train" / "labels"

        pre_image_path = images_dir / f"{disaster_id}_pre_disaster.png"
        post_image_path = images_dir / f"{disaster_id}_post_disaster.png"
        label_path = labels_dir / f"{disaster_id}_pre_disaster.json"

        if not all([pre_image_path.exists(), post_image_path.exists(), label_path.exists()]):
            logger.warning(f"Missing files for {disaster_id}, skipping")
            return []

        # Load images
        pre_image = cv2.imread(str(pre_image_path), cv2.IMREAD_COLOR)
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)

        post_image = cv2.imread(str(post_image_path), cv2.IMREAD_COLOR)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        # Parse labels
        polygons, damage_classes, building_uids = self.label_parser.parse(label_path)

        if not polygons:
            logger.warning(f"No buildings found in {disaster_id}, skipping")
            return []

        # Rasterize masks
        image_shape = pre_image.shape[:2]

        building_mask = rasterize_building_masks(polygons, image_shape)
        damage_mask = rasterize_damage_masks(
            polygons, damage_classes, image_shape, num_classes=4
        )

        # Create tiles
        tiles = self._create_tiles(
            disaster_id, pre_image, post_image, building_mask, damage_mask
        )

        return tiles

    def _create_tiles(
        self,
        disaster_id: str,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        building_mask: np.ndarray,
        damage_mask: np.ndarray,
    ) -> List[Dict]:
        """Create overlapping tiles from images and masks."""
        tile_size = self.config["tiling"]["tile_size"]
        overlap = self.config["tiling"]["overlap"]
        min_building_pixels = self.config["tiling"]["min_building_pixels"]

        # Tile pre-image and building mask
        pre_tiles, building_tiles, tile_infos = create_tiles_with_overlap(
            pre_image,
            building_mask,
            tile_size=tile_size,
            overlap=overlap,
            min_building_pixels=min_building_pixels,
        )

        # Tile post-image and damage mask
        post_tiles, damage_tiles, _ = create_tiles_with_overlap(
            post_image,
            damage_mask,
            tile_size=tile_size,
            overlap=overlap,
            min_building_pixels=0,  # Use same tiles as pre-image
        )

        # Package tiles
        tiles = []
        for i, info in enumerate(tile_infos):
            tile_id = f"{disaster_id}_tile_{info.tile_id:04d}"
            tiles.append(
                {
                    "tile_id": tile_id,
                    "pre_image": pre_tiles[i],
                    "post_image": post_tiles[i],
                    "building_mask": building_tiles[i],
                    "damage_mask": damage_tiles[i],
                    "info": info,
                }
            )

        return tiles

    def _save_tiles(
        self,
        tiles: List[Dict],
        images_dir: Path,
        masks_building_dir: Path,
        masks_damage_dir: Path,
        tile_counter: int,
    ) -> int:
        """Save tiles to disk."""
        for tile in tiles:
            tile_id = tile["tile_id"]

            # Save images
            pre_path = images_dir / f"{tile_id}_pre.png"
            post_path = images_dir / f"{tile_id}_post.png"

            cv2.imwrite(
                str(pre_path), cv2.cvtColor(tile["pre_image"], cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                str(post_path), cv2.cvtColor(tile["post_image"], cv2.COLOR_RGB2BGR)
            )

            # Save masks
            building_mask_path = masks_building_dir / f"{tile_id}.png"
            damage_mask_path = masks_damage_dir / f"{tile_id}.png"

            cv2.imwrite(str(building_mask_path), tile["building_mask"])
            cv2.imwrite(str(damage_mask_path), tile["damage_mask"])

            tile_counter += 1

        return tile_counter

    def _print_statistics(self) -> None:
        """Print preprocessing statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("Preprocessing Statistics")
        logger.info("=" * 60)

        for split in ["train", "val", "test"]:
            split_dir = self.processed_root / split
            tile_list_file = split_dir / "tile_list.txt"

            if tile_list_file.exists():
                with open(tile_list_file, "r") as f:
                    num_tiles = len(f.readlines())
                logger.info(f"{split:>6s}: {num_tiles:>6d} tiles")

        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess xBD dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocess.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup logging
    log_level = args.log_level or config["logging"]["level"]
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run preprocessing
    preprocessor = xBDPreprocessor(config)
    preprocessor.process_all()


if __name__ == "__main__":
    main()
