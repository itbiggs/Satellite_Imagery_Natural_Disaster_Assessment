"""
xBD JSON label parser.

Parses xBD/xView2 JSON label files to extract building polygons and damage classifications.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
from shapely import wkt
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)


# Damage class mapping
DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 0,  # Treat unclassified as no-damage for pre-disaster
}

DAMAGE_CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


class xBDLabelParser:
    """
    Parser for xBD JSON label files.

    The xBD dataset provides labels as JSON files with the following structure:
    {
        "features": {
            "xy": [
                {
                    "wkt": "POLYGON ((x1 y1, x2 y2, ...))",
                    "properties": {
                        "feature_type": "building",
                        "subtype": "no-damage" | "minor-damage" | "major-damage" | "destroyed",
                        "uid": "unique_building_id"
                    }
                },
                ...
            ]
        },
        "metadata": {...}
    }
    """

    def __init__(self, damage_class_map: Dict[str, int] = None):
        """
        Initialize the label parser.

        Args:
            damage_class_map: Optional custom mapping from damage subtype strings to class IDs.
                             Defaults to xBD standard mapping.
        """
        self.damage_class_map = damage_class_map or DAMAGE_CLASSES

    def parse(
        self, json_path: Union[str, Path]
    ) -> Tuple[List[Polygon], List[int], List[str]]:
        """
        Parse an xBD JSON label file.

        Args:
            json_path: Path to the JSON label file

        Returns:
            Tuple containing:
                - polygons: List of Shapely Polygon objects (building footprints)
                - damage_classes: List of damage class IDs (0-3)
                - building_uids: List of unique building identifiers
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"Label file not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        polygons = []
        damage_classes = []
        building_uids = []

        # Extract features
        features = data.get("features", {}).get("xy", [])

        if not features:
            logger.warning(f"No features found in {json_path}")
            return polygons, damage_classes, building_uids

        for feature in features:
            try:
                # Parse WKT geometry
                wkt_string = feature.get("wkt", "")
                if not wkt_string:
                    continue

                polygon = wkt.loads(wkt_string)

                # Validate polygon
                if not isinstance(polygon, Polygon) or not polygon.is_valid:
                    logger.warning(f"Invalid polygon in {json_path}: {wkt_string[:50]}...")
                    continue

                # Extract properties
                properties = feature.get("properties", {})
                subtype = properties.get("subtype", "no-damage")
                uid = properties.get("uid", f"building_{len(building_uids)}")

                # Map damage class
                damage_class = self.damage_class_map.get(subtype, 0)

                polygons.append(polygon)
                damage_classes.append(damage_class)
                building_uids.append(uid)

            except Exception as e:
                logger.warning(f"Error parsing feature in {json_path}: {e}")
                continue

        logger.info(
            f"Parsed {len(polygons)} buildings from {json_path.name} "
            f"(damage distribution: {self._get_damage_distribution(damage_classes)})"
        )

        return polygons, damage_classes, building_uids

    def _get_damage_distribution(self, damage_classes: List[int]) -> Dict[str, int]:
        """Get distribution of damage classes."""
        distribution = {name: 0 for name in DAMAGE_CLASS_NAMES}
        for damage_class in damage_classes:
            if 0 <= damage_class < len(DAMAGE_CLASS_NAMES):
                distribution[DAMAGE_CLASS_NAMES[damage_class]] += 1
        return distribution

    @staticmethod
    def get_image_dimensions(json_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Extract image dimensions from JSON metadata.

        Args:
            json_path: Path to JSON label file

        Returns:
            Tuple of (width, height) in pixels
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        width = metadata.get("width", 1024)  # xBD default is 1024x1024
        height = metadata.get("height", 1024)

        return width, height
