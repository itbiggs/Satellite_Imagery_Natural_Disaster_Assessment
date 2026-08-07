#!/usr/bin/env python
"""
Generate synthetic sample data that mimics xBD structure.
This allows testing the pipeline without downloading the full dataset.
"""

import json
import random
from pathlib import Path
import numpy as np
import cv2
from shapely.geometry import Polygon
import argparse


def generate_random_building_polygon(img_size=1024, min_size=30, max_size=100):
    """Generate a random rectangular building polygon."""
    width = random.randint(min_size, max_size)
    height = random.randint(min_size, max_size)

    x = random.randint(0, img_size - width)
    y = random.randint(0, img_size - height)

    coords = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
        (x, y)  # Close the polygon
    ]

    return Polygon(coords)


def generate_synthetic_image(img_size=1024, num_buildings=20):
    """Generate a synthetic satellite-like image with buildings."""
    # Create base image (greenish for vegetation)
    image = np.random.randint(40, 80, (img_size, img_size, 3), dtype=np.uint8)
    image[:, :, 1] = np.random.randint(60, 100, (img_size, img_size))  # More green

    # Add some texture
    noise = np.random.randint(-20, 20, (img_size, img_size, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    buildings = []
    for _ in range(num_buildings):
        polygon = generate_random_building_polygon(img_size)
        buildings.append(polygon)

        # Draw building on image (gray/white rectangles)
        coords = np.array(polygon.exterior.coords, dtype=np.int32)
        color = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
        cv2.fillPoly(image, [coords], color)

        # Add some shadows/detail
        shadow_offset = 2
        shadow_coords = coords + [shadow_offset, shadow_offset]
        cv2.polylines(image, [shadow_coords], True, (100, 100, 100), 1)

    return image, buildings


def generate_post_disaster_image(pre_image, buildings, damage_classes):
    """Generate post-disaster image with damage effects."""
    post_image = pre_image.copy()

    for polygon, damage_class in zip(buildings, damage_classes):
        coords = np.array(polygon.exterior.coords, dtype=np.int32)

        if damage_class == 0:  # no-damage
            # Keep the same
            pass
        elif damage_class == 1:  # minor-damage
            # Slight darkening
            mask = np.zeros(pre_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 255)
            post_image[mask > 0] = (post_image[mask > 0] * 0.8).astype(np.uint8)
        elif damage_class == 2:  # major-damage
            # Add cracks/debris (dark patches)
            mask = np.zeros(pre_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 255)
            post_image[mask > 0] = (post_image[mask > 0] * 0.5).astype(np.uint8)
            # Add some dark spots
            centroid = polygon.centroid
            cv2.circle(post_image, (int(centroid.x), int(centroid.y)), 5, (50, 50, 50), -1)
        elif damage_class == 3:  # destroyed
            # Make it very dark/rubble
            mask = np.zeros(pre_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 255)
            post_image[mask > 0] = np.random.randint(20, 60, post_image[mask > 0].shape).astype(np.uint8)

    return post_image


def generate_xbd_json(buildings, damage_classes, disaster_id, img_size=1024):
    """Generate xBD-format JSON label file."""
    features = []

    for i, (polygon, damage_class) in enumerate(zip(buildings, damage_classes)):
        wkt_string = polygon.wkt
        damage_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]

        feature = {
            "wkt": wkt_string,
            "properties": {
                "feature_type": "building",
                "subtype": damage_names[damage_class],
                "uid": f"building_{i:04d}"
            }
        }
        features.append(feature)

    label_data = {
        "features": {
            "xy": features
        },
        "metadata": {
            "width": img_size,
            "height": img_size,
            "disaster": disaster_id,
            "disaster_type": "synthetic"
        }
    }

    return label_data


def generate_sample_dataset(output_dir, num_disasters=10, buildings_per_disaster=20, img_size=1024):
    """Generate a complete sample dataset."""
    output_dir = Path(output_dir)

    # Create directory structure
    images_dir = output_dir / "train" / "images"
    labels_dir = output_dir / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_disasters} synthetic disaster events...")

    for disaster_idx in range(num_disasters):
        disaster_id = f"synthetic_disaster_{disaster_idx:03d}"

        # Generate pre-disaster image and buildings
        pre_image, buildings = generate_synthetic_image(img_size, buildings_per_disaster)

        # Assign random damage classes (more no-damage, fewer destroyed)
        damage_weights = [0.4, 0.3, 0.2, 0.1]  # no, minor, major, destroyed
        damage_classes = random.choices([0, 1, 2, 3], weights=damage_weights, k=len(buildings))

        # Generate post-disaster image
        post_image = generate_post_disaster_image(pre_image, buildings, damage_classes)

        # Save images
        pre_path = images_dir / f"{disaster_id}_pre_disaster.png"
        post_path = images_dir / f"{disaster_id}_post_disaster.png"
        cv2.imwrite(str(pre_path), cv2.cvtColor(pre_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(post_path), cv2.cvtColor(post_image, cv2.COLOR_RGB2BGR))

        # Save label JSON
        label_data = generate_xbd_json(buildings, damage_classes, disaster_id, img_size)
        label_path = labels_dir / f"{disaster_id}_pre_disaster.json"
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)

        if (disaster_idx + 1) % 5 == 0:
            print(f"  Generated {disaster_idx + 1}/{num_disasters} disasters")

    print(f"\n✓ Sample dataset created at: {output_dir}")
    print(f"  - {num_disasters} disaster events")
    print(f"  - {num_disasters * 2} images")
    print(f"  - ~{num_disasters * buildings_per_disaster} building annotations")
    print(f"\nNext steps:")
    print(f"  1. Run preprocessing: python -m xbd_damage_assessment.data.preprocess")
    print(f"  2. Train models: python scripts/train.py")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic xBD sample data")
    parser.add_argument("--output", type=str, default="data/raw/xbd",
                       help="Output directory for sample data")
    parser.add_argument("--num-disasters", type=int, default=20,
                       help="Number of synthetic disaster events")
    parser.add_argument("--buildings", type=int, default=15,
                       help="Number of buildings per disaster")
    parser.add_argument("--size", type=int, default=1024,
                       help="Image size (square)")
    args = parser.parse_args()

    generate_sample_dataset(
        args.output,
        num_disasters=args.num_disasters,
        buildings_per_disaster=args.buildings,
        img_size=args.size
    )


if __name__ == "__main__":
    main()
