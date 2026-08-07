#!/usr/bin/env python
"""
Generate simple synthetic sample data using only numpy and PIL.
Minimal dependencies version.
"""

import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import argparse


def generate_random_building(img_size=1024, min_size=30, max_size=100):
    """Generate a random rectangular building."""
    width = random.randint(min_size, max_size)
    height = random.randint(min_size, max_size)
    x = random.randint(0, img_size - width)
    y = random.randint(0, img_size - height)
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def create_wkt_polygon(coords):
    """Create WKT polygon string from coordinates."""
    coords_str = ", ".join([f"{x} {y}" for x, y in coords])
    return f"POLYGON (({coords_str}, {coords[0][0]} {coords[0][1]}))"


def generate_synthetic_image(img_size=1024, num_buildings=15):
    """Generate a synthetic satellite-like image."""
    # Base image - greenish background
    image = np.random.randint(50, 90, (img_size, img_size, 3), dtype=np.uint8)
    image[:, :, 1] += 20  # More green

    # Convert to PIL for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    buildings = []
    for _ in range(num_buildings):
        coords = generate_random_building(img_size)
        buildings.append(coords)

        # Draw building (light gray)
        color = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
        draw.polygon(coords, fill=color, outline=(150, 150, 150))

    return np.array(pil_image), buildings


def generate_post_disaster_image(pre_image, buildings, damage_classes):
    """Generate post-disaster image with damage effects."""
    post_image = pre_image.copy()
    pil_image = Image.fromarray(post_image)
    draw = ImageDraw.Draw(pil_image)

    for coords, damage_class in zip(buildings, damage_classes):
        if damage_class == 0:  # no-damage
            pass
        elif damage_class == 1:  # minor-damage
            # Slightly darker
            draw.polygon(coords, fill=(140, 140, 140))
        elif damage_class == 2:  # major-damage
            # Much darker with some pattern
            draw.polygon(coords, fill=(80, 80, 80))
        elif damage_class == 3:  # destroyed
            # Very dark/rubble
            draw.polygon(coords, fill=(40, 40, 40))

    return np.array(pil_image)


def generate_xbd_json(buildings, damage_classes, disaster_id, img_size=1024):
    """Generate xBD-format JSON label file."""
    features = []
    damage_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]

    for i, (coords, damage_class) in enumerate(zip(buildings, damage_classes)):
        wkt_string = create_wkt_polygon(coords)

        feature = {
            "wkt": wkt_string,
            "properties": {
                "feature_type": "building",
                "subtype": damage_names[damage_class],
                "uid": f"building_{i:04d}"
            }
        }
        features.append(feature)

    return {
        "features": {"xy": features},
        "metadata": {
            "width": img_size,
            "height": img_size,
            "disaster": disaster_id,
            "disaster_type": "synthetic"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic xBD sample data")
    parser.add_argument("--output", type=str, default="data/raw/xbd")
    parser.add_argument("--num-disasters", type=int, default=20)
    parser.add_argument("--buildings", type=int, default=15)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    output_dir = Path(args.output)
    images_dir = output_dir / "train" / "images"
    labels_dir = output_dir / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_disasters} synthetic disaster events...")

    for disaster_idx in range(args.num_disasters):
        disaster_id = f"synthetic_{disaster_idx:03d}"

        # Generate images and buildings
        pre_image, buildings = generate_synthetic_image(args.size, args.buildings)

        # Random damage classes (weighted towards less damage)
        damage_classes = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1], k=len(buildings))

        # Generate post image
        post_image = generate_post_disaster_image(pre_image, buildings, damage_classes)

        # Save images
        Image.fromarray(pre_image).save(images_dir / f"{disaster_id}_pre_disaster.png")
        Image.fromarray(post_image).save(images_dir / f"{disaster_id}_post_disaster.png")

        # Save JSON
        label_data = generate_xbd_json(buildings, damage_classes, disaster_id, args.size)
        with open(labels_dir / f"{disaster_id}_pre_disaster.json", 'w') as f:
            json.dump(label_data, f, indent=2)

        if (disaster_idx + 1) % 5 == 0:
            print(f"  Generated {disaster_idx + 1}/{args.num_disasters}")

    print(f"\n✓ Sample dataset created at: {output_dir}")
    print(f"  - {args.num_disasters} disasters")
    print(f"  - ~{args.num_disasters * args.buildings} buildings")
    print(f"\nNext: python -m xbd_damage_assessment.data.preprocess")


if __name__ == "__main__":
    main()
