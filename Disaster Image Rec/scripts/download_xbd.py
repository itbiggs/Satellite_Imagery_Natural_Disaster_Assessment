#!/usr/bin/env python
"""
Download specific xBD disaster events for training.

Targets small, class-balanced events suitable for rapid prototyping:
- pinery-bushfire: ~75 images
- lower-puna-volcano: ~120 images
- santa-rosa-wildfire: ~100 images
- tuscaloosa-tornado: ~60 images

Total: ~355 images, ~1-2GB

Usage:
    python scripts/download_xbd.py --output data/raw/xbd
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
import sys

# xBD dataset is hosted at multiple locations
# Primary: xView2 challenge (requires registration)
# Alternative: Assume user has downloaded the full dataset and we'll extract specific events

# For this script, we'll provide two modes:
# 1. Extract from local full xBD download
# 2. Download from public mirrors (if available)

TARGET_EVENTS = [
    "pinery-bushfire",
    "lower-puna-volcano",
    "santa-rosa-wildfire",
    "tuscaloosa-tornado"
]


def print_instructions():
    """Print download instructions for xBD dataset."""
    print("\n" + "="*70)
    print("xBD DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nThe xBD dataset requires registration at https://xview2.org/")
    print("\nOption 1: Download from xView2 (RECOMMENDED)")
    print("-" * 70)
    print("1. Go to https://xview2.org/dataset")
    print("2. Register/login to access the dataset")
    print("3. Download 'train.tar.gz' (~7.8GB)")
    print("4. Extract and point this script to the directory")
    print("\nOption 2: Download from Kaggle mirror")
    print("-" * 70)
    print("1. Install kaggle CLI: pip install kaggle")
    print("2. Setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
    print("3. Run: kaggle competitions download -c xview2-challenge")
    print("\nOption 3: Use AWS CLI (if you have S3 access)")
    print("-" * 70)
    print("The dataset may be available on AWS Open Data Registry")
    print("Check: https://registry.opendata.aws/")
    print("\n" + "="*70)
    print()


def extract_events_from_full_dataset(xbd_root: Path, output_dir: Path):
    """
    Extract specific disaster events from full xBD dataset.

    Args:
        xbd_root: Path to full xBD dataset (should contain train/images, train/labels)
        output_dir: Where to save extracted events
    """
    xbd_train_images = xbd_root / "train" / "images"
    xbd_train_labels = xbd_root / "train" / "labels"

    if not xbd_train_images.exists() or not xbd_train_labels.exists():
        print(f"ERROR: xBD dataset not found at {xbd_root}")
        print(f"Expected structure: {xbd_root}/train/images/ and {xbd_root}/train/labels/")
        print_instructions()
        return False

    # Create output directories
    out_images = output_dir / "train" / "images"
    out_labels = output_dir / "train" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"\nSearching for target events in {xbd_train_images}")
    print(f"Target events: {', '.join(TARGET_EVENTS)}\n")

    copied_count = 0
    event_counts = {event: 0 for event in TARGET_EVENTS}

    # Find all images
    all_images = list(xbd_train_images.glob("*.png"))
    print(f"Found {len(all_images)} total images in dataset")

    for img_path in all_images:
        # Check if this image belongs to any target event
        img_name = img_path.stem

        # xBD naming: {disaster-name}_{image-id}_pre_disaster.png
        for event in TARGET_EVENTS:
            if event in img_name:
                # Copy image
                shutil.copy2(img_path, out_images / img_path.name)

                # Find and copy corresponding label
                # Label naming: {disaster-name}_{image-id}_pre_disaster.json
                label_name = img_name.replace("_disaster", "_disaster") + ".json"
                label_path = xbd_train_labels / label_name

                if label_path.exists():
                    shutil.copy2(label_path, out_labels / label_path.name)
                else:
                    # Try alternative naming
                    label_path = xbd_train_labels / (img_name + ".json")
                    if label_path.exists():
                        shutil.copy2(label_path, out_labels / label_path.name)

                copied_count += 1
                event_counts[event] += 1
                break

    print(f"\n✓ Extracted {copied_count} images")
    print("\nPer-event breakdown:")
    for event, count in event_counts.items():
        print(f"  {event:30s}: {count:4d} images")

    if copied_count == 0:
        print("\nWARNING: No images found for target events!")
        print("The xBD dataset may use different naming conventions.")
        print("Available disasters in your dataset:")

        # Show available disasters
        disasters = set()
        for img in all_images[:50]:  # Sample first 50
            parts = img.stem.split("_")
            if len(parts) >= 2:
                disaster = "_".join(parts[:-2])  # Everything before last 2 parts
                disasters.add(disaster)

        print(", ".join(sorted(disasters)))
        return False

    return True


def analyze_class_distribution(output_dir: Path):
    """
    Analyze damage class distribution in downloaded events.
    """
    from collections import Counter

    labels_dir = output_dir / "train" / "labels"
    if not labels_dir.exists():
        return

    print("\n" + "="*70)
    print("DAMAGE CLASS DISTRIBUTION ANALYSIS")
    print("="*70)

    damage_counts = Counter()
    total_buildings = 0

    for label_file in labels_dir.glob("*_pre_disaster.json"):
        try:
            with open(label_file) as f:
                data = json.load(f)

            for feature in data.get("features", {}).get("xy", []):
                damage_type = feature.get("properties", {}).get("subtype", "no-damage")
                damage_counts[damage_type] += 1
                total_buildings += 1
        except Exception as e:
            print(f"Warning: Could not parse {label_file.name}: {e}")

    print(f"\nTotal buildings: {total_buildings}")
    print(f"\nDamage distribution:")

    for damage_class in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
        count = damage_counts.get(damage_class, 0)
        pct = (count / total_buildings * 100) if total_buildings > 0 else 0
        print(f"  {damage_class:15s}: {count:6d} ({pct:5.1f}%)")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Download small, class-balanced xBD disaster events"
    )
    parser.add_argument(
        "--xbd-path",
        type=str,
        help="Path to full xBD dataset (if already downloaded)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/xbd_selected",
        help="Output directory for extracted events"
    )
    parser.add_argument(
        "--show-instructions",
        action="store_true",
        help="Show download instructions and exit"
    )

    args = parser.parse_args()

    if args.show_instructions:
        print_instructions()
        return

    output_dir = Path(args.output)

    if args.xbd_path:
        # Extract from existing dataset
        xbd_root = Path(args.xbd_path)
        print(f"\nExtracting events from {xbd_root}")
        success = extract_events_from_full_dataset(xbd_root, output_dir)

        if success:
            analyze_class_distribution(output_dir)
            print(f"\n✓ Dataset ready at: {output_dir}")
            print(f"\nNext steps:")
            print(f"  1. Preprocess: python scripts/preprocess_xbd.py --input {output_dir}")
            print(f"  2. Visualize: python scripts/visualize_samples.py")
            print(f"  3. Train: python scripts/train.py --config configs/train_localization.yaml")
        else:
            print("\nFailed to extract events. See instructions above.")
    else:
        # No dataset provided - show instructions
        print("\nERROR: No xBD dataset path provided")
        print_instructions()
        print("\nUsage:")
        print(f"  python {sys.argv[0]} --xbd-path /path/to/xbd/dataset --output {args.output}")
        print(f"\nOr download full dataset first, then run:")
        print(f"  python {sys.argv[0]} --xbd-path /path/to/xbd/dataset")


if __name__ == "__main__":
    main()
