#!/usr/bin/env python
"""
Analyze damage class distribution across dataset.

Prints pixel counts and percentages for each damage level.

Usage:
    python scripts/analyze_distribution.py --split train
    python scripts/analyze_distribution.py --split val --task damage_classification
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xbd_damage_assessment.data.dataset import xBDDataset


def analyze_localization_distribution(dataset):
    """Analyze building vs background distribution."""
    print(f"\n{'='*70}")
    print("BUILDING LOCALIZATION CLASS DISTRIBUTION")
    print(f"{'='*70}\n")

    total_pixels = 0
    building_pixels = 0

    print(f"Analyzing {len(dataset)} samples...")

    for i in tqdm(range(len(dataset)), desc="Processing"):
        sample = dataset[i]
        mask = sample["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        total_pixels += mask.size
        building_pixels += (mask == 1).sum()

    background_pixels = total_pixels - building_pixels

    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"\nClass distribution:")
    print(f"  Background:  {background_pixels:12,} pixels ({background_pixels/total_pixels*100:5.2f}%)")
    print(f"  Building:    {building_pixels:12,} pixels ({building_pixels/total_pixels*100:5.2f}%)")

    print(f"\nClass balance ratio: 1:{background_pixels/building_pixels:.1f} (building:background)")
    print(f"{'='*70}\n")


def analyze_damage_distribution(dataset):
    """Analyze damage class distribution."""
    print(f"\n{'='*70}")
    print("DAMAGE CLASSIFICATION CLASS DISTRIBUTION")
    print(f"{'='*70}\n")

    damage_names = {
        0: "No damage",
        1: "Minor damage",
        2: "Major damage",
        3: "Destroyed"
    }

    total_pixels = 0
    class_counts = {i: 0 for i in range(4)}

    print(f"Analyzing {len(dataset)} samples...")

    for i in tqdm(range(len(dataset)), desc="Processing"):
        sample = dataset[i]
        mask = sample["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        total_pixels += mask.size

        for damage_class in range(4):
            class_counts[damage_class] += (mask == damage_class).sum()

    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"\nDamage class distribution:")

    for damage_class in range(4):
        count = class_counts[damage_class]
        pct = count / total_pixels * 100
        name = damage_names[damage_class]
        print(f"  Class {damage_class} ({name:15s}): {count:12,} pixels ({pct:5.2f}%)")

    # Calculate imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    print(f"\nClass imbalance ratio: 1:{max_count/min_count:.1f} (most:least common)")

    # Building pixels (anything not class 0)
    building_pixels = sum(class_counts[i] for i in range(4)) - class_counts[0]
    print(f"\nTotal building pixels: {building_pixels:,} ({building_pixels/total_pixels*100:.2f}%)")
    print(f"Background pixels: {class_counts[0]:,} ({class_counts[0]/total_pixels*100:.2f}%)")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze class distribution")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--task", type=str, default="localization",
                       choices=["localization", "damage_classification"])

    args = parser.parse_args()

    # Load dataset (no transforms to get raw masks)
    dataset = xBDDataset(
        data_root=args.data_root,
        split=args.split,
        task=args.task,
        transform=None,  # No transforms - we want raw masks
    )

    print(f"\nDataset: {args.split} split, {args.task} task")
    print(f"Samples: {len(dataset)}")

    if len(dataset) == 0:
        print("\nERROR: Dataset is empty!")
        return

    # Analyze
    if args.task == "localization":
        analyze_localization_distribution(dataset)
    else:
        analyze_damage_distribution(dataset)


if __name__ == "__main__":
    main()
