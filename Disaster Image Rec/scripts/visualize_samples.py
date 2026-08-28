#!/usr/bin/env python
"""
Sanity-check visualization: render 5 random samples with mask overlays.

Verifies that images and masks are correctly aligned.

Usage:
    python scripts/visualize_samples.py --split train --task localization --num-samples 5
    python scripts/visualize_samples.py --split val --task damage_classification --num-samples 5
"""

import argparse
import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader

from xbd_damage_assessment.data.dataset import xBDDataset


def visualize_localization_sample(sample, ax):
    """Visualize building localization sample (pre-disaster image + building mask)."""
    # Get image and mask
    image = sample["image"]  # (C, H, W)
    mask = sample["mask"]    # (H, W)
    tile_id = sample["tile_id"]

    # Convert image tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        # Denormalize if normalized
        if image.min() < 0:  # Likely normalized
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
        image = np.clip(image, 0, 1)

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Create overlay: buildings in green
    overlay = image.copy()
    building_pixels = mask == 1
    overlay[building_pixels] = overlay[building_pixels] * 0.5 + np.array([0, 1, 0]) * 0.5

    ax.imshow(overlay)
    ax.set_title(f"Localization: {tile_id}\n{building_pixels.sum()} building pixels",
                 fontsize=10)
    ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.5, label='Buildings')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def visualize_damage_sample(sample, ax_pre, ax_post, ax_mask):
    """Visualize damage classification sample (pre/post images + damage mask)."""
    image_pre = sample["image_pre"]   # (C, H, W)
    image_post = sample["image_post"] # (C, H, W)
    mask = sample["mask"]             # (H, W)
    tile_id = sample["tile_id"]

    # Convert tensors to numpy
    def tensor_to_image(tensor):
        if isinstance(tensor, torch.Tensor):
            img = tensor.permute(1, 2, 0).cpu().numpy()
            if img.min() < 0:  # Normalized
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
            return np.clip(img, 0, 1)
        return tensor

    image_pre = tensor_to_image(image_pre)
    image_post = tensor_to_image(image_post)

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Show pre-disaster
    ax_pre.imshow(image_pre)
    ax_pre.set_title(f"Pre-disaster\n{tile_id}", fontsize=9)
    ax_pre.axis("off")

    # Show post-disaster
    ax_post.imshow(image_post)
    ax_post.set_title(f"Post-disaster", fontsize=9)
    ax_post.axis("off")

    # Show damage mask with color coding
    damage_colors = {
        0: [0.2, 0.2, 0.2],      # no-damage: dark gray
        1: [1.0, 1.0, 0.0],      # minor-damage: yellow
        2: [1.0, 0.5, 0.0],      # major-damage: orange
        3: [1.0, 0.0, 0.0],      # destroyed: red
    }

    # Create colored mask
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3))
    for damage_class, color in damage_colors.items():
        colored_mask[mask == damage_class] = color

    ax_mask.imshow(colored_mask)
    ax_mask.set_title(f"Damage mask", fontsize=9)
    ax_mask.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color=damage_colors[0], label=f'No damage ({(mask==0).sum()})'),
        mpatches.Patch(color=damage_colors[1], label=f'Minor ({(mask==1).sum()})'),
        mpatches.Patch(color=damage_colors[2], label=f'Major ({(mask==2).sum()})'),
        mpatches.Patch(color=damage_colors[3], label=f'Destroyed ({(mask==3).sum()})'),
    ]
    ax_mask.legend(handles=legend_elements, loc='upper right', fontsize=7)


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--data-root", type=str, default="data", help="Data root directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--task", type=str, default="localization",
                       choices=["localization", "damage_classification"])
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output", type=str, default="outputs/visualizations/samples.png")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*70}")
    print(f"DATASET SANITY CHECK VISUALIZATION")
    print(f"{'='*70}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.num_samples}")
    print()

    # Load dataset
    dataset = xBDDataset(
        data_root=args.data_root,
        split=args.split,
        task=args.task,
        transform=xBDDataset.get_default_transform(args.split, image_size=512),
    )

    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        print(f"Check that data exists at: {args.data_root}/processed/{args.split}/")
        return

    # Sample random indices
    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    # Create figure
    if args.task == "localization":
        # Single column: pre-disaster image with building mask overlay
        fig, axes = plt.subplots(num_samples, 1, figsize=(8, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for idx, ax in zip(indices, axes):
            sample = dataset[idx]
            visualize_localization_sample(sample, ax)

    else:  # damage_classification
        # Three columns: pre, post, damage mask
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for idx, (ax_pre, ax_post, ax_mask) in zip(indices, axes):
            sample = dataset[idx]
            visualize_damage_sample(sample, ax_pre, ax_post, ax_mask)

    plt.tight_layout()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    # Also show if in interactive environment
    try:
        plt.show(block=False)
        print("\n✓ Visualization displayed (close window to continue)")
    except:
        pass

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
