#!/usr/bin/env python
"""
Create before/after prediction visualizations for README.

Shows: Input Image | Ground Truth Mask | Model Prediction

Usage:
    python scripts/create_prediction_comparison.py \
        --checkpoint checkpoints/localization/best_localization_model.pth \
        --data-root data/processed \
        --output docs/figures/predictions.png \
        --num-samples 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

from xbd_damage_assessment.data.dataset import xBDDataset
from xbd_damage_assessment.models.unet import BuildingLocalizationModel


def create_prediction_comparison(
    model,
    dataloader,
    device,
    num_samples=5,
    output_path="outputs/prediction_comparison.png"
):
    """Create side-by-side comparison of input, ground truth, and prediction."""
    model.eval()

    # Collect samples
    samples = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(samples) >= num_samples:
                break

            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()

            # Get predictions
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()

            # Store first image from batch
            samples.append({
                'image': images[0].cpu().numpy(),
                'ground_truth': masks[0],
                'prediction': preds[0]
            })

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        # Denormalize image (assuming ImageNet normalization)
        image = sample['image'].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        ground_truth = sample['ground_truth'][0]  # Remove channel dim
        prediction = sample['prediction'][0] > 0.5  # Threshold at 0.5

        # Plot input image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')

        # Plot ground truth
        axes[idx, 1].imshow(image)
        # Overlay ground truth in green
        overlay_gt = np.zeros_like(image)
        overlay_gt[ground_truth == 1] = [0, 1, 0]  # Green for buildings
        axes[idx, 1].imshow(overlay_gt, alpha=0.5)
        axes[idx, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[idx, 1].axis('off')

        # Plot prediction
        axes[idx, 2].imshow(image)
        # Overlay prediction in cyan
        overlay_pred = np.zeros_like(image)
        overlay_pred[prediction == 1] = [0, 1, 1]  # Cyan for predicted buildings
        axes[idx, 2].imshow(overlay_pred, alpha=0.5)
        axes[idx, 2].set_title('Model Prediction', fontsize=14, fontweight='bold')
        axes[idx, 2].axis('off')

    # Add legend
    green_patch = mpatches.Patch(color='green', alpha=0.5, label='Ground Truth Buildings')
    cyan_patch = mpatches.Patch(color='cyan', alpha=0.5, label='Predicted Buildings')
    fig.legend(handles=[green_patch, cyan_patch], loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved prediction comparison to {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create prediction comparison visualizations')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of processed data')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='outputs/prediction_comparison.png')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = BuildingLocalizationModel(encoder_name='resnet18')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✓ Model loaded")

    # Load dataset
    print(f"\nLoading {args.split} dataset from {args.data_root}...")
    dataset = xBDDataset(
        data_root=args.data_root,
        split=args.split,
        task='localization',
        transform=xBDDataset.get_default_transform(args.split, image_size=256)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"✓ Loaded {len(dataset)} images")

    # Create visualizations
    print(f"\nGenerating {args.num_samples} prediction comparisons...")
    create_prediction_comparison(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=args.num_samples,
        output_path=args.output
    )

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
