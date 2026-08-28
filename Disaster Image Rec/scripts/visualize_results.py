#!/usr/bin/env python
"""
Generate beautiful visualization outputs for README/portfolio.

Creates side-by-side comparisons showing:
- Original image
- Ground truth mask
- Predicted mask
- Overlay visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from xbd_damage_assessment.models import get_model
from xbd_damage_assessment.data.dataset import xBDDataset
from xbd_damage_assessment.utils.io import ensure_dir


# Color maps for visualization
DAMAGE_COLORS = {
    0: (255, 255, 0),    # no-damage - yellow
    1: (255, 165, 0),    # minor-damage - orange
    2: (255, 69, 0),     # major-damage - red-orange
    3: (139, 0, 0),      # destroyed - dark red
}

DAMAGE_NAMES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]


def mask_to_rgb(mask, colors=DAMAGE_COLORS):
    """Convert class mask to RGB visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        rgb[mask == class_id] = color

    return rgb


def overlay_mask_on_image(image, mask, alpha=0.5, colors=DAMAGE_COLORS):
    """Overlay colored mask on image."""
    mask_rgb = mask_to_rgb(mask, colors)

    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Blend
    overlay = (image * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return overlay


def create_comparison_image(pre_image, post_image, gt_mask, pred_mask, damage_names=DAMAGE_NAMES):
    """
    Create a 2x2 grid comparison image.

    Layout:
    [Pre Image]  [Post Image]
    [GT Overlay] [Pred Overlay]
    """
    # Convert torch tensors to numpy
    if isinstance(pre_image, torch.Tensor):
        pre_image = pre_image.cpu().permute(1, 2, 0).numpy()
        pre_image = (pre_image * 255).astype(np.uint8)

    if isinstance(post_image, torch.Tensor):
        post_image = post_image.cpu().permute(1, 2, 0).numpy()
        post_image = (post_image * 255).astype(np.uint8)

    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    # Create overlays
    gt_overlay = overlay_mask_on_image(post_image.copy(), gt_mask)
    pred_overlay = overlay_mask_on_image(post_image.copy(), pred_mask)

    # Resize all to same size
    h, w = pre_image.shape[:2]

    # Create 2x2 grid
    grid = np.zeros((h * 2 + 60, w * 2 + 60, 3), dtype=np.uint8)
    grid.fill(255)  # White background

    # Place images with margins
    margin = 30
    grid[margin:margin+h, margin:margin+w] = pre_image
    grid[margin:margin+h, margin+w+30:margin+2*w+30] = post_image
    grid[margin+h+30:margin+2*h+30, margin:margin+w] = gt_overlay
    grid[margin+h+30:margin+2*h+30, margin+w+30:margin+2*w+30] = pred_overlay

    # Convert to PIL for text
    pil_image = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_image)

    # Add labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    draw.text((margin + 10, 5), "Pre-Disaster", fill=(0, 0, 0), font=font)
    draw.text((margin + w + 40, 5), "Post-Disaster", fill=(0, 0, 0), font=font)
    draw.text((margin + 10, margin + h + 5), "Ground Truth", fill=(0, 0, 0), font=font)
    draw.text((margin + w + 40, margin + h + 5), "Prediction", fill=(0, 0, 0), font=font)

    return np.array(pil_image)


def create_legend(damage_names=DAMAGE_NAMES, colors=DAMAGE_COLORS):
    """Create a legend image."""
    legend = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "Damage Classes:", fill=(0, 0, 0), font=font)

    y = 40
    for class_id, name in enumerate(damage_names):
        color = colors[class_id]
        # Draw color box
        draw.rectangle([10, y, 30, y+15], fill=color)
        # Draw text
        draw.text((40, y), name, fill=(0, 0, 0), font=font)
        y += 30

    return np.array(legend)


@torch.no_grad()
def generate_visualizations(checkpoint_path, output_dir, num_samples=5):
    """Generate visualization images from trained model."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    task = config["task"]

    print(f"Task: {task}")
    print(f"Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")

    # Create model
    model = get_model(
        task=task,
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,  # Loading from checkpoint
        num_classes=config["model"].get("num_classes", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test dataset
    test_dataset = xBDDataset(
        data_root=config["data"]["root"],
        split="test",
        task=task,
        transform=xBDDataset.get_default_transform("val", config["data"]["image_size"]),
    )

    print(f"Test set: {len(test_dataset)} samples")

    # Create output directory
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Generate samples
    for i in range(min(num_samples, len(test_dataset))):
        print(f"\nGenerating visualization {i+1}/{num_samples}")
        sample = test_dataset[i]

        if task == "localization":
            image = sample["image"].unsqueeze(0)
            gt_mask = sample["mask"]

            # Predict
            pred = model(image)
            pred_mask = (torch.sigmoid(pred.squeeze(0).squeeze(0)) > 0.5).long()

            # Simple visualization for binary segmentation
            # TODO: Implement binary visualization
            continue

        else:  # damage classification
            pre_image = sample["image_pre"].unsqueeze(0)
            post_image = sample["image_post"].unsqueeze(0)
            gt_mask = sample["mask"]

            # Predict
            pred = model(pre_image, post_image)
            pred_mask = pred.squeeze(0).argmax(0)

            # Create comparison image
            comparison = create_comparison_image(
                sample["image_pre"],
                sample["image_post"],
                gt_mask,
                pred_mask
            )

            # Save
            output_path = output_dir / f"demo_{i+1}.png"
            Image.fromarray(comparison).save(output_path)
            print(f"  Saved: {output_path}")

    # Create legend
    legend = create_legend()
    legend_path = output_dir / "legend.png"
    Image.fromarray(legend).save(legend_path)
    print(f"\n  Saved legend: {legend_path}")

    print(f"\n✓ Generated {num_samples} visualizations in {output_dir}")
    print("\nAdd these images to your README to showcase results!")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization outputs")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/demo",
                       help="Output directory for visualizations")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to visualize")
    args = parser.parse_args()

    generate_visualizations(args.checkpoint, args.output, args.num_samples)


if __name__ == "__main__":
    main()
