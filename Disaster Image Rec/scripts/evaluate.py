#!/usr/bin/env python
"""
Evaluate trained model on test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/localization/best_localization_model.pth --split test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from xbd_damage_assessment.data.dataset import xBDDataset
from xbd_damage_assessment.models import get_model
from xbd_damage_assessment.training.metrics import MetricsTracker
from xbd_damage_assessment.utils.device import get_device


@torch.no_grad()
def evaluate(model, dataloader, device, task):
    """Evaluate model on dataset."""
    model.eval()
    metrics = MetricsTracker(num_classes=4 if task == "damage" else 2, task=task)

    print(f"\nEvaluating on {len(dataloader.dataset)} samples...")

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        # Move to device
        if task == "localization":
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(images)

        else:  # damage classification
            pre_images = batch["image_pre"].to(device)
            post_images = batch["image_post"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(pre_images, post_images)

        # Update metrics (no loss for eval)
        metrics.update(0, outputs, masks)

    return metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    task = config["task"]
    dataset_task = "damage_classification" if task == "damage" else task

    print(f"Task: {task}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best metric: {checkpoint['best_metric']:.4f}")

    # Device
    device = get_device(prefer_gpu=False)  # Use CPU for eval

    # Load dataset
    test_dataset = xBDDataset(
        data_root=args.data_root,
        split=args.split,
        task=dataset_task,
        transform=xBDDataset.get_default_transform(args.split, config["data"]["image_size"]),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Create model
    model = get_model(
        task=task,
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,  # Don't reload pretrained weights
        num_classes=config["model"].get("num_classes", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Evaluate
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS - {args.split.upper()} SET")
    print(f"{'='*70}")

    metrics = evaluate(model, test_loader, device, task)

    # Print results
    print(f"\nPixel Accuracy: {metrics['pixel_accuracy']:.4f}")

    if task == "damage":
        print(f"Mean IoU:       {metrics['mean_iou']:.4f}")
        print(f"Macro F1:       {metrics['macro_f1']:.4f}")
        print(f"\nPer-class IoU:")
        for i, iou in enumerate(metrics.get('iou_per_class', [])):
            print(f"  Class {i}: {iou:.4f}")
    else:
        print(f"IoU (Building): {metrics['iou']:.4f}")

    print(f"\n{'='*70}")

    # Save results
    results_path = Path(args.checkpoint).parent / f"eval_{args.split}_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Evaluation Results - {args.split.upper()} SET\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Dataset: {args.split}\n")
        f.write(f"Samples: {len(test_loader.dataset)}\n\n")

        f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")

        if task == "damage":
            f.write(f"Mean IoU:       {metrics['mean_iou']:.4f}\n")
            f.write(f"Macro F1:       {metrics['macro_f1']:.4f}\n")
        else:
            f.write(f"IoU (Building): {metrics['iou']:.4f}\n")

    print(f"\n✓ Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
