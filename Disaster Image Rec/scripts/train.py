#!/usr/bin/env python
"""
Simple training script for xBD models.

Usage:
    python scripts/train.py --config configs/train_localization.yaml
    python scripts/train.py --config configs/train_damage.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import yaml
from tqdm import tqdm
import numpy as np

from xbd_damage_assessment.data.dataset import xBDDataset
from xbd_damage_assessment.models import get_model
from xbd_damage_assessment.training.losses import get_loss
from xbd_damage_assessment.training.metrics import MetricsTracker
from xbd_damage_assessment.utils.device import get_device, set_seed
from xbd_damage_assessment.utils.io import ensure_dir


def train_epoch(model, dataloader, criterion, optimizer, device, task):
    """Train for one epoch."""
    model.train()
    metrics = MetricsTracker(num_classes=4 if task == "damage" else 2, task=task)

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        if task == "localization":
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)

        else:  # damage classification
            pre_images = batch["image_pre"].to(device)
            post_images = batch["image_post"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(pre_images, post_images)
            loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        metrics.update(loss.item(), outputs.detach(), masks)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return metrics.compute()


@torch.no_grad()
def validate(model, dataloader, criterion, device, task):
    """Validate model."""
    model.eval()
    metrics = MetricsTracker(num_classes=4 if task == "damage" else 2, task=task)

    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        # Move to device
        if task == "localization":
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)

        else:  # damage classification
            pre_images = batch["image_pre"].to(device)
            post_images = batch["image_post"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(pre_images, post_images)
            loss = criterion(outputs, masks)

        # Update metrics
        metrics.update(loss.item(), outputs, masks)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return metrics.compute()


def train(config):
    """Main training function."""
    # Set seed
    set_seed(config.get("seed", 42))

    # Device
    device = get_device(prefer_gpu=config.get("use_gpu", True))

    # Task
    task = config["task"]
    print(f"\n{'='*60}")
    print(f"Training {task} model")
    print(f"{'='*60}\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset = xBDDataset(
        data_root=config["data"]["root"],
        split="train",
        task=task,
        transform=xBDDataset.get_default_transform("train", config["data"]["image_size"]),
    )

    val_dataset = xBDDataset(
        data_root=config["data"]["root"],
        split="val",
        task=task,
        transform=xBDDataset.get_default_transform("val", config["data"]["image_size"]),
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Create model
    print("Creating model...")
    model = get_model(
        task=task,
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"].get("pretrained", "imagenet"),
        num_classes=config["model"].get("num_classes", 4),
    )
    model = model.to(device)
    print(f"Model: {config['model']['encoder']}-based U-Net\n")

    # Loss
    criterion = get_loss(task, config["training"]["loss"])

    # Optimizer
    if config["training"]["optimizer"] == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 0),
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=config["training"]["lr"],
            momentum=0.9,
            weight_decay=config["training"].get("weight_decay", 0),
        )

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_metric = 0.0
    output_dir = Path(config["output"]["checkpoint_dir"])
    ensure_dir(output_dir)

    print(f"Starting training for {config['training']['epochs']} epochs...\n")

    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, task)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, task)

        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Acc: {train_metrics['pixel_accuracy']:.4f} | Val Acc: {val_metrics['pixel_accuracy']:.4f}")

        if task == "damage":
            print(f"Train mIoU: {train_metrics['mean_iou']:.4f} | Val mIoU: {val_metrics['mean_iou']:.4f}")
            print(f"Train F1: {train_metrics['macro_f1']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")
            metric = val_metrics["mean_iou"]
        else:
            print(f"Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            metric = val_metrics["iou"]

        # Learning rate scheduling
        scheduler.step(metric)

        # Save best model
        if metric > best_metric:
            best_metric = metric
            checkpoint_path = output_dir / f"best_{task}_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"✓ Saved best model (metric: {best_metric:.4f})")

        print()

    print(f"\n{'='*60}")
    print(f"Training complete! Best metric: {best_metric:.4f}")
    print(f"Model saved to: {output_dir / f'best_{task}_model.pth'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train xBD model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Train
    train(config)


if __name__ == "__main__":
    main()
