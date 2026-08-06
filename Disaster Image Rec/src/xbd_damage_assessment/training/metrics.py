"""
Evaluation metrics for segmentation tasks.
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = None) -> dict:
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices
        num_classes: Number of classes

    Returns:
        Dictionary with IoU per class and mean IoU
    """
    if num_classes is None:
        num_classes = max(pred.max().item(), target.max().item()) + 1

    ious = []
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            iou = float('nan')  # No ground truth for this class
        else:
            iou = intersection / union

        ious.append(iou)

    # Compute mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    return {
        'iou_per_class': ious,
        'mean_iou': mean_iou
    }


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute pixel-wise accuracy.

    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices

    Returns:
        Pixel accuracy
    """
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def compute_f1_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = None) -> dict:
    """
    Compute F1 score for each class.

    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices
        num_classes: Number of classes

    Returns:
        Dictionary with F1 per class and macro F1
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    if num_classes is None:
        num_classes = max(pred.max(), target.max()) + 1

    precision, recall, f1, support = precision_recall_fscore_support(
        target, pred, labels=list(range(num_classes)), zero_division=0
    )

    return {
        'f1_per_class': f1.tolist(),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'macro_f1': f1.mean(),
        'support_per_class': support.tolist()
    }


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = None) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices
        num_classes: Number of classes

    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    if num_classes is None:
        num_classes = max(pred.max(), target.max()) + 1

    cm = confusion_matrix(target, pred, labels=list(range(num_classes)))
    return cm


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self, num_classes: int, task: str = "damage"):
        self.num_classes = num_classes
        self.task = task
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_preds = []
        self.all_targets = []

    def update(self, loss: float, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with batch results.

        Args:
            loss: Loss value
            pred: Predictions (B, C, H, W) - logits or (B, H, W) - class indices
            target: Ground truth (B, H, W) - class indices
        """
        self.total_loss += loss * target.size(0)
        self.total_samples += target.size(0)

        # Convert logits to class predictions if needed
        if pred.dim() == 4:  # (B, C, H, W)
            if self.task == "localization":
                pred = (torch.sigmoid(pred.squeeze(1)) > 0.5).long()
            else:
                pred = pred.argmax(dim=1)

        self.all_preds.append(pred.cpu())
        self.all_targets.append(target.cpu())

    def compute(self) -> dict:
        """
        Compute final metrics.

        Returns:
            Dictionary of metrics
        """
        avg_loss = self.total_loss / self.total_samples if self.total_samples > 0 else 0

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.all_preds, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)

        # Compute metrics
        metrics = {
            'loss': avg_loss,
            'pixel_accuracy': compute_pixel_accuracy(all_preds, all_targets),
        }

        if self.task == "damage":
            iou_metrics = compute_iou(all_preds, all_targets, self.num_classes)
            f1_metrics = compute_f1_score(all_preds, all_targets, self.num_classes)

            metrics.update({
                'mean_iou': iou_metrics['mean_iou'],
                'iou_per_class': iou_metrics['iou_per_class'],
                'macro_f1': f1_metrics['macro_f1'],
                'f1_per_class': f1_metrics['f1_per_class'],
            })
        else:
            # Binary IoU for localization
            iou_metrics = compute_iou(all_preds, all_targets, num_classes=2)
            metrics['iou'] = iou_metrics['iou_per_class'][1]  # IoU for building class

        return metrics
