"""
Loss functions for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Good for handling class imbalance.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices

        Returns:
            Dice loss
        """
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Flatten
        pred = pred.contiguous().view(-1)
        target_one_hot = target_one_hot.contiguous().view(-1)

        # Dice coefficient
        intersection = (pred * target_one_hot).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target_one_hot.sum() + self.smooth
        )

        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses on hard examples by down-weighting easy ones.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combination of Cross Entropy and Dice Loss.

    Works well for segmentation tasks.
    """

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target)
        dice_loss = self.dice(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class BCEDiceLoss(nn.Module):
    """
    Binary Cross Entropy + Dice Loss for binary segmentation.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, H, W) - logits
            target: Ground truth (B, H, W) - binary 0/1

        Returns:
            Combined loss
        """
        # BCE loss
        pred_sigmoid = torch.sigmoid(pred.squeeze(1))
        bce_loss = F.binary_cross_entropy(pred_sigmoid, target.float())

        # Dice loss
        smooth = 1.0
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1).float()

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        dice_loss = 1 - dice

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_loss(task: str, loss_type: str = "combined"):
    """
    Factory function to get loss by task and type.

    Args:
        task: 'localization' or 'damage'
        loss_type: 'ce', 'dice', 'focal', 'combined'

    Returns:
        Loss function
    """
    if task == "localization":
        # Binary segmentation
        if loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_type == "combined":
            return BCEDiceLoss()
        else:
            raise ValueError(f"Unknown loss type for localization: {loss_type}")

    elif task == "damage":
        # Multi-class segmentation
        if loss_type == "ce":
            return nn.CrossEntropyLoss()
        elif loss_type == "dice":
            return DiceLoss()
        elif loss_type == "focal":
            return FocalLoss()
        elif loss_type == "combined":
            return CombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    else:
        raise ValueError(f"Unknown task: {task}")
