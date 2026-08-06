"""
U-Net models for building localization and damage classification.

Uses segmentation_models_pytorch for production-quality architectures.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class BuildingLocalizationModel(nn.Module):
    """
    U-Net for binary building segmentation (localization task).

    Args:
        encoder_name: Encoder backbone (e.g., 'resnet34', 'efficientnet-b0')
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Input channels (3 for RGB)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,  # Binary segmentation
            activation=None,  # We'll apply sigmoid in loss/inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits (B, 1, H, W)
        """
        return self.model(x)


class DamageClassificationModel(nn.Module):
    """
    U-Net for multi-class damage classification.

    Takes pre/post image pair and outputs damage class per pixel.

    Args:
        encoder_name: Encoder backbone
        encoder_weights: Pretrained weights
        num_classes: Number of damage classes (default 4)
        fusion: How to combine pre/post ('concat' or 'diff')
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        num_classes: int = 4,
        fusion: str = "concat",
    ):
        super().__init__()

        self.fusion = fusion

        if fusion == "concat":
            # Concatenate pre/post images (6 channels input)
            in_channels = 6
        elif fusion == "diff":
            # Use difference image (3 channels)
            in_channels = 3
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if fusion == "diff" else None,  # Can't use imagenet with 6 channels
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre/post image pair.

        Args:
            pre: Pre-disaster image (B, 3, H, W)
            post: Post-disaster image (B, 3, H, W)

        Returns:
            Logits (B, num_classes, H, W)
        """
        if self.fusion == "concat":
            x = torch.cat([pre, post], dim=1)  # (B, 6, H, W)
        elif self.fusion == "diff":
            x = post - pre  # (B, 3, H, W)

        return self.model(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for change detection (alternative architecture).

    Processes pre/post images through shared encoder, then fuses features.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        num_classes: int = 4,
    ):
        super().__init__()

        # Shared encoder for both images
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        # U-Net decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Siamese architecture.

        Args:
            pre: Pre-disaster image (B, 3, H, W)
            post: Post-disaster image (B, 3, H, W)

        Returns:
            Logits (B, num_classes, H, W)
        """
        # Encode both images through shared weights
        features_pre = self.encoder(pre)
        features_post = self.encoder(post)

        # Compute difference features at each level
        features_diff = [
            post_feat - pre_feat
            for pre_feat, post_feat in zip(features_pre, features_post)
        ]

        # Decode difference features
        decoder_output = self.decoder(*features_diff)

        # Segmentation head
        masks = self.segmentation_head(decoder_output)

        return masks


def get_model(task: str, **kwargs):
    """
    Factory function to get model by task name.

    Args:
        task: 'localization' or 'damage'
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    if task == "localization":
        return BuildingLocalizationModel(**kwargs)
    elif task == "damage":
        architecture = kwargs.pop("architecture", "concat")  # or 'siamese'
        if architecture == "siamese":
            return SiameseUNet(**kwargs)
        else:
            return DamageClassificationModel(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
