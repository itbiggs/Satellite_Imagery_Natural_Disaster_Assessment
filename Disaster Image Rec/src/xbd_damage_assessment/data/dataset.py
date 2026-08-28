"""
PyTorch Dataset for xBD disaster damage assessment.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

from .label_parser import xBDLabelParser
from .rasterize import rasterize_building_masks, rasterize_damage_masks

logger = logging.getLogger(__name__)


class xBDDataset(Dataset):
    """
    PyTorch Dataset for xBD disaster damage assessment.

    Supports two tasks:
    1. Building localization: Binary segmentation of building footprints
    2. Damage classification: Multi-class damage assessment from pre/post pairs

    Args:
        data_root: Root directory containing processed data or raw xBD data
        split: Dataset split ('train', 'val', 'test')
        task: Task type ('localization' or 'damage_classification')
        tile_list_path: Optional path to txt file listing tile IDs to use
        transform: Optional albumentations transform
        use_cache: Cache loaded images in memory (faster but uses more RAM)
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        task: str = "localization",
        tile_list_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        use_cache: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.task = task
        self.transform = transform
        self.use_cache = use_cache

        if task not in ["localization", "damage_classification"]:
            raise ValueError(
                f"Task must be 'localization' or 'damage_classification', got '{task}'"
            )

        # Load tile list
        if tile_list_path is not None:
            self.tile_ids = self._load_tile_list(tile_list_path)
        else:
            # Auto-discover tiles in processed directory
            self.tile_ids = self._discover_tiles()

        logger.info(
            f"Initialized xBDDataset: split={split}, task={task}, "
            f"num_samples={len(self.tile_ids)}"
        )

        # Cache
        self.cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _load_tile_list(self, path: Union[str, Path]) -> List[str]:
        """Load list of tile IDs from text file."""
        with open(path, "r") as f:
            tile_ids = [line.strip() for line in f if line.strip()]
        return tile_ids

    def _discover_tiles(self) -> List[str]:
        """Auto-discover processed tiles in data directory."""
        # Look for processed tiles in data_root/processed/{split}/images/
        processed_dir = self.data_root / "processed" / self.split / "images"

        if not processed_dir.exists():
            logger.warning(f"Processed directory not found: {processed_dir}")
            return []

        # Find all image files
        image_files = sorted(processed_dir.glob("*_pre.png"))
        # Remove the "_pre" suffix to get tile IDs
        tile_ids = [f.stem.replace("_pre", "") for f in image_files]

        logger.info(f"Discovered {len(tile_ids)} tiles in {processed_dir}")
        return tile_ids

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample.

        Returns:
            Dictionary containing:
                - 'image': Image tensor (C, H, W)
                - 'mask': Mask tensor (H, W) for localization or (H, W) for damage
                - 'tile_id': Tile identifier string
                - For damage_classification task:
                    - 'image_pre': Pre-disaster image
                    - 'image_post': Post-disaster image
        """
        tile_id = self.tile_ids[idx]

        # Check cache
        if self.use_cache and idx in self.cache:
            sample = self.cache[idx].copy()
        else:
            sample = self._load_sample(tile_id)
            if self.use_cache:
                self.cache[idx] = sample.copy()

        # Apply transforms
        if self.transform is not None:
            sample = self._apply_transform(sample)
        else:
            # Default: convert to tensor
            sample = self._to_tensor(sample)

        sample["tile_id"] = tile_id
        return sample

    def _load_sample(self, tile_id: str) -> Dict[str, np.ndarray]:
        """Load image and mask for a tile."""
        processed_dir = self.data_root / "processed" / self.split

        if self.task == "localization":
            # Load pre-disaster image and building mask
            image_path = processed_dir / "images" / f"{tile_id}_pre.png"
            mask_path = processed_dir / "masks_building" / f"{tile_id}.png"

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not load mask: {mask_path}")

            return {"image": image, "mask": mask}

        elif self.task == "damage_classification":
            # Load pre/post image pair and damage mask
            image_pre_path = processed_dir / "images" / f"{tile_id}_pre.png"
            image_post_path = processed_dir / "images" / f"{tile_id}_post.png"
            mask_path = processed_dir / "masks_damage" / f"{tile_id}.png"

            image_pre = cv2.imread(str(image_pre_path), cv2.IMREAD_COLOR)
            image_pre = cv2.cvtColor(image_pre, cv2.COLOR_BGR2RGB)

            image_post = cv2.imread(str(image_post_path), cv2.IMREAD_COLOR)
            image_post = cv2.cvtColor(image_post, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            return {
                "image_pre": image_pre,
                "image_post": image_post,
                "mask": mask,
            }

    def _apply_transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply albumentations transform."""
        if self.task == "localization":
            transformed = self.transform(image=sample["image"], mask=sample["mask"])
            return {
                "image": transformed["image"],
                "mask": transformed["mask"],
            }
        else:
            # For damage classification, apply same transform to both images
            # Concatenate pre/post for joint augmentation
            h, w = sample["image_pre"].shape[:2]
            combined = np.concatenate([sample["image_pre"], sample["image_post"]], axis=1)

            transformed = self.transform(image=combined, mask=sample["mask"])

            # Split back into pre/post
            combined_transformed = transformed["image"]
            if isinstance(combined_transformed, torch.Tensor):
                # Already converted to tensor by ToTensorV2
                image_pre = combined_transformed[:, :, :w]
                image_post = combined_transformed[:, :, w:]
            else:
                image_pre = combined_transformed[:, :w, :]
                image_post = combined_transformed[:, w:, :]

            return {
                "image_pre": image_pre,
                "image_post": image_post,
                "mask": transformed["mask"],
            }

    def _to_tensor(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        converted = {}

        for key, value in sample.items():
            if "image" in key:
                # Image: (H, W, C) -> (C, H, W), normalize to [0, 1]
                tensor = torch.from_numpy(value).permute(2, 0, 1).float() / 255.0
            else:
                # Mask: (H, W) -> (H, W)
                tensor = torch.from_numpy(value).long()

            converted[key] = tensor

        return converted

    @staticmethod
    def get_default_transform(split: str, image_size: int = 512) -> A.Compose:
        """
        Get default augmentation pipeline.

        Args:
            split: 'train' or 'val'/'test'
            image_size: Target image size

        Returns:
            Albumentations transform
        """
        if split == "train":
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(p=1.0),
                            A.GaussianBlur(p=1.0),
                            A.MotionBlur(p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1.0),
                            A.HueSaturationValue(p=1.0),
                            A.RGBShift(p=1.0),
                        ],
                        p=0.3,
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
