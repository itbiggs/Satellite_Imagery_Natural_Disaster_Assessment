"""Device and reproducibility utilities."""

import random
import numpy as np
import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Handles CPU vs GPU gracefully for local development vs cloud training.

    Args:
        prefer_gpu: If True and GPU is available, use GPU. Otherwise use CPU.

    Returns:
        torch.device: The device to use for computations
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        if prefer_gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but not available. Falling back to CPU.")

    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")
