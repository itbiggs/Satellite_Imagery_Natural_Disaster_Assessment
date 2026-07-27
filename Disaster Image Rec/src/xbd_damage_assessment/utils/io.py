"""File I/O utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Union
import yaml


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary containing configuration
    """
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save data to a YAML file.

    Args:
        data: Dictionary to save
        filepath: Output path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
