#!/usr/bin/env python
"""Standalone preprocessing script that works without package installation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import yaml

from xbd_damage_assessment.data.preprocess import xBDPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess xBD dataset")
    parser.add_argument("--config", type=str, default="configs/preprocess.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    preprocessor = xBDPreprocessor(config)
    preprocessor.process_all()


if __name__ == "__main__":
    main()
