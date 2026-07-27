#!/usr/bin/env python
"""
Environment smoke test script.

Verifies that all required dependencies are installed and working correctly.
Run this after setting up your environment to ensure everything is configured properly.
"""

import sys
from typing import List, Tuple


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name}"
    except ImportError as e:
        return False, f"✗ {package_name or module_name}: {e}"


def test_pytorch() -> Tuple[bool, str]:
    """Test PyTorch installation and device availability."""
    try:
        import torch

        version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda
            device = torch.cuda.get_device_name(0)
            return (
                True,
                f"✓ PyTorch {version} with CUDA {cuda_version} ({device})",
            )
        else:
            return True, f"✓ PyTorch {version} (CPU only)"
    except ImportError as e:
        return False, f"✗ PyTorch: {e}"


def test_geospatial() -> List[Tuple[bool, str]]:
    """Test geospatial libraries."""
    results = []

    # Test rasterio
    try:
        import rasterio

        version = rasterio.__version__
        results.append((True, f"✓ rasterio {version}"))
    except ImportError as e:
        results.append((False, f"✗ rasterio: {e}"))

    # Test shapely
    try:
        from shapely.geometry import Polygon
        from shapely import wkt

        # Test basic functionality
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert poly.is_valid
        results.append((True, "✓ shapely"))
    except Exception as e:
        results.append((False, f"✗ shapely: {e}"))

    # Test geopandas
    try:
        import geopandas as gpd

        version = gpd.__version__
        results.append((True, f"✓ geopandas {version}"))
    except ImportError as e:
        results.append((False, f"✗ geopandas: {e}"))

    return results


def test_computer_vision() -> List[Tuple[bool, str]]:
    """Test computer vision libraries."""
    results = []

    # Test OpenCV
    try:
        import cv2

        version = cv2.__version__
        results.append((True, f"✓ opencv {version}"))
    except ImportError as e:
        results.append((False, f"✗ opencv: {e}"))

    # Test albumentations
    try:
        import albumentations as A

        version = A.__version__
        results.append((True, f"✓ albumentations {version}"))
    except ImportError as e:
        results.append((False, f"✗ albumentations: {e}"))

    # Test scikit-image
    try:
        import skimage

        version = skimage.__version__
        results.append((True, f"✓ scikit-image {version}"))
    except ImportError as e:
        results.append((False, f"✗ scikit-image: {e}"))

    return results


def test_ml_ops() -> List[Tuple[bool, str]]:
    """Test ML ops and tracking libraries."""
    results = []

    # Test wandb
    try:
        import wandb

        version = wandb.__version__
        results.append((True, f"✓ wandb {version}"))
    except ImportError as e:
        results.append((False, f"✗ wandb: {e}"))

    # Test pytorch-lightning
    try:
        import pytorch_lightning as pl

        version = pl.__version__
        results.append((True, f"✓ pytorch-lightning {version}"))
    except ImportError as e:
        results.append((False, f"✗ pytorch-lightning: {e}"))

    # Test segmentation-models-pytorch
    try:
        import segmentation_models_pytorch as smp

        version = smp.__version__
        results.append((True, f"✓ segmentation-models-pytorch {version}"))
    except ImportError as e:
        results.append((False, f"✗ segmentation-models-pytorch: {e}"))

    return results


def test_core_libraries() -> List[Tuple[bool, str]]:
    """Test core data science libraries."""
    results = []

    libraries = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("yaml", "PyYAML"),
        ("omegaconf", "omegaconf"),
    ]

    for module, package in libraries:
        success, message = test_import(module, package)
        results.append((success, message))

    return results


def test_package_imports() -> List[Tuple[bool, str]]:
    """Test imports from our package."""
    results = []

    # Add src to path if not already there
    import os
    from pathlib import Path

    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    tests = [
        ("xbd_damage_assessment.data.label_parser", "Label Parser"),
        ("xbd_damage_assessment.data.rasterize", "Rasterization"),
        ("xbd_damage_assessment.data.tiling", "Tiling"),
        ("xbd_damage_assessment.data.dataset", "Dataset"),
        ("xbd_damage_assessment.utils.device", "Device Utils"),
        ("xbd_damage_assessment.utils.io", "I/O Utils"),
    ]

    for module, name in tests:
        success, _ = test_import(module)
        if success:
            results.append((True, f"✓ {name}"))
        else:
            results.append((False, f"✗ {name}"))

    return results


def run_smoke_test() -> bool:
    """Run all smoke tests."""
    print("=" * 70)
    print("xBD Damage Assessment - Environment Smoke Test")
    print("=" * 70)
    print()

    all_results = []

    # Test sections
    sections = [
        ("Core Libraries", test_core_libraries()),
        ("PyTorch", [test_pytorch()]),
        ("Geospatial Libraries", test_geospatial()),
        ("Computer Vision", test_computer_vision()),
        ("ML Ops & Tracking", test_ml_ops()),
        ("Package Imports", test_package_imports()),
    ]

    for section_name, results in sections:
        print(f"{section_name}:")
        print("-" * 70)
        for success, message in results:
            print(f"  {message}")
            all_results.append(success)
        print()

    # Summary
    total = len(all_results)
    passed = sum(all_results)
    failed = total - passed

    print("=" * 70)
    print(f"Summary: {passed}/{total} checks passed")

    if failed > 0:
        print(f"⚠ {failed} checks failed")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("  or")
        print("  pip install -e .")
        return False
    else:
        print("✓ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("  1. Download xBD dataset to data/raw/xbd/")
        print("  2. Run preprocessing: python -m xbd_damage_assessment.data.preprocess")
        print("  3. Start training (in next session)")
        return True


def main():
    """Main entry point."""
    success = run_smoke_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
