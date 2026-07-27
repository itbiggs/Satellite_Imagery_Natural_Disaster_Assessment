# xBD Disaster Damage Assessment

End-to-end deep learning pipeline for satellite imagery disaster damage assessment using the xBD/xView2 dataset. Production-quality implementation demonstrating PyTorch, computer vision, geospatial engineering, and full-stack ML deployment.

## Overview

This project implements a two-stage deep learning system for assessing building damage from satellite imagery:

1. **Building Localization** - Semantic segmentation model that identifies building footprints from pre-disaster satellite imagery
2. **Damage Classification** - Change detection model that analyzes pre/post disaster image pairs and classifies building damage on a 4-level scale (no damage, minor, major, destroyed)

The complete pipeline includes:
- Production-quality data preprocessing with geospatial polygon rasterization and intelligent tiling
- PyTorch training with experiment tracking (Weights & Biases)
- Inference on real disaster imagery from Maxar's Open Data Program
- PostGIS spatial database for storing results
- FastAPI backend (Dockerized) serving predictions
- Interactive damage assessment map using ArcGIS Maps SDK for JavaScript

## Project Structure

```
xbd-damage-assessment/
├── src/xbd_damage_assessment/     # Main package
│   ├── data/                      # Data preprocessing & loading
│   ├── models/                    # Model architectures
│   ├── training/                  # Training loops & callbacks
│   ├── inference/                 # Inference pipeline
│   ├── evaluation/                # Metrics & evaluation
│   └── utils/                     # Shared utilities
├── tests/                         # pytest unit & integration tests
│   ├── unit/
│   └── integration/
├── notebooks/                     # Jupyter notebooks for exploration
├── configs/                       # Hydra configuration files
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Original xBD dataset
│   ├── interim/                   # Intermediate processing outputs
│   └── processed/                 # Final processed datasets
├── scripts/                       # Utility scripts
├── docs/                          # Documentation
└── outputs/                       # Model outputs & predictions
```

## Setup

### Prerequisites

- Python 3.9 - 3.11
- CUDA 11.8+ (for GPU training, optional for development)
- ~100GB disk space for xBD dataset

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xbd-damage-assessment.git
cd xbd-damage-assessment
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

**Option A: Using pip with requirements.txt (recommended for quick start)**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

**Option B: Using pyproject.toml (recommended for development)**
```bash
pip install -e .                    # Install package in editable mode
pip install -e ".[dev]"             # Include development dependencies
```

4. Verify installation:
```bash
python scripts/smoke_test.py
```

### Environment Configuration

Create a `.env` file in the project root:
```bash
# Weights & Biases
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=xbd-damage-assessment

# Data paths
XBD_DATA_ROOT=/path/to/xbd/dataset
```

## Data

### xBD/xView2 Dataset

The [xBD dataset](https://xview2.org/) contains ~22k building annotations across 850k km² of pre/post disaster imagery from 19 global disasters.

**Download:**
1. Register at [xView2 Challenge](https://xview2.org/)
2. Download train/test/tier3 sets
3. Extract to `data/raw/xbd/`

**Dataset Structure:**
```
data/raw/xbd/
├── train/
│   ├── images/
│   │   ├── {disaster-id}_pre_disaster.png
│   │   └── {disaster-id}_post_disaster.png
│   └── labels/
│       └── {disaster-id}_pre_disaster.json  # Building polygons + damage labels
├── test/
└── tier3/
```

**Label Format:**
Each JSON contains:
- `features.wkt` - Building footprint polygon in WKT format
- `features.properties.subtype` - Damage class: `no-damage`, `minor-damage`, `major-damage`, `destroyed`

### Data Preprocessing

Process the raw xBD data into model-ready format:

```bash
python -m xbd_damage_assessment.data.preprocess \
    --config configs/preprocess.yaml
```

This will:
1. Parse JSON labels and rasterize building polygons to segmentation masks
2. Create binary building masks (localization task) and 4-class damage masks (classification task)
3. Tile large images into 512x512 patches with configurable overlap
4. Apply train/val split and save to `data/processed/`

**Configuration:** Edit `configs/preprocess.yaml` to adjust:
- Tile size and overlap
- Train/val split ratio
- Damage class mappings
- Output resolution

## Architecture

### Localization Model
- **Architecture:** U-Net with EfficientNet-B4 encoder (pretrained on ImageNet)
- **Task:** Binary segmentation of building footprints
- **Input:** RGB pre-disaster image (512x512)
- **Output:** Binary mask (building vs background)

### Damage Classification Model
- **Architecture:** Siamese U-Net with ResNet50 encoders
- **Task:** 4-class damage classification per pixel
- **Input:** Concatenated pre/post image pair (6 channels, 512x512)
- **Output:** 4-class damage mask (no-damage, minor, major, destroyed)

### Training

*Training implementation coming in next session*

Configuration managed with Hydra. See `configs/train.yaml` for hyperparameters.

## Results

*Results and metrics will be added after training*

### Localization Performance
- IoU: TBD
- Precision/Recall: TBD

### Damage Classification Performance
- Overall Accuracy: TBD
- Per-class F1: TBD
- Confusion Matrix: TBD

## Demo

*Interactive demo deployment coming in future session*

The final deployment includes:
- **Backend:** FastAPI serving predictions, PostGIS for spatial queries
- **Frontend:** ArcGIS Maps SDK for JavaScript with interactive damage visualization
- **Inference:** Real-time damage assessment on Maxar Open Data imagery

## Development

### Running Tests
```bash
pytest tests/                  # Run all tests
pytest tests/unit/             # Unit tests only
pytest --cov                   # With coverage report
```

### Code Quality
```bash
black src/ tests/              # Format code
isort src/ tests/              # Sort imports
flake8 src/ tests/             # Lint
mypy src/                      # Type checking
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## References

- [xView2 Dataset Paper](https://arxiv.org/abs/1911.09296)
- [xBD Dataset](https://xview2.org/)
- [Maxar Open Data Program](https://www.maxar.com/open-data)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)

## License

MIT License - see LICENSE file for details

## Contact

Isaac Biggs - [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

Built with PyTorch, Rasterio, Shapely, and FastAPI.
