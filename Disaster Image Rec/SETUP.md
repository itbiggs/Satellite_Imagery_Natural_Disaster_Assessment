# Setup Instructions

Complete setup guide for the xBD Damage Assessment project.

## Prerequisites

- **Python**: 3.9, 3.10, or 3.11
- **Operating System**: macOS, Linux, or Windows
- **Disk Space**: ~100GB for xBD dataset + processed data
- **Optional**: CUDA-capable GPU for training (not required for development)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/xbd-damage-assessment.git
cd xbd-damage-assessment
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n xbd python=3.10
conda activate xbd
```

### 3. Install Dependencies

**Option A: Quick Install (pip + requirements.txt)**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

**Option B: Development Install (editable mode)**
```bash
pip install --upgrade pip
pip install -e .  # Install package in editable mode
pip install -e ".[dev]"  # Include dev dependencies
```

**For GPU support (PyTorch with CUDA):**

If you have an NVIDIA GPU and want to use it for training, install PyTorch with CUDA:

```bash
# CUDA 11.8 (check your CUDA version first)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Then install remaining dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

Run the smoke test to verify all dependencies are installed correctly:

```bash
python scripts/smoke_test.py
```

You should see all checks passing. If any fail, install the missing dependencies.

### 5. Download xBD Dataset

1. **Register** at [xView2 Challenge](https://xview2.org/)
2. **Download** the dataset:
   - Training data
   - (Optional) Test and Tier3 data
3. **Extract** to `data/raw/xbd/`

Expected directory structure:
```
data/raw/xbd/
├── train/
│   ├── images/
│   │   ├── guatemala-volcano_00000000_pre_disaster.png
│   │   ├── guatemala-volcano_00000000_post_disaster.png
│   │   └── ...
│   └── labels/
│       ├── guatemala-volcano_00000000_pre_disaster.json
│       └── ...
├── test/  (optional)
└── tier3/  (optional)
```

### 6. Configure Environment

Create a `.env` file in the project root:

```bash
# Weights & Biases (optional - for experiment tracking)
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=xbd-damage-assessment
WANDB_ENTITY=your_username

# Data paths (optional - defaults work if you followed step 5)
XBD_DATA_ROOT=data/raw/xbd
```

## Running Tests

After installation, verify everything works:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/xbd_damage_assessment --cov-report=html

# Run specific test file
pytest tests/unit/test_label_parser.py -v
```

## Data Preprocessing

Once the dataset is downloaded, preprocess it into model-ready format:

```bash
python -m xbd_damage_assessment.data.preprocess \
    --config configs/preprocess.yaml
```

This will:
- Parse JSON labels
- Rasterize building polygons to masks
- Tile images into 512x512 patches
- Split into train/val/test sets
- Save to `data/processed/`

**Configuration**: Edit `configs/preprocess.yaml` to customize tiling parameters, split ratios, etc.

## Development Workflow

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pre-commit install
pre-commit run --all-files
```

## Troubleshooting

### ImportError: No module named 'xbd_damage_assessment'

Add the src directory to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Or install in editable mode: `pip install -e .`

### PyTorch not detecting GPU

Check CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with the correct CUDA version for your system.

### Rasterio installation fails

On macOS, you may need GDAL:
```bash
brew install gdal
pip install rasterio
```

On Ubuntu/Debian:
```bash
sudo apt-get install libgdal-dev
pip install rasterio
```

### Out of memory during preprocessing

Reduce `num_workers` in `configs/preprocess.yaml` or process fewer disasters at a time.

## Next Steps

After setup:

1. **Explore the data**: Check `notebooks/` for exploratory notebooks
2. **Run preprocessing**: Convert raw xBD data to model-ready format
3. **Train models**: (Coming in next session)
4. **Run inference**: Test on new imagery
5. **Deploy**: Set up FastAPI backend and web interface

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/xbd-damage-assessment/issues)
- **xView2 Dataset**: [Official Documentation](https://xview2.org/)
- **PyTorch**: [Documentation](https://pytorch.org/docs/)

---

Ready to build! 🚀
