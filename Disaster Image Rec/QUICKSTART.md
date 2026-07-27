# Quick Start Guide

Get up and running with the xBD Damage Assessment project in under 10 minutes.

## Prerequisites

- Python 3.9+ installed
- ~100GB free disk space
- (Optional) NVIDIA GPU with CUDA for training

## Installation (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Verify installation
python scripts/smoke_test.py
```

Expected output: All 22 checks should pass ✓

## Download Data (depends on internet speed)

1. **Register** at https://xview2.org/
2. **Download** train.tar.gz (~9GB)
3. **Extract** to `data/raw/xbd/`

Expected structure:
```
data/raw/xbd/
└── train/
    ├── images/
    └── labels/
```

## Preprocess Data (15-30 minutes)

```bash
python -m xbd_damage_assessment.data.preprocess \
    --config configs/preprocess.yaml
```

This creates ~15k-20k tiles in `data/processed/`:
- `train/` - Training tiles (70%)
- `val/` - Validation tiles (15%)
- `test/` - Test tiles (15%)

Each split contains:
- `images/` - Pre/post satellite imagery
- `masks_building/` - Binary building masks
- `masks_damage/` - 4-class damage masks

## Explore Data (5 minutes)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest --cov=src/xbd_damage_assessment --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
# or: xdg-open htmlcov/index.html  # Linux
```

## Project Status

✅ **Complete:**
- Repository structure
- Data preprocessing pipeline
- Unit tests (23 tests)
- Documentation

🔜 **Next Session:**
- Model architectures
- Training pipeline
- Evaluation metrics

## Common Commands

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Install pre-commit hooks
pre-commit install
```

## File Organization

```
├── src/xbd_damage_assessment/    # Main package
│   └── data/                     # ✅ Preprocessing ready
├── tests/                        # ✅ 23 unit tests
├── configs/                      # ✅ preprocess.yaml
├── notebooks/                    # ✅ Starter notebook
└── data/processed/               # Created after preprocessing
```

## Getting Help

- **Setup issues:** See `SETUP.md` for detailed troubleshooting
- **Full docs:** See `README.md`
- **This session:** See `SESSION_SUMMARY.md`

## Next Steps

After preprocessing is complete:

1. **Explore the data:** Use the Jupyter notebook to visualize samples
2. **Check preprocessing output:** Verify tiles in `data/processed/`
3. **Ready for training:** Wait for next session to implement models

---

**Estimated time to working preprocessing pipeline: 30-45 minutes** ⚡
