# Quick Demo Setup

Get this project running and generating visual results in ~10 minutes.

## Step 1: Install Dependencies (~5 min)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or minimal install for quick demo:
pip install torch torchvision numpy pandas pillow shapely pyyaml tqdm scikit-learn
pip install segmentation-models-pytorch
```

## Step 2: Generate Sample Data (~30 sec)

```bash
# Generate 20 synthetic disaster events
python3 scripts/generate_sample_data_simple.py --num-disasters 20
```

## Step 3: Preprocess Data (~2 min)

```bash
# Run preprocessing pipeline
python3 scripts/preprocess_standalone.py --config configs/preprocess.yaml
```

This creates:
- `data/processed/train/` - Training tiles
- `data/processed/val/` - Validation tiles
- `data/processed/test/` - Test tiles

## Step 4: Train Models (~5-10 min on CPU, ~2 min on GPU)

```bash
# Train localization model (building segmentation)
python3 scripts/train.py --config configs/train_localization.yaml

# Train damage classification model
python3 scripts/train.py --config configs/train_damage.yaml
```

Models saved to `checkpoints/`

## Step 5: Generate Visualizations (~1 min)

```bash
# Run inference and create demo images
python3 scripts/visualize_results.py --checkpoint checkpoints/damage/best_damage_model.pth
```

Creates images in `outputs/demo/` that you can add to README.

## Step 6: Update README

Add generated images to show off your results!

---

**Total time:** ~15-20 minutes from start to visual demo
**GPU recommended** but works on CPU (just slower training)
