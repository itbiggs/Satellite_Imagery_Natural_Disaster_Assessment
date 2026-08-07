# README Results Section Template

Add this section to your README.md after training models and generating visualizations.

---

## Results

### Demo: Damage Assessment in Action

The model successfully identifies and classifies building damage from satellite imagery. Below are example predictions on the test set:

![Damage Assessment Example 1](outputs/demo/demo_1.png)
*Pre-disaster image (top-left), post-disaster (top-right), ground truth overlay (bottom-left), and model prediction (bottom-right)*

![Damage Assessment Example 2](outputs/demo/demo_2.png)

![Damage Assessment Example 3](outputs/demo/demo_3.png)

### Damage Classification Legend

![Legend](outputs/demo/legend.png)

- **Yellow:** No damage - Building intact
- **Orange:** Minor damage - Partial damage visible
- **Red-Orange:** Major damage - Significant structural damage
- **Dark Red:** Destroyed - Building completely destroyed

### Performance Metrics

#### Building Localization Model
- **IoU:** 0.XX (X.X% improvement over baseline)
- **Pixel Accuracy:** XX.X%
- **Architecture:** U-Net with ResNet34 encoder (pretrained on ImageNet)
- **Training time:** ~X minutes on [GPU/CPU]

#### Damage Classification Model
- **Mean IoU:** 0.XX
- **Macro F1:** 0.XX
- **Per-Class F1 Scores:**
  - No Damage: 0.XX
  - Minor Damage: 0.XX
  - Major Damage: 0.XX
  - Destroyed: 0.XX
- **Architecture:** U-Net with concatenated pre/post images
- **Training time:** ~X minutes on [GPU/CPU]

### Model Architecture

The pipeline uses a two-stage approach:

1. **Stage 1: Building Localization**
   - U-Net with ResNet34 encoder
   - Binary segmentation (building vs background)
   - Trained on pre-disaster imagery

2. **Stage 2: Damage Classification**
   - U-Net processing pre+post image pairs
   - 4-class pixel-wise classification
   - Combined Cross-Entropy + Dice Loss for class imbalance

### Key Features

✅ **Production-quality preprocessing** - Handles 45K+ satellite image pairs
✅ **Geospatial engineering** - Vector polygon rasterization with Shapely/Rasterio
✅ **Smart tiling** - 64-pixel overlap prevents edge artifacts
✅ **Comprehensive testing** - 23 unit tests covering all modules
✅ **Reproducible** - Config-driven with fixed random seeds

---

## Example Usage

```python
from xbd_damage_assessment.models import get_model
from xbd_damage_assessment.data.dataset import xBDDataset
import torch

# Load trained model
model = get_model(task="damage", encoder_name="resnet34")
checkpoint = torch.load("checkpoints/damage/best_damage_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load test data
dataset = xBDDataset(data_root="data", split="test", task="damage")
sample = dataset[0]

# Predict
with torch.no_grad():
    pred = model(sample["image_pre"].unsqueeze(0), sample["image_post"].unsqueeze(0))
    damage_map = pred.argmax(1).squeeze().numpy()

# Visualize
import matplotlib.pyplot as plt
plt.imshow(damage_map, cmap="viridis")
plt.colorbar(label="Damage Class")
plt.show()
```

---

## Training Your Own Models

See [DEMO_QUICKSTART.md](DEMO_QUICKSTART.md) for instructions on:
1. Generating sample data
2. Running preprocessing
3. Training models
4. Creating visualizations

Or use the real xBD dataset from [xView2 Challenge](https://xview2.org/).

---

## Future Work

- [ ] Implement uncertainty quantification (MC Dropout)
- [ ] Add attention mechanisms to model architecture
- [ ] Multi-scale/multi-resolution processing
- [ ] Deploy with FastAPI backend
- [ ] Create interactive web demo
- [ ] Integrate with PostGIS for spatial queries
- [ ] Add model explainability (GradCAM)

---
