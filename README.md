# CSIRO - Image2Biomass Prediction Pipeline

High-performance biomass prediction architecture inspired by the **1st-Place Solution** of the [CSIRO Image2Biomass Kaggle Competition](https://www.kaggle.com/competitions/csiro-biomass).

---

## Architecture Overview

Traditional approaches resize the competition's wide $2000 \times 1000$ panoramic pasture images down to standard square inputs ($224 \times 224$), losing up to 97% of spatial resolution and distorting vegetation textures. This repository implements an end-to-end **Dual-Stream Vision Transformer with Auxiliary Interval Classification**:

```
Raw Field Image (2000 x 1000)
             │
             ▼
  Centerline Vertical Split
   ┌───────────────────┐
   │                   │
   ▼                   ▼
Left View          Right View
(1000 x 1000)      (1000 x 1000)
   │                   │
   ▼ (Camera Scaling)  ▼ (Camera Scaling)
   │                   │
   ▼                   ▼
Shared DINOv3 Vision Transformer Backbone
(vit_base_patch16_dinov3_qkvb @ 512x512)
   │                   │
   ▼                   ▼
Left Token Feats   Right Token Feats
   └─────────┬─────────┘
             │
             ▼
Cross-View Multi-Head Self-Attention Layer
(Allows features across the center seam to interact)
             │
             ▼
   LayerNorm + Residual Fusion MLP
             │
      ┌──────┴──────────────────────────┐
      │                                 │
      ▼                                 ▼
5 Independent Regression Heads    5 Auxiliary Interval Heads
(3-layer MLP with Softplus)       (Predicts 7 UEPNet count bins)
      │                                 │
      ▼                                 ▼
Continuous Biomass (grams)        Discrete Interval Logits
[Green, Dead, Clover, GDM, Total]  (Stabilizes regression gradients)
```

---

## Key Pillars of the Pipeline

### 1. Dual-Stream High-Resolution Tiling

- **Preserved Aspect Ratio**: Images are split along the vertical centerline into two square sub-images ($1000 \times 1000$) and resized to $512 \times 512$ (or $1024 \times 1024$) without squashing.
- **Shared Backbone**: Both views pass through the same pre-trained DINOv3 vision transformer, producing rich visual token representations.
- **Cross-View Attention**: A `MultiheadAttention` layer models spatial continuity across the left and right halves before feature projection.

### 2. Auxiliary Interval Classification (UEPNet Crowd Counting Formulation)

Biomass estimation without segmentation maps is structurally analogous to crowd counting. Continuous target grams are partitioned into 7 non-uniform density intervals:

```python
BORDERS_DICT = {
    'Dry_Green_g':  [1.6e-05, 13.4232, 27.0782, 45.5236, 79.834, 157.9836],
    'Dry_Dead_g':   [1.6e-05, 6.1407, 13.1192, 23.277, 38.8581, 83.8407],
    'Dry_Clover_g': [1.6e-05, 3.9, 10.5353, 20.6523, 37.5911, 71.7865],
    'GDM_g':        [1.6e-05, 16.5143, 30.507, 49.5585, 81.0, 157.9836],
    'Dry_Total_g':  [1.6e-05, 23.4907, 41.1, 61.1, 96.8288, 185.7],
}
```

Five classification heads predict the interval class for each target alongside continuous regression, providing strong gradient guidance and clustering features by density.

### 3. Decoupled Training & Soft Post-Processing

- **Unconstrained Backpropagation**: No rigid mathematical equality constraints ($Total = Green + Dead + Clover$) are enforced inside the forward pass, preventing human measurement/drying noise from producing conflicting gradients.
- **Test-Time Soft Blending**: Harmonizes components softly during post-processing:
  - $Clover \leftarrow 0.8 \times Clover$ (corrects systematic overestimation)
  - Piecewise dead thatch calibration (scaled up if $>20$, down if $<10$)
  - $GDM \leftarrow 0.5 \times GDM + 0.5 \times (Green + Clover)$
  - $Total \leftarrow 0.5 \times Total + 0.5 \times (Green + Clover + Dead)$

### 4. Camera Focal-Scaling Augmentation

Random downscaling ($0.85 - 1.0$) embedded into a black background simulates varying camera sensor heights and focal lengths without altering local pixel density.

### 5. Two-Stage Training Schedule

- **Stage 1 (Epochs 1–8)**: Freeze DINO backbone; train cross-view attention and MLP heads.
- **Stage 2 (Epochs 9–35)**: Full end-to-end fine-tuning with differential learning rate (`backbone_lr = 0.1 * lr`).

---

## Repository Structure

```
├── .vscode/
│   ├── launch.json              # VS Code debug configurations for training and inference
│   └── settings.json            # Python interpreter and analysis paths
├── src/
│   ├── training/
│   │   ├── config/
│   │   │   ├── config.yaml      # Central pipeline hyperparameter configuration
│   │   │   ├── loader.py        # Config loader
│   │   │   └── schemas.py       # Dataclass schemas
│   │   ├── common.py            # Loss functions, UEPNet borders, metric & post-processing
│   │   ├── configs.py           # Configuration exporter
│   │   ├── dataset.py           # DualStreamBiomassDataset with centerline split & focal scaling
│   │   ├── models.py            # DualStreamBiomassModel (DINO + Attention + 10 Heads)
│   │   └── train_unified.py     # Two-stage StratifiedGroupKFold training pipeline
│   ├── inference/
│   │   ├── local_inference.py   # Multi-fold ensembling with TTA and submission generation
│   │   └── test_time_pseudolabel.py # Online pseudo-labeling with Stochastic Weight Averaging
│   └── notebooks/
│       ├── biomass-lastbatchnorm-ensemble.ipynb      # Self-contained Kaggle GPU training + inference
│       ├── biomass-inference-submission.ipynb        # Fast Kaggle inference using uploaded .pt models
│       └── biomass-inference-pseudo-labeling.ipynb   # Test-time pseudo-labeling & online adaptation
├── train/                       # Raw training pasture images (2000x1000)
├── test/                        # Raw test pasture images
├── wide.csv                     # Pivoted sample metadata and target records
└── README.md
```

---

### 1. Exploratory Data Analysis (EDA)

Run the unified EDA pipeline from VS Code via **Run & Debug (`F5`)** $\to$ `Run EDA Insights` or execute:

```powershell
& "C:\Users\Precision\anaconda3\envs\audio_signal_processing\python.exe" src/scripts/eda_insights.py
```

### 2. Training Locally

Launch training from VS Code via **Run & Debug (`F5`)** $\to$ `Train Unified (Dual-Stream DINO)` or execute:

```powershell
& "C:\Users\Precision\anaconda3\envs\audio_signal_processing\python.exe" src/training/train_unified.py
```

### 3. Generating Submissions Locally

Run inference across all trained fold checkpoints with horizontal flip TTA and soft post-processing:

```powershell
& "C:\Users\Precision\anaconda3\envs\audio_signal_processing\python.exe" src/inference/local_inference.py
```

### 3. Test-Time Online Training (Optional)

Generate test pseudo-labels and fine-tune an online model with Stochastic Weight Averaging (SWA):

```powershell
& "C:\Users\Precision\anaconda3\envs\audio_signal_processing\python.exe" src/inference/test_time_pseudolabel.py
```

### 4. Running on Kaggle

Three dedicated, standalone notebooks are provided in [`src/notebooks/`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/):

1. **Full Training + Inference**: [`biomass-lastbatchnorm-ensemble.ipynb`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/biomass-lastbatchnorm-ensemble.ipynb)
   - Runs 5-fold training and generates `submission.csv` directly in the Kaggle GPU kernel.
2. **Fast Dedicated Inference (Uploaded Weights)**: [`biomass-inference-submission.ipynb`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/biomass-inference-submission.ipynb)
   - Upload your local `best_model_fold*.pt` files as a Kaggle Dataset.
   - Attach the dataset to this notebook (`+ Add Input`).
   - Automatically discovers all uploaded model checkpoints, runs dual-stream TTA inference with soft physics post-processing, and generates `submission.csv` in ~30 seconds on GPU.
3. **Test-Time Pseudo-Labeling & Online Adaptation**: [`biomass-inference-pseudo-labeling.ipynb`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/biomass-inference-pseudo-labeling.ipynb)
   - Implements the 1st-place solution test-time adaptation technique.
   - Runs initial 5-fold ensemble with TTA on the test set.
   - Generates calibrated pseudo-labels for test images.
   - Performs 4 epochs of fast online fine-tuning on `train + pseudo_test` with low LR (`3e-5`).
   - Blends adapted predictions (25%) with the 5-fold ensemble (75%) and writes `submission.csv`.
