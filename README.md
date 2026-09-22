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

## Benchmark Progression & Kaggle Leaderboard Results

The Dual-Stream DINO ViT-Base architecture was evaluated systematically across local cross-validation and official Kaggle submission evaluations:

### 1. Official Kaggle Leaderboard Progression

| Evaluation Setup | Public Score | Private Score | Notes / Progression |
| :--- | :---: | :---: | :--- |
| **Initial Baseline** (prior to refactor) | $\approx 0.56000$ | $\approx 0.50000$ | Standard single-view ViT / tabular baseline |
| **3-Fold Ensemble** | $0.61058$ | $0.54996$ | Initial Dual-Stream DINOv3 + UEPNet interval heads |
| **5-Fold Ensemble** | **$0.62868$** | $0.56703$ | Full 5-fold ensemble with horizontal-flip TTA (+0.0181 Public / +0.0171 Private) |
| **5-Fold + Test-Time Adaptation (Pseudo-Labeling)** | $0.62598$ | $0.57239$ | Initial online adaptation run |
| **5-Fold + Kitchen Sink Augs (Seed 42)** | $0.62446$ | $0.59253$ | 4-strip perm + grayscale + view swap |
| **5-Fold + Anti-Leakage Split (`seed=223`)** | $0.62446$ | **`0.59825`** 🏆 | **New Peak Private Score (+0.09825 lift from baseline; just 0.0017 from 0.60!)** |

### 2. Local 5-Fold Stratified Group Cross-Validation (OOF)

| Fold | Baseline (Standard Augs, Seed 42) | Kitchen Sink (Seed 42) | Anti-Leakage Split (Seed 223) |
| :---: | :---: | :---: | :---: |
| **Fold 1** | $0.6097$ | **$0.6204$** | $0.3964$ (Zero WA clover artifact) |
| **Fold 2** | $0.6161$ | $0.5528$ | **$0.7350$ (+0.1189 lift!)** 🚀 |
| **Fold 3** | **$0.7882$** | $0.7915$ | $0.6573$ |
| **Fold 4** | **$0.8020$** | $0.7759$ | $0.7746$ |
| **Fold 5** | $0.6638$ | $0.6392$ | **$0.7652$ (+0.1014 lift!)** 🚀 |
| **Overall 5-Fold OOF** | `0.7251` | `0.7058` | **`0.7266` (Highest Single-Model OOF)** 🏆 |
| **10-Model Blend (Baseline + New Run)** | — | — | **`0.8215` (+0.0108 lift over baseline)** 🏆 |

### 3. Cross-Validation Alignment & Anti-Leakage Strategy

To prevent overfitting and eliminate the CV–LB discrepancy (frequently observed on Kaggle where naive splits gave inflated CV $0.77 \to$ LB $0.60$ drops):
1. **Group by `Sampling_Date` (Zero Temporal Leakage)**:
   The competition host confirmed that test set images are captured on distinct sampling dates. Holding out entire dates prevents the network from simply memorizing that day's specific solar angle, soil moisture, and pasture growth stage.
2. **Stratify by `State` (Preserving Regional Phenology)**:
   Extreme regional divergence exists across Australia:
   - **Western Australia (WA)**: Has $0.0\text{g}$ dead thatch across all plots, but high clover ($22.1\text{g}$).
   - **New South Wales (NSW)**: Has virtually $0.0\text{g}$ clover ($0.13\text{g}$), but high dry green ($56.6\text{g}$).
   - **Tasmania**: Densest dead thatch ($15.2\text{g}$).
   Stratifying by `State` guarantees every validation fold has an identical, realistic national distribution.
3. **Monte Carlo Seed Search (`seed=223`)**:
   Standard seeds with `StratifiedGroupKFold` produce lopsided fold sizes due to lumpy date clusters (e.g., Fold 3 had 50 images while Fold 4 had 120 images). A 1,000-seed search identified **`seed=223`**, balancing validation fold counts to an even **`[95, 80, 89, 86, 89]`** ($\text{Std} = 4.87$ vs $13.17$ originally).

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

## 1st-Place Solution Heritage vs. Codebase Enhancements

This codebase is directly influenced by the core architectural innovations and empirical findings of the **1st-Place Solution** in the CSIRO Image2Biomass competition. Below is a structured breakdown detailing what principles were adopted, why they work, and how this repository enhances and structures them for production and competition reuse.

### Summary Comparison Table

| Dimension | 1st-Place Solution | This Codebase | Rationale / Benefit |
| :--- | :--- | :--- | :--- |
| **Image Tiling** | Centerline vertical split ($1000 \times 1000 \times 2$) | Centerline vertical split ($1000 \times 1000 \times 2$) | Preserves 1:1 aspect ratio without squashing plant geometry |
| **Backbone Architecture** | DINO ViT Base (`vit_base_patch16_dinov3_qkvb`) | Shared DINO ViT Base with Multi-Head Self-Attention | Captures lighting-invariant token embeddings across the quadrat seam |
| **Multi-Task Objective** | SmoothL1 regression + 7-bin classification | SmoothL1 regression + 7-bin UEPNet classification | Discrete interval logits stabilize gradients and handle 38% zero-inflation |
| **Physical Constraints** | Decoupled training + soft post-processing | Decoupled continuous heads + soft post-processing | Avoids gradient conflicts caused by human drying/weighing measurement noise |
| **Clover Calibration** | Soft scalar dampening ($\times 0.8$) | Soft scalar dampening ($\times 0.8$) | Corrects systematic visual overestimation of dense canopy clover leaves |
| **Code Structure** | Monolithic competition notebook | Modular package (`src/training`, `src/inference`, `src/scripts`, `src/notebooks`) | Enables local debugging, modular testing, and reproducible experiments |
| **Kaggle Execution** | Single heavy all-in-one script | 3 distinct standalone notebooks (Train, Fast Inference, Pseudo-Labeling) | Decouples ~30s inference from 2-hour training; isolates online adaptation |
| **Cross-Validation** | Standard K-Fold / Random splitting | 5-fold Stratified Group K-Fold (by `State`) | Prevents same-farm geographic/temporal leakage between train and val |
| **Python 3.12 Safety** | Unhandled multiprocessing errors | Enforced `num_workers=0` + AMP dual-device fallback | Eliminates Kaggle Python 3.12 semaphore deadlocks and worker crashes |
| **EDA & Diagnostics** | Scattered ad-hoc tabular exploration | Unified `src/scripts/eda_insights.py` with `logs/eda/` | Discards useless test-absent tabular interactions; highlights vision realities |

---

### Key Principles Adopted from the 1st-Place Solution

1. **Centerline Vertical 1:1 Tiling**:
   - Standard resizing of panoramic $2000 \times 1000$ images down to $224 \times 224$ discards up to 97% of native pixels and squashes pasture textures. Slicing vertically down the centerline creates two square $1000 \times 1000$ sub-images, preserving native leaf geometry and resolution.
2. **Cross-View Self-Attention Interaction**:
   - Instead of treating the two halves independently or naively concatenating features, a `MultiheadAttention` layer models spatial continuity across the left and right plot seam before passing to the fusion MLP.
3. **Auxiliary Interval Classification (UEPNet CVPR 2021)**:
   - Biomass estimation without segmentation maps resembles crowd counting. Discretizing continuous target grams into 7 non-uniform density intervals provides discrete classification logits that anchor regression gradients against extreme zero-inflation (37.8% zero clover, 11.2% zero dead).
4. **Decoupled Training with Soft Post-Processing**:
   - Field measurements have empirical noise ($\pm 5–15\%$ slack from drying and sorting loss). Imposing hard mathematical equality ($Total = Green + Dead + Clover$) during backpropagation creates conflicting gradients. Decoupled training with soft blending ($0.5 \times \text{Prediction} + 0.5 \times \text{Components}$) achieves the highest $R^2$.
5. **Test-Time Adaptation (Online Pseudo-Labeling)**:
   - Generating soft-calibrated pseudo-labels on unlabelled test images followed by 4 rapid epochs of fine-tuning at low learning rate (`3e-5`) adapts the attention heads to the exact soil and lighting distribution of the test set.

---

### Key Differences & Engineering Enhancements in This Codebase

1. **Modular Architecture & VS Code Integration**:
   - Rather than relying on a fragile monolithic notebook, logic is partitioned into dedicated modules (`dataset.py`, `models.py`, `common.py`, `train_unified.py`, `local_inference.py`, `eda_insights.py`) supported by `.vscode/launch.json` debug profiles.
2. **Dedicated, Offline-Capable Kaggle Notebooks**:
   - **`biomass-inference-submission.ipynb`**: Pure inference running in ~30 seconds on GPU with `pretrained=False` (offline weight loading).
   - **`biomass-inference-pseudo-labeling.ipynb`**: Online adaptation notebook blending 5-fold ensemble with adapted predictions.
   - **`biomass-lastbatchnorm-ensemble.ipynb`**: Full 5-fold training pipeline from scratch.
3. **Leakage-Free Stratified Group Validation**:
   - Groups samples by `State` and continuous target bins to ensure train and validation folds never share pasture plots from the same farm or sampling date.
4. **Container & Worker Stability**:
   - Kaggle's migration to Python 3.12 introduced multi-process semaphore errors (`can only test a child process`). All notebook data loaders enforce safe worker pooling (`num_workers=0`) and adaptive AMP autocast handling for both CUDA and CPU.
5. **Streamlined EDA**:
   - Removed ~2,200 lines of obsolete tabular feature interactions (NDVI/Height features absent from test set) and replaced them with a consolidated, logging-driven EDA tool that saves structured summaries to `logs/eda/`.

---

## Repository Structure

```
├── .vscode/
│   ├── launch.json              # VS Code debug configurations for training and inference
│   └── settings.json            # Python interpreter and analysis paths
├── src/
│   ├── training/
│   │   ├── config/
│   │   │   ├── config.yaml      # Central pipeline hyperparameter configuration (ViT & ConvNeXt)
│   │   │   ├── loader.py        # Config loader
│   │   │   └── schemas.py       # Dataclass schemas
│   │   ├── common.py            # Loss functions, UEPNet borders, metric & post-processing
│   │   ├── configs.py           # Configuration exporter
│   │   ├── dataset.py           # DualStreamBiomassDataset with centerline split & focal scaling
│   │   ├── models.py            # DualStreamBiomassModel (DINO ViT / ConvNeXt-V2 + Cross-View Attention + 10 Heads)
│   │   └── train_unified.py     # Two-stage StratifiedGroupKFold training pipeline (with grad accum)
│   ├── inference/
│   │   ├── local_inference.py   # Multi-fold ensembling with TTA and submission generation
│   │   └── test_time_pseudolabel.py # Online pseudo-labeling with Stochastic Weight Averaging
│   └── notebooks/
│       ├── training.ipynb       # Self-contained 5-fold training pipeline with anti-leakage split (0.59825 Private LB)
│       └── inference.ipynb      # Universal TTA inference & ensembling (auto-detects ViT & ConvNeXt models)
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

### 4. Running on Kaggle

Two standalone notebooks are provided in [`src/notebooks/`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/):

1. **Training Notebook**: [`src/notebooks/training.ipynb`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/training.ipynb)
   - Runs full 5-fold cross-validation with the anti-leakage split (`seed=223`, `Sampling_Date` grouping, `State` stratification) and 3rd-place kitchen sink augmentations.
   - Dual-Stream 1:1 square tiling + cross-view attention + UEPNet 7-bin interval classification.
   - Achieved our project-peak **`0.59825` Private LB**.
   - Saves `best_model_fold1.pt` ... `best_model_fold5.pt` and produces standalone `submission.csv`.

2. **Inference Notebook**: [`src/notebooks/inference.ipynb`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/inference.ipynb)
   - Fast, offline-capable inference and ensembling notebook.
   - **Automatic Architecture Detection**: Automatically detects whether checkpoints are DINO ViT (768 channels) or ConvNeXt-V2 (1536 channels) and dynamically loads them.
   - Upload your trained `.pt` files as a Kaggle Dataset, attach via `+ Add Input`, and click **Run All**.
   - Runs dual-stream inference with horizontal-flip TTA, applies soft physics post-processing, and writes `submission.csv` in ~30 seconds on GPU.
