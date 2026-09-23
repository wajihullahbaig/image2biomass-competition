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
| **5-Fold Ensemble** | $0.62868$ | $0.56703$ | Full 5-fold ensemble with horizontal-flip TTA (+0.0181 Public / +0.0171 Private) |
| **5-Fold + Test-Time Adaptation (Pseudo-Labeling)** | $0.62598$ | $0.57239$ | Initial online adaptation run |
| **5-Fold + Kitchen Sink Augs (Seed 42)** | $0.62446$ | $0.59253$ | 4-strip perm + grayscale + view swap |
| **5-Fold + Anti-Leakage Split (`seed=223`, Run 1)** | $0.62446$ | **`0.59825`** 🏆 | **All-Time Peak Private Score (+0.09825 lift from baseline; within 0.0017 of 0.60!)** |
| **5-Fold + Anti-Leakage Split (`seed=223`, Run 2)** | **`0.63104`** 🚀 | `0.59582` | **All-Time Peak Public Score (+0.07104 over baseline!)** |

### 2. Local 5-Fold Stratified Group Cross-Validation (OOF)

| Fold | Baseline (Seed 42) | Kitchen Sink (Seed 42) | Run 1 (`seed=223`) | Run 2 (`seed=223`) |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | $0.6097$ | **$0.6204$** | $0.3964$ (Zero WA clover artifact) | **`0.5023` (+0.1059 lift!)** 🚀 |
| **Fold 2** | $0.6161$ | $0.5528$ | **$0.7350$** | **`0.7091`** |
| **Fold 3** | **$0.7882$** | $0.7915$ | $0.6573$ | **`0.6174`** |
| **Fold 4** | **$0.8020$** | $0.7759$ | $0.7746$ | **`0.7677`** |
| **Fold 5** | $0.6638$ | $0.6392$ | $0.7652$ | **`0.7674`** |
| **Overall 5-Fold OOF** | `0.7251` | `0.7058` | `0.7266` | **`0.7275` (New All-Time High Single-Model OOF)** 🏆 |
| **Per-Target R² (Run 2)** | — | — | — | **Green: `0.7944` \| GDM: `0.7821` \| Total: `0.7066` \| Clover: `0.5481` \| Dead: `0.4270`** |

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

### 4. 3rd-Place Kitchen Sink Augmentations

- **Vertical 4-Strip Permutation ($p=0.5$)**: Slices sub-images into 4 vertical strips and permutes their order. Mass is strictly conserved while breaking spatial position bias.
- **Random Grayscale ($p=0.2$)**: Forces representation learning on leaf geometry and canopy texture rather than purely color shortcuts.
- **View Swap ($p=0.5$)**: Swaps Left and Right views into cross-view attention.
- **Camera Scaling ($p=0.2$)**: Jitters scale with black padding.

### 5. 3-Stage "Sandwich" Training Schedule (LP → FT → Re-Freeze)

```
Epoch 01 ──────────────────────── Epoch 14 ──────────────────────── Epoch 30 ────── Epoch 35
  │                                      │                                 │           │
  ▼                                      ▼                                 ▼           ▼
┌──────────────────────────────────────┐┌────────────────────────────────┐┌───────────┐
│     STAGE 1: Heads Warm-up           ││  STAGE 2: Full Fine-Tuning     ││ STAGE 3:   │
│  • Backbone: FROZEN                  ││ • Backbone: UNFROZEN (3e-5 LR) ││ Calibration│
│  • Heads & Attention: LR = 3e-4      ││ • Heads: LR = 3e-4 (Cosine LR) ││ • Backbone:│
│  • Heads mature to R² ≈ 0.35-0.45    ││ • Adapts to pasture textures   ││   RE-FROZEN│
│  • Zero risk to DINO representations ││ • End-to-end multi-task loss   ││ • LR: 3e-5 │
└──────────────────────────────────────┘└────────────────────────────────┘└───────────┘
```

1. **Stage 1 (14 Epochs — Heads Warm-up / Linear Probing)**:
   * **Backbone is FROZEN**. Trains only the cross-view multi-head attention, fusion MLP, and the 10 regression/interval heads with base LR (`3e-4`).
   * *Benefit*: Completely eliminates early gradient shock. The heads reach full maturity ($R^2 \approx 0.35 - 0.45$) *before* the pre-trained DINOv3 backbone is touched.
2. **Stage 2 (16 Epochs — Full Fine-Tuning / Pasture Adaptation)**:
   * **Backbone is UNFROZEN**. Differential learning rate: `backbone_lr = 3e-5` ($0.1\times$), `heads_lr = 3e-4` with Cosine Annealing.
   * *Benefit*: Deep end-to-end visual feature adaptation directly tailored to Australian pasture canopies.
3. **Stage 3 (5 Epochs — Head Calibration / Re-Freeze)**:
   * **Backbone is RE-FROZEN**. Starts from the best checkpoint saved during Stage 2.
   * Fine-tunes only the attention, fusion, and heads at low learning rate (`3e-5` decaying to `1e-6`).
   * *Benefit*: Eliminates backbone feature drift in the final epochs, allowing the regression heads and softplus boundaries to lock into optimal calibration against the learned pasture features.

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

### 4. Fast 2-Stage DINOv2-Small Training (2nd-Place Solution Integration)

The pipeline integrates the core findings from the **2nd-Place Solution** (Public **`0.77839`** / Private **`0.67558`**):
1. **Predict 3 Base Targets Only (`Green`, `Dead`, `Clover`)**:
   - Physical composite targets are derived mathematically:
     $$\text{GDM} = \text{Green} + \text{Clover}$$
     $$\text{Total} = \text{Green} + \text{Dead} + \text{Clover}$$
   - Eliminates contradictory gradient backpropagation across composite quantities.
2. **State-Level Post-Processing (+0.014 Private LB lift)**:
   - **WA Dead Zeroing**: Forces `Dry_Dead_g = 0.0` for all Western Australia samples (matches ground-truth zero thatch).
   - **State Multipliers**: Calibrates regional collection offsets: NSW ($\text{Green} \times 1.03$), Vic ($\text{Clover} \times 0.85$), WA ($\text{Clover} \times 0.80, \text{Dead} \times 0.80, \text{Green} \times 0.97$).
   - **Training Bound Clipping**: Restricts predictions to physical pasture bounds ($\text{Clover} \le 71.79$, $\text{Dead} \le 83.84$, $\text{Green} \le 157.98$).
3. **Color Space Preprocessing**:
   - Gray World adaptive white balance normalizes sunlight and camera variations across states and dates.
   - HSV shadow correction boosts the $V$ channel in detected shadow regions ($V < \mu - 0.5\sigma$) to prevent shadowed living grass from being misclassified as dead material.
4. **Fast 5-Fold State Stratification**:
   - 357 clean real images split across 5 stratified folds on `State` (`seed=42`), guaranteeing identical state distributions per fold:
     - Fold 1: 72 samples (Tas: 28, Vic: 23, NSW: 15, WA: 6)
     - Fold 2: 72 samples (Tas: 28, Vic: 23, NSW: 15, WA: 6)
     - Fold 3: 71 samples (Tas: 28, Vic: 22, NSW: 15, WA: 6)
     - Fold 4: 71 samples (Tas: 27, Vic: 22, NSW: 15, WA: 7)
     - Fold 5: 71 samples (Tas: 27, Vic: 22, NSW: 15, WA: 7)
   - Backbone: `vit_small_patch14_dinov2` (21M params, native $518 \times 518$ patch14).
   - Schedule: **7 epochs warm-up (Stage 1)** + **23 epochs fine-tuning (Stage 2)** (30 total epochs per fold) with batch size 16.
   - Saves visual sample batches for each fold to inspect data entering the model.
   - **Complete 5-fold cross-validation finishes in ~30–35 minutes on GPU**.

### 5. Running on Kaggle

Two standalone, zero-dependency notebooks are maintained in [`src/notebooks/`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/):

1. **Training Notebook**: [`src/notebooks/training.ipynb`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/training.ipynb)
   - Self-contained 5-fold DINOv2-Small training notebook with 2nd-place post-processing, Gray World white balance, and HSV shadow compensation.
   - Predicts 3 base targets, derives 5 full targets, displays sample batch grids, and saves `best_model_fold1.pt` ... `best_model_fold5.pt`.
2. **Inference Notebook**: [`src/notebooks/inference.ipynb`](file:///c:/Users/Precision/Onus/GitHub/image2biomass-competition/src/notebooks/inference.ipynb)
   - Universal multi-backbone inference and ensembling notebook across all 5 folds.
   - **Smart Architecture Detection**: Auto-detects whether uploaded checkpoints are DINOv2-Small (384-dim, 3 targets), DINOv3 ViT-Base (768-dim, 5 targets), or ConvNeXt-V2 Large (1536-dim).
   - Dynamically blends predictions, applies 2nd-place post-processing, and generates `submission.csv` in ~30 seconds on GPU.

