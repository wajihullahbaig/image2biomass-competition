# CSIRO Image2Biomass — Dual-Stream DINOv3

Pasture biomass regression for the [CSIRO Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass) Kaggle competition (late submissions; project closed 2026-10-08).

**Best result:** private **0.602** / public 0.649 — one ViT-B model trained on all images, with 1st-place post-processing.
Full experiment log and lessons: [HISTORY.md](HISTORY.md).

## Approach

Based on the 1st-place solution, with fixes to validation and the loss:

- **Dual view:** each 2000×1000 panorama is split into two 1000×1000 halves. Both go through one shared DINOv3 backbone (`vit_base_patch16_dinov3_qkvb` @ 512), then cross-view attention and a fusion MLP.
- **Heads:** 5 regression heads (Green, Dead, Clover, GDM, Total) plus 5 auxiliary 7-interval classification heads (UEPNet borders).
- **Loss:** squared error weighted like the official metric. The metric squares errors in grams, and L1-type losses squeezed predictions toward the mean (+0.06 OOF from switching).
- **Training:**
  - Stage 1: frozen backbone.
  - Stage 2: full fine-tune with a 0.1× backbone LR and cosine decay.
  - Regression heads start at the training means.
  - Fixed epoch budget, saving the average of the last 5 epochs' weights (SWA) — no best-epoch picking.
- **Augmentations:** flips and 90° rotations, colour jitter, grayscale, CLAHE, Gaussian noise, camera-scale padding, vertical strip shuffle, left/right view swap.
- **Validation:** 5 folds grouped by `Sampling_Date` and stratified by `State`, on the 357 real images.
  - The 82 `is_synthetic` rows in `wide.csv` are dropped: they duplicate 19 real images and leaked across folds.
  - Metric: the official global weighted R² over all (image, target) pairs, in raw grams.
- **Inference:**
  - Mirrored-panorama TTA, averaging every checkpoint matched by `--models`.
  - 1st-place post-processing (clover ×0.8, dead fringe, GDM/Total blends), on by default. It lowers CV but adds about +0.01 on the private leaderboard.

## Results

| Model | OOF R² | Public | Private |
|---|---|---|---|
| ViT-B, 5-fold ensemble + post-processing | 0.751 | 0.656 | 0.599 |
| **ViT-B, full data + post-processing** | ≈0.759 (same recipe) | 0.649 | **0.602** |

See [HISTORY.md](HISTORY.md) for every run, ablation and submission.

## Usage

```powershell
pip install -r requirements.txt

python train.py                                            # 5-fold CV -> models/, logs/<session>/
python train.py --full_train --output_dir models_full      # one all-data model for submission
python inference.py --models "models_full/*.pt"            # -> submission.csv
python make_notebooks.py                                   # rebuild the Kaggle notebooks after editing the scripts
```

Defaults: ViT-B @512, batch 8, 8 + 25 epochs. Use `--backbone vit_large_patch16_dinov3_qkvb` for ViT-L, which fits in 16 GB with the built-in gradient checkpointing.

**On Kaggle:**
- **Training:** in `notebooks/training.ipynb`, attach `wide.csv` as a dataset and run all cells.
- **Inference:** in `notebooks/inference.ipynb`, attach the `.pt` checkpoints and set `--models` in the last cell.

## Repository

```
train.py              data, folds, augmentations, model, loss, metric, training loop
inference.py          checkpoint discovery, TTA, post-processing, submission.csv
make_notebooks.py     builds both notebooks from the two scripts (edit the scripts, not the notebooks)
notebooks/            Kaggle training and inference notebooks (generated)
HISTORY.md            experiment log, leaderboard results, findings
requirements.txt      pinned versions
```

`wide.csv`, images, logs and checkpoints are gitignored.

## If resumed

1. **Test-time pseudo-labelling.** The 1st-place team reported more than +0.02 private from it, and the private test differs from the training data.
2. **Finish ViT-L.** Folds 1–4 beat ViT-B by about 0.011 OOF. Then train a full-data ViT-L, then try 1024 px.
3. **Average 2–3 full-data models trained with different seeds.**
