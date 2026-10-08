# Experiment History

Metric: official **global weighted R²** over all (image, target) pairs on raw grams.
OOF = out-of-fold score on the 357 real images (synthetic rows removed), 5 folds grouped by `Sampling_Date`, stratified by `State`, seed 223.
Gold zone (private LB) ≈ **0.679**. 1st place ≈ 0.67–0.68.

## Kaggle submissions

| Date | Submission | OOF | Public | Private |
|---|---|---|---|---|
| 2026-09 | ViT-B @512, SmoothL1, 439 rows incl. synthetic, best-epoch, 1st-place post-proc (run 1) | ~0.746¹ | 0.624 | **0.598** |
| 2026-09 | Same recipe (run 2) | — | 0.631 | 0.596 |
| 2026-10-08 | **Run C** — ViT-B @512, weighted MSE, SWA, clean folds, no post-proc | **0.759** | **0.656** | 0.588 |

¹ Re-scored with the official metric on real rows minus the 19 duplicated images; still inflated by best-epoch picking.

## Local CV runs (same clean folds)

| Date | Run | Backbone | Loss | Batch / steps per epoch | OOF raw | OOF post-proc | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-06 | ViT-L ε | ViT-L @512 | ε-insensitive L1 | 4×4 accum / 18 | 0.563 | 0.532 | Undertrained; Stage 1 wasted (heads started at ~0.7 g) |
| 2026-10-07 | A | ViT-B @512 | SmoothL1 | 8 / 36 | 0.700 | 0.693 | + head bias init at train means |
| 2026-10-07 | **C** | ViT-B @512 | weighted MSE | 8 / 36 | **0.759** | 0.751 | NSW R² 0.585 → 0.692; folds 0.770 / 0.659 / 0.757 / 0.759 / 0.743 |

## Findings

- **Synthetic rows leaked.** `wide.csv`'s 82 `is_synthetic` rows are 19 real images duplicated with jittered labels and shifted dates; 35 validation images also appeared in training. Dropped.
- **The metric had been implemented inconsistently** (per-target raw vs log1p). Now one official implementation.
- **Best-epoch picking inflates OOF** by 0.01–0.04 per fold. Replaced with a fixed budget + SWA of the last 5 epochs.
- **L1-type losses compress predictions** toward the mean (predicted Total sd 19 g vs true 28 g) and under-predict heavy NSW pastures. Weighted MSE matches the metric: +0.06 OOF.
- **1st-place post-processing** (clover ×0.8, dead fringe, mass blends) lowered OOF in every run (−0.007 to −0.031). Not yet A/B-tested on the leaderboard with the new models.
- **CV gains did not reach the private LB** (CV +0.06, private −0.01). Private scores sit within ~±0.01 of each other, and public–private gaps of 0.03–0.07 suggest the private test differs from training. Prefer test-adaptive ideas (pseudo-labelling) and large changes over small CV-tuned tweaks.

## Queue

- [ ] Run C checkpoints with `--postprocess first_place` (LB A/B, no training)
- [ ] Full-data ViT-B MSE model (`models_full/`)
- [ ] ViT-L @512 MSE 5-fold CV (`models_vitl_mse/`)
- [ ] Test-time pseudo-labelling in the inference notebook
