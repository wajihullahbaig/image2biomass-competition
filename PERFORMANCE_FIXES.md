# Performance Issues & Fixes

**Current Score:** 0.59 | **Target:** 0.68+ (matching image-only models)

## Critical Issues Found

### 1. ⚠️ ULTRA-LOW LEARNING RATE (HIGHEST PRIORITY)

**Problem:** Actual LR is 3e-05 (0.00003) - model barely learns

- Config says 0.00075 but backbone_lr_factor (0.2) makes backbone LR = 0.00015
- Scheduler reduced it further to 3e-05

**Fix:**

```yaml
learning_rate: 0.003 # 4x increase
backbone_lr_factor: 0.5 # Less aggressive reduction
```

### 2. ⚠️ FROZEN BACKBONE

**Problem:** 80% of backbone is frozen, preventing domain adaptation
**Fix:**

```yaml
freeze_backbone: false # Or set to true with:
backbone_freeze_fraction: 0.3 # Freeze only first 30%
```

### 3. ⚠️ TOO FEW EPOCHS

**Problem:** Model hasn't converged at epoch 29, but config limits to 20
**Fix:**

```yaml
epochs: 100 # Let early stopping decide when to stop
early_stop_patience: 15 # More patience
```

### 4. ⚠️ BATCH SIZE TOO LARGE

**Problem:** 229 samples ÷ 64 batch = only 3.6 batches/epoch (unstable gradients)
**Fix:**

```yaml
batch_size: 16 # or 8 - allows 14-28 batches/epoch
```

### 5. ⚠️ NO AUGMENTATION

**Problem:** Disabled tile_prob and mixup_prob = severe underfitting
**Fix:**

```yaml
tile_prob: 1.0 # Always use tile augmentation (6x data)
mixup_prob: 0.2 # Enable mixup for regularization
```

### 6. ⚠️ OVER-COMPLEXITY

**Problem:** Multi-task learning with aux/species/physics hurts vs simple image-only
**Consider:** Simplify to pure regression (disable aux/species losses temporarily)

```yaml
aux_feat_weight: 0.1 # Reduce from 2.0
species_feat_weight: 0.1 # Reduce from 1.0
biomass_feat_weight: 20.0 # Keep high (currently 11.0)
```

### 7. ⚠️ WEIGHT DECAY TOO HIGH

**Problem:** 0.1 is very aggressive, may prevent fitting
**Fix:**

```yaml
weight_decay: 0.01 # Reduce 10x
```

## Quick Win Config (Copy-Paste Ready)

```yaml
hyperparameters:
  batch_size: 16 # Changed from 64
  learning_rate: 0.003 # Changed from 0.00075
  n_folds: 3
  epochs: 100 # Changed from 20
  weight_decay: 0.01 # Changed from 0.1
  early_stop_patience: 15 # Changed from 10
  backbone: timm/vit_small_patch16_dinov3.lvd1689m
  min_train_samples: 50
  backbone_freeze_threshold: 400
  max_grad_norm: 1.0
  backbone_lr_factor: 0.5 # Changed from 0.2

training:
  freeze_backbone: false # Changed from true - CRITICAL
  use_tta: true
  fusion_dim: 256
  biomass_feat_weight: 20.0 # Changed from 11.0 - focus on main task
  aux_feat_weight: 0.5 # Changed from 2.0 - reduce distraction
  species_feat_weight: 0.5 # Changed from 1.0 - reduce distraction
  ema_decay: 0.9

augmentation:
  tile_prob: 1.0 # Changed from 0.0 - CRITICAL for small dataset
  mixup_prob: 0.2 # Changed from 0.0
  mixup_alpha: 0.40
```

## Expected Improvements

- **LR fixes:** Should reach R² 0.65-0.70 in validation
- **Unfrozen backbone:** +0.05-0.10 R² from domain adaptation
- **More epochs:** Proper convergence
- **Tile augmentation:** Effective 1374 samples instead of 229
- **Smaller batches:** More stable gradients, better generalization

## Alternative: Pure Image-Only Baseline

If the above doesn't reach 0.68, try a **stripped-down version**:

- Remove all auxiliary heads (comment out aux_head, species_head)
- Single biomass_head: backbone → 256 → 5 outputs
- No physics gating, no multi-task learning
- Focus 100% on image → biomass regression

This mirrors what winning teams are doing.
