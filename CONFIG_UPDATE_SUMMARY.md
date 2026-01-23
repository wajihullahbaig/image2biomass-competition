# Config Update Summary - Optimized for Better Kaggle Performance

**Date:** 2026-01-23
**Goal:** Improve from 0.59 → 0.68+ R² score

## Changes Made (Based on Training Plot Analysis)

### 1. **Learning Rate & Schedule** ⚡

```yaml
learning_rate: 0.0015 # Was: 0.00075 (2x increase)
backbone_lr_factor: 0.3 # Was: 0.2 (less aggressive reduction)
```

**Rationale:** Plots showed stable, smooth learning curves. Model can handle faster learning without instability.

### 2. **Backbone Unfrozen** 🔓 (CRITICAL)

```yaml
freeze_backbone: false # Was: true
```

**Rationale:** Most important change. DINOv3 needs domain adaptation from ImageNet → pasture images. Your frozen backbone was limiting performance ceiling.

### 3. **More Training Time** ⏱️

```yaml
epochs: 60 # Was: 20
early_stop_patience: 15 # Was: 10
```

**Rationale:** Plots showed continuous improvement at epoch 32. Model needs more time to converge.

### 4. **Reduced Regularization** 📉

```yaml
weight_decay: 0.05 # Was: 0.1
```

**Rationale:** 0.1 was too aggressive for small dataset. Reducing allows better fitting.

### 5. **Task Weight Rebalancing** ⚖️

```yaml
biomass_feat_weight: 15.0 # Was: 11.0 (focus on main task)
aux_feat_weight: 1.0 # Was: 2.0 (reduce overfitting)
species_feat_weight: 0.5 # Was: 1.0 (reduce distraction)
```

**Rationale:**

- Aux component plots showed Species_Count has huge train/val gap (overfitting)
- Dead biomass prediction is noisy - need more focus on biomass task
- Reduce auxiliary task influence

### 6. **Augmentation Enabled** 🎲

```yaml
tile_prob: 0.8 # Was: 0.0 (6x effective data)
mixup_prob: 0.15 # Was: 0.0 (regularization)
```

**Rationale:** With only 229 training samples, augmentation is critical. Tile augmentation provides 6x data without destabilizing training.

## Expected Improvements

### Short-term (First 20 epochs):

- **Faster convergence** due to higher LR
- **Better R² scores** from unfrozen backbone adapting to pasture images
- **More stable Dead predictions** from increased biomass focus

### Long-term (40-60 epochs):

- **Validation R²:** 0.50 → 0.60-0.65 (target)
- **Holdout R²:** 0.50 → 0.60-0.68 (competitive range)
- **Reduced overfitting** on auxiliary tasks
- **Better generalization** from augmentation

## Key Metrics to Watch

1. **Holdout R²** - This is your Kaggle proxy score
2. **Dead Loss convergence** - Should be less noisy now
3. **Train/Val gap** - Should reduce with unfrozen backbone
4. **Species Count loss** - Should have smaller train/val gap with reduced weight

## Next Steps

1. **Run training** with new config
2. **Monitor first 10 epochs** - ensure LR isn't too high (check for loss spikes)
3. **Check epoch 20** - should already see improvement over previous best
4. **Let it run to convergence** - early stopping will catch optimal point

## Rollback Plan (if needed)

If training becomes unstable (loss spikes, NaN values):

1. Reduce LR to 0.001
2. Reduce tile_prob to 0.5
3. Keep backbone unfrozen (most critical change)

## Files Modified

- `src/training/config/config.yaml`

## Previous Best Performance

- Validation R²: 0.473
- Holdout R²: 0.506
- Score: 0.4392 (epoch 29)

## Target Performance

- Validation R²: 0.60+
- Holdout R²: 0.65-0.68
- Competitive with image-only models
