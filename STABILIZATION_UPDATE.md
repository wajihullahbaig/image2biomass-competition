# Training Stabilization Update

**Date:** 2026-01-23 20:16
**Issue:** Training too noisy after unfreezing backbone

## Problem Observed

Looking at the training plots from `logs/unified_holdout_20260123_192731/`:

### Noise Indicators:

- ❌ **Biomass components** (Clover, Dead, Green): Wild oscillations in val/holdout
- ❌ **HSV components**: Very noisy, especially Green HSV and Dry Green HSV
- ❌ **R² metrics**: Unstable with negative values and large swings
- ❌ **Train loss** much lower than val/holdout initially (overfitting on augmented data)

### Root Causes:

1. **LR too high** (0.0015) + unfrozen ViT = unstable gradients
2. **Mixup (0.15)** adding noise on top of tile augmentation
3. **Batch size 64** too large with aggressive augmentation
4. **Backbone LR factor 0.3** allowing backbone to learn too fast

## Stabilization Changes

### 1. **Reduced Learning Rate** 🔽

```yaml
learning_rate: 0.0008 # Was: 0.0015 (47% reduction)
```

**Why:** Unfrozen ViT needs gentler learning to avoid destroying pretrained features

### 2. **Reduced Batch Size** 🔽

```yaml
batch_size: 32 # Was: 64 (50% reduction)
```

**Why:**

- More gradient updates per epoch (better for small dataset)
- More stable gradients with unfrozen backbone
- 229 samples ÷ 32 = 7 batches/epoch (vs 3.6 before)

### 3. **Disabled Mixup** ❌

```yaml
mixup_prob: 0.0 # Was: 0.15
```

**Why:**

- Tile augmentation (0.8) already provides 6x data
- Mixup + tile aug + unfrozen backbone = too much noise
- Simplify training dynamics

### 4. **Slower Backbone Learning** 🐌

```yaml
backbone_lr_factor: 0.2 # Was: 0.3
```

**Why:**

- Backbone should adapt slowly (it's pretrained)
- Effective backbone LR: 0.0008 × 0.2 = **0.00016**
- Heads LR: **0.0008** (5x faster than backbone)

### 5. **Increased Regularization** 📈

```yaml
weight_decay: 0.08 # Was: 0.05
```

**Why:** Unfrozen backbone has more parameters to regularize

## Expected Improvements

### Training Dynamics:

- ✅ **Smoother loss curves** (less oscillation)
- ✅ **More stable R² metrics** (no wild swings)
- ✅ **Better train/val alignment** (less overfitting)
- ✅ **Cleaner component plots** (HSV, biomass, aux)

### Performance:

- ✅ **Better generalization** from stable training
- ✅ **Higher final R²** from proper convergence
- ✅ **More reliable model selection** (less noise in validation)

### Training Speed:

- ⚠️ **Slower per epoch** (32 vs 64 batch size)
- ✅ **But more epochs needed anyway** (60 total)
- ✅ **Better final result** worth the extra time

## Key Metrics to Monitor

1. **Loss curves should be smooth** (not oscillating)
2. **R² should steadily increase** (no negative values after epoch 2-3)
3. **Train/Val gap should be small** (< 0.1 R² difference)
4. **Component plots should converge** (all targets improving together)

## Comparison: Before vs After

| Metric                | Previous (Noisy) | Current (Stable) |
| --------------------- | ---------------- | ---------------- |
| Learning Rate         | 0.0015           | 0.0008           |
| Batch Size            | 64               | 32               |
| Mixup                 | 0.15             | 0.0              |
| Backbone LR Factor    | 0.3              | 0.2              |
| Weight Decay          | 0.05             | 0.08             |
| Effective Backbone LR | 0.00045          | 0.00016          |
| Batches/Epoch         | 3.6              | 7.2              |

## What Stayed the Same (Good Things)

✅ **Backbone unfrozen** - Still getting domain adaptation  
✅ **Smart ViT unfreezing** - Last 4 blocks only  
✅ **Tile augmentation** - Still 0.8 (6x data)  
✅ **60 epochs** - Enough time to converge  
✅ **Task weights** - Biomass-focused (15.0)

## Next Steps

1. **Restart training** with stabilized config
2. **Check first 5 epochs** - should see smooth curves
3. **Monitor epoch 10** - R² should be > 0.4 on holdout
4. **Let it converge** - should reach 0.60-0.65 R² by epoch 40-50

## Success Criteria

Training is stable if:

- Loss curves are smooth (no spikes)
- R² increases monotonically (with small fluctuations)
- Val/Holdout losses track each other closely
- No NaN or inf values

If still noisy, further reduce LR to 0.0005.
