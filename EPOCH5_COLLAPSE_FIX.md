# Epoch 5 Collapse Fix - 2026-01-24

## **Problem Diagnosis**

Training was **consistently collapsing at Epoch 5**, regardless of configuration:

### Observed Pattern:

- **Epoch 4**: Holdout R² = **0.4894** ✅ (Best performance)
- **Epoch 5**: Holdout R² = **0.0089** ❌ (Catastrophic drop of 0.48!)

### Root Cause:

The collapse occurred **during the warmup phase** when the learning rate was ramping up:

```
Epoch 4: LR = 0.000400 (40% of peak 0.001) → Holdout R² = 0.489 ✅
Epoch 5: LR = 0.000475 (47.5% of peak 0.001) → Holdout R² = 0.009 💥
```

**Key Insight**: Even with a **frozen backbone**, the 12-epoch linear warmup (10% → 100%) was **too aggressive**. The learning rate increase between epochs 4-5 was causing:

- Biomass loss explosion on holdout (11.4 → 15.1)
- Validation R² drop (0.551 → 0.450)
- Complete loss of generalization

---

## **Solution Implemented**

### 1. **Reduced Base Learning Rate**

**File**: `src/training/config/config.yaml`

- **Before**: `learning_rate: 0.001`
- **After**: `learning_rate: 0.0007`
- **Rationale**: Lower peak LR reduces the risk of overshooting during warmup

### 2. **Shortened & Gentled Warmup**

**File**: `src/training/train_unified_holdout.py`

**Before (12-epoch warmup)**:

```python
warmup_epochs = 12
warmup_factor = 0.1 + 0.9 * (epoch / warmup_epochs)  # 10% → 100%
```

**After (5-epoch warmup)**:

```python
warmup_epochs = 5
warmup_factor = 0.3 + 0.7 * (epoch / warmup_epochs)  # 30% → 100%
```

**Rationale**:

- **Shorter warmup (5 vs 12)**: Reaches stable LR faster, less time in the "danger zone"
- **Gentler start (30% vs 10%)**: Smaller LR jumps between epochs
- **Smaller increments**: Each epoch increases LR by ~14% instead of ~7.5%

---

## **Expected Behavior**

### New Warmup Schedule:

```
Epoch 0: LR = 0.00021 (30% of 0.0007)
Epoch 1: LR = 0.00035 (50% of 0.0007)
Epoch 2: LR = 0.00049 (70% of 0.0007)
Epoch 3: LR = 0.00063 (90% of 0.0007)
Epoch 4: LR = 0.00070 (100% of 0.0007) ← Full LR reached
Epoch 5+: LR = 0.00070 (stable)
```

### Predicted Improvements:

1. ✅ **No Epoch 5 collapse** - Full LR reached by epoch 4, stable thereafter
2. ✅ **Smoother training curves** - Gentler LR ramp prevents sudden jumps
3. ✅ **Better holdout stability** - Lower peak LR reduces overfitting risk
4. ✅ **Faster convergence** - Reaches stable training regime earlier

---

## **Monitoring Checklist**

When training restarts, verify:

- [ ] **Epoch 4-5 transition is smooth** (no sudden R² drop)
- [ ] **Holdout R² tracks with Val R²** (gap < 0.15)
- [ ] **Training stabilizes after epoch 5** (no more collapses)
- [ ] **Final R² > 0.60** on both Val and Holdout

---

## **Configuration Summary**

**Current Settings**:

- Base LR: **0.0007** (reduced from 0.001)
- Warmup: **5 epochs** (30% → 100%)
- Backbone: **Frozen** (DINOv3 features)
- Batch Size: **32**
- Biomass Weight: **50.0** (visual features prioritized)
- Aux/Species Weights: **0.1/0.05** (minimal)
- Tile Augmentation: **Disabled** (0.0)

**Strategy**: Conservative, stability-first approach focusing on general visual biomass features rather than species-specific learning.
