# Experiment: Regularization + Capacity Boost - 2026-01-24

## **Baseline Performance:**

- **Local CV R²**: 0.59 (Holdout)
- **Kaggle LB**: 0.54
- **Gap**: -0.05 (good - validation is representative)
- **Target**: 0.68+ (need +0.14 improvement)

---

## **Changes Made:**

### **Step 1: Re-enable Auxiliary Tasks (Gentle Regularization)**

**Previous:**

```yaml
aux_feat_weight: 0.05 # Minimal
species_feat_weight: 0.0 # Disabled
```

**New:**

```yaml
aux_feat_weight: 0.1 # Gentle regularization
species_feat_weight: 0.05 # Very light species signal
```

**Loss Distribution:**

- Biomass: 50.0 (99.7%)
- Consistency: 5.0 (10.0%)
- Aux: 0.1 (0.2%)
- Species: 0.05 (0.1%)
- **Total**: ~50.15

**Rationale:**

- Auxiliary tasks act as **regularizers** preventing overfitting
- Species signal provides **implicit data augmentation**
- Multi-task learning encourages **richer feature representations**
- Still **99.7% biomass-focused** - auxiliary is just a gentle nudge

### **Step 2: Increase Model Capacity**

**Previous:**

```yaml
fusion_dim: 256
```

**New:**

```yaml
fusion_dim: 384 # +50% capacity
```

**Impact:**

- Fusion layer: `(backbone_dim + aux + species) → 384 → 128 → 5`
- More parameters in the critical fusion stage
- Can learn **more complex non-linear mappings**
- Better capacity to integrate multi-modal features

---

## **Expected Improvements:**

### **From Auxiliary Re-enabling:**

- **+0.02-0.04** R² improvement
- Better regularization → less overfitting
- Richer features → better generalization
- Multi-task gradients → more robust learning

### **From Increased Capacity:**

- **+0.02-0.03** R² improvement
- More expressive fusion layer
- Better integration of image + auxiliary features
- Can model more complex biomass patterns

### **Combined Expected:**

- **Local CV**: 0.59 → **0.63-0.66**
- **Kaggle LB**: 0.54 → **0.58-0.62**

---

## **Why This Should Work:**

### **1. Regularization Hypothesis (User's Insight)**

> "I believe if we allow small presence of species and other features (0.05) they act as regularizers"

**Evidence:**

- Image-only models achieving 0.68 likely use multi-task learning
- Small auxiliary weights prevent overfitting to biomass noise
- Species classification forces model to learn species-agnostic visual features

### **2. Capacity Bottleneck**

Current fusion layer (256 dim) may be **too small** to:

- Integrate backbone features (384 dim) + aux (3 dim) + species (14 dim)
- Learn complex non-linear relationships
- Model interactions between different feature types

Increasing to 384 provides **breathing room** for the fusion layer.

### **3. Frozen Backbone + Rich Features**

Since backbone is frozen:

- Can't learn domain-specific low-level features
- Must rely on **high-quality feature fusion**
- Larger fusion_dim compensates for frozen backbone limitation

---

## **Training Configuration:**

```yaml
# Hyperparameters (unchanged)
learning_rate: 0.0007
batch_size: 32
weight_decay: 0.10
epochs: 60
warmup: 5 epochs (30% → 100%)

# Architecture
backbone: DINOv3 (frozen)
fusion_dim: 384 (was 256)

# Loss Weights
biomass: 50.0
aux: 0.1
species: 0.05
consistency: 5.0

# Regularization
- Weight decay: 0.10
- Dropout: 0.3
- LayerNorm (stable)
- Physics consistency penalty
```

---

## **Monitoring Checklist:**

During training, watch for:

### **Positive Signs:**

- [ ] **Val R² > 0.62** (improvement from 0.59)
- [ ] **Holdout R² > 0.63** (improvement from 0.59)
- [ ] **Train/Val gap < 0.15** (not overfitting)
- [ ] **Smooth convergence** (no oscillations)
- [ ] **Species loss is stable** (not causing noise)

### **Warning Signs:**

- [ ] **Species loss oscillating** → Reduce species_weight to 0.02
- [ ] **Train/Val gap widening** → Increase weight_decay to 0.12
- [ ] **Holdout collapse** → Revert to previous config

---

## **Next Steps Based on Results:**

### **Scenario 1: Local CV 0.63-0.66, LB 0.59-0.62**

✅ **Success!** Regularization + capacity worked

- Submit to Kaggle
- If LB < 0.62, proceed to **Step 3** (careful backbone unfreezing)

### **Scenario 2: Local CV 0.60-0.62, LB 0.56-0.58**

⚠️ **Modest improvement**

- Try increasing aux_weight to 0.15, species to 0.1
- Or proceed directly to **Step 3** (backbone unfreezing)

### **Scenario 3: Local CV < 0.60, LB < 0.56**

❌ **No improvement or regression**

- Revert changes
- Auxiliary tasks may be hurting, not helping
- Focus on **Step 3** (backbone unfreezing) as primary strategy

---

## **Step 3 Preview (If Needed):**

If this experiment gets you to ~0.61-0.62 LB, the next step is:

**Careful Backbone Unfreezing:**

```yaml
freeze_backbone: false
learning_rate: 0.0005 # Lower
backbone_lr_factor: 0.02 # Very conservative (1/50th)
weight_decay: 0.12 # More regularization
```

**Unfreeze only last 1 ViT block** (ultra-conservative)

**Expected**: +0.06-0.08 → **0.67-0.70 LB**

---

## **Summary:**

**This experiment tests the hypothesis that:**

1. Auxiliary tasks provide **regularization** (not just noise)
2. Increased capacity allows **better feature fusion**
3. Frozen backbone can reach **0.60-0.62** with right architecture

**If successful**, this validates the multi-task learning approach and sets up for the final push to 0.68+ with careful backbone unfreezing.

**Training time**: ~3-4 hours (3 folds × 25 epochs × 2.5 min/epoch)

---

**Good luck! Let's see if regularization + capacity gets us closer to 0.68!** 🚀
