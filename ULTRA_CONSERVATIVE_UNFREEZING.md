# Ultra-Conservative Backbone Unfreezing - 2026-01-24

## **Current Performance:**

- **Frozen Backbone (fusion=384)**: LB = 0.55, CV = 0.56
- **Target**: LB = 0.68 (need +0.13 improvement)
- **Strategy**: Careful domain adaptation via minimal backbone unfreezing

---

## **Configuration: Ultra-Conservative Unfreezing**

### **Key Principle:**

> "Unfreeze as little as possible, as slowly as possible, with maximum regularization"

---

## **Changes Made:**

### **1. Backbone Unfreezing (Minimal)**

```yaml
freeze_backbone: false # Was true
```

**In train_unified_holdout.py:**

```python
unfreeze_last_n = 1  # Only LAST 1 transformer block (out of 12)
```

**What's Frozen:**

- ✅ Patch embedding (keeps general visual features)
- ✅ First 11 transformer blocks (keeps DINOv3 features)

**What's Unfrozen:**

- 🔓 Last 1 transformer block (learns biomass-specific features)
- 🔓 Normalization layer (adapts feature scales)

**Rationale**:

- Minimal unfreezing = minimal risk
- Last block is most important for task-specific adaptation
- Early blocks contain general features we want to preserve

---

### **2. Ultra-Low Learning Rates**

```yaml
learning_rate: 0.0004 # Was 0.0007 (-43%)
backbone_lr_factor: 0.015 # Was 0.05 (-70%)
```

**Effective Learning Rates:**

- **Head LR**: 0.0004 (very conservative)
- **Backbone LR**: 0.000006 (1/67th of head = ultra-slow)

**Comparison:**
| Component | Previous | New | Ratio |
|-----------|----------|-----|-------|
| Head | 0.0007 | 0.0004 | 0.57x |
| Backbone | 0.000035 | 0.000006 | 0.17x |

**Rationale**:

- Backbone learns **67x slower** than heads
- Prevents catastrophic forgetting of DINOv3 features
- Allows gentle adaptation over many epochs

---

### **3. Strong Regularization**

```yaml
weight_decay: 0.15 # Was 0.10 (+50%)
```

**Rationale**:

- Unfrozen backbone has more capacity → needs more regularization
- Prevents overfitting to training set
- Encourages sparse, generalizable weight updates

---

### **4. Extended Warmup (10 Epochs)**

**Previous:**

```python
warmup_epochs = 5
warmup_factor = 0.3 + 0.7 * (epoch / 5)  # 30% → 100%
```

**New:**

```python
warmup_epochs = 10
warmup_factor = 0.15 + 0.85 * (epoch / 10)  # 15% → 100%
```

**Warmup Schedule:**

```
Epoch 0:  LR = 0.00006  (15% of 0.0004)
Epoch 1:  LR = 0.00010  (24%)
Epoch 2:  LR = 0.00013  (32%)
Epoch 3:  LR = 0.00016  (41%)
Epoch 4:  LR = 0.00020  (49%)
Epoch 5:  LR = 0.00023  (58%)
Epoch 6:  LR = 0.00026  (66%)
Epoch 7:  LR = 0.00030  (75%)
Epoch 8:  LR = 0.00033  (83%)
Epoch 9:  LR = 0.00036  (91%)
Epoch 10: LR = 0.00040  (100%) ← Full LR reached
```

**Rationale**:

- Very gentle ramp prevents early instability
- Gives model time to adjust to unfrozen backbone
- Prevents the "epoch 5 collapse" we saw before

---

### **5. Maintained Capacity & Regularization**

```yaml
fusion_dim: 384 # Kept from previous experiment
aux_feat_weight: 0.1 # Gentle regularization
species_feat_weight: 0.05 # Light species signal
consistency_weight: 5.0 # Physics constraint
use_species_count_feature: false # Disabled (noisy)
```

---

## **Complete Configuration Summary:**

```yaml
# Hyperparameters
learning_rate: 0.0004 # Ultra-conservative
backbone_lr_factor: 0.015 # 1/67th of head LR
batch_size: 32
weight_decay: 0.15 # Strong regularization
epochs: 60
early_stop_patience: 10

# Architecture
backbone: DINOv3 (last 1 block unfrozen)
fusion_dim: 384

# Training
freeze_backbone: false # Unfrozen
warmup: 10 epochs (15% → 100%)

# Loss Weights
biomass: 50.0
aux: 0.1
species: 0.05
consistency: 5.0
```

---

## **Expected Behavior:**

### **Warmup Phase (Epochs 0-10):**

- Very gentle LR ramp
- Model adjusts to unfrozen backbone
- Minimal feature drift
- **Expected**: Stable, gradual improvement

### **Stable Phase (Epochs 10-30):**

- Full LR reached
- Backbone adapts slowly to biomass domain
- Heads learn to use adapted features
- **Expected**: Steady R² improvement

### **Convergence Phase (Epochs 30-50):**

- Scheduler may reduce LR if plateau detected
- Fine-tuning of domain-specific features
- **Expected**: Final push to 0.65-0.68

---

## **Risk Mitigation:**

### **What Could Go Wrong:**

1. **Backbone collapse** - Features degrade, R² drops
2. **Overfitting** - Train R² high, Val/Holdout low
3. **Instability** - Noisy loss curves, oscillations

### **Safety Mechanisms:**

✅ **Minimal unfreezing** (only 1 block)
✅ **Ultra-low backbone LR** (1/67th of head)
✅ **Strong regularization** (weight_decay=0.15)
✅ **Extended warmup** (10 epochs)
✅ **Early stopping** (patience=10)
✅ **Gradient clipping** (max_grad_norm=1.0)
✅ **ReduceLROnPlateau** (auto LR reduction)

---

## **Monitoring Checklist:**

### **Critical Metrics:**

**During Warmup (Epochs 0-10):**

- [ ] **No sudden R² drops** (should improve gradually)
- [ ] **Train/Val gap < 0.20** (not overfitting)
- [ ] **Holdout R² tracks Val R²** (generalizing)
- [ ] **Losses decreasing smoothly** (no oscillations)

**After Warmup (Epochs 10+):**

- [ ] **Val R² > 0.60** by epoch 15 (on track)
- [ ] **Holdout R² > 0.62** by epoch 20 (good generalization)
- [ ] **Target: Val R² > 0.65** (competitive)
- [ ] **Target: Holdout R² > 0.66** (Kaggle ~0.68)

### **Warning Signs:**

**Immediate Stop If:**

- ❌ **Holdout R² drops > 0.10** from best
- ❌ **Train/Val gap > 0.30** (severe overfitting)
- ❌ **Negative R²** on validation/holdout
- ❌ **NaN losses** (training collapse)

**Consider Reverting If:**

- ⚠️ **No improvement by epoch 20** (R² still < 0.58)
- ⚠️ **Holdout R² < frozen baseline** (0.56) after warmup
- ⚠️ **Very noisy loss curves** (high variance)

---

## **Expected Results:**

### **Conservative Estimate:**

- **Local CV**: 0.56 → **0.62-0.65** (+0.06-0.09)
- **Kaggle LB**: 0.55 → **0.63-0.66** (+0.08-0.11)

### **Optimistic Estimate:**

- **Local CV**: 0.56 → **0.66-0.68** (+0.10-0.12)
- **Kaggle LB**: 0.55 → **0.67-0.70** (+0.12-0.15)

### **Why This Should Work:**

**Domain Adaptation:**

- Unfrozen block learns biomass-specific features
- Adapts DINOv3's general features to pasture imagery
- Learns to distinguish green/dead/clover visually

**Stability:**

- Ultra-conservative settings minimize risk
- Extended warmup prevents early collapse
- Strong regularization prevents overfitting

**Capacity:**

- fusion_dim=384 provides room for complex patterns
- Auxiliary tasks provide regularization
- Physics constraint ensures realistic predictions

---

## **Next Steps Based on Results:**

### **Scenario 1: Success (LB 0.65-0.70)**

✅ **Achieved target!**

- Submit to Kaggle
- Consider ensemble with frozen model (0.55) for diversity
- Possible minor tuning: increase aux_weight to 0.15

### **Scenario 2: Good Progress (LB 0.60-0.64)**

⚠️ **Close but not quite**

- Try unfreezing 2 blocks instead of 1
- Or increase backbone_lr_factor to 0.02
- Or reduce weight_decay to 0.12

### **Scenario 3: No Improvement (LB < 0.58)**

❌ **Unfreezing didn't help**

- Revert to frozen backbone
- Focus on data augmentation
- Consider different backbone architecture

---

## **Training Time:**

**Estimated**: ~5-6 hours for 3 folds

- Slower than frozen (backward pass through unfrozen block)
- But still reasonable for experimentation

---

## **Summary:**

This configuration represents the **safest possible approach** to backbone unfreezing:

- ✅ Minimal unfreezing (1 block)
- ✅ Ultra-slow learning (1/67th LR)
- ✅ Strong regularization (0.15 weight decay)
- ✅ Extended warmup (10 epochs)

**If this doesn't reach 0.68, nothing will** (without major architecture changes or more data).

**Good luck! This is your best shot at 0.68!** 🎯🚀
