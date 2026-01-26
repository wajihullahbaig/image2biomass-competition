# Dead Biomass Target Strategy - Expert Council Recommendations

## Problem Statement

The **Dry_Dead_g** target is the hardest to predict in the Image2Biomass competition. According to the Kaggle discussion:

> **Dead means whatever is already dead naturally on the ground**
>
> - Yellowish/white dry parts = Dead
> - Dead material is **senescent** (naturally died and fell to the ground)
> - This is distinct from "Dry Green" which is stressed but still standing vegetation

### Key Challenge

Dead biomass has **two distinct visual signatures**:

1. **Visually Observable Dead**: Yellowish-brown senescent material visible in RGB images (HSV detectable)
2. **Hidden/Occluded Dead**: Dead material hidden under green vegetation, in shadows, or at ground level (NOT HSV detectable)

## Current Implementation Analysis

### ✅ What's Working

1. **HSV-based Dead Detection** (lines 1063-1068 in `common.py`):

   ```python
   dead_lower = np.array([10, 40, 40])  # Hue: 10-25 (yellow-brown)
   dead_upper = np.array([25, 200, 255])
   ```

   - Good hue range for senescent material
   - Captures visible dead grass/plants

2. **Huber Loss for Dead** (config.yaml line 35-37):

   ```yaml
   use_huber_loss_for_dead: true
   huber_delta: 1.0
   ```

   - Robust to label noise (good for mixed visible/hidden dead)

3. **Dead HSV Score as Auxiliary Feature**:
   - Fed into model as `dead_hsv` feature
   - Helps model learn correlation between visual dead and total dead

### ❌ What's Missing

1. **No Explicit Derivation Logic**:
   - Model treats Dead as a direct prediction target
   - Doesn't leverage the physics constraint: `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g`

2. **HSV Dead Score Underutilized**:
   - Currently just an auxiliary feature
   - Should be a **strong prior** for minimum dead biomass

3. **No Visibility-Aware Loss**:
   - Samples with high `dead_hsv` score → Dead is **visible** → trust direct prediction
   - Samples with low `dead_hsv` score → Dead is **hidden** → derive from Total - (Green + Clover)

## Expert Council Recommendations

### 🎯 Strategy 1: Dual-Path Dead Prediction (RECOMMENDED)

**Concept**: Predict Dead via two pathways and blend based on visibility

```python
# In model forward pass:
dead_direct = biomass_head_direct(features)  # Direct RGB→Dead prediction
dead_derived = total_pred - (green_pred + clover_pred)  # Physics-based derivation

# Visibility gate (learned or HSV-based)
visibility_score = dead_hsv_feature  # From auxiliary features
dead_final = visibility_score * dead_direct + (1 - visibility_score) * dead_derived
```

**Advantages**:

- ✅ Leverages both visual cues AND physics constraints
- ✅ Automatically adapts to visibility conditions
- ✅ Maintains differentiability for end-to-end training

**Implementation**:

1. Add `dead_visibility_gate` to model architecture
2. Modify loss to encourage:
   - High-visibility samples → trust direct prediction
   - Low-visibility samples → trust derived prediction
3. Add consistency loss: `L_consistency = |dead_direct - dead_derived|` weighted by uncertainty

---

### 🎯 Strategy 2: HSV-Guided Minimum Dead Constraint

**Concept**: Use HSV dead score as a **lower bound** for dead biomass

```python
# In training loss:
dead_hsv_score = aux_features[:, dead_hsv_idx]  # 0-1 score
dead_pred_linear = torch.expm1(dead_pred_log)

# Minimum dead mass based on visible dead (heuristic: 5g per 10% visible dead)
min_dead_from_hsv = dead_hsv_score * 50.0  # Tunable scaling factor

# Penalty for predicting less than visible minimum
hsv_violation = F.relu(min_dead_from_hsv - dead_pred_linear)
loss_dead_hsv_constraint = hsv_violation.mean() * 2.0  # Add to total loss
```

**Advantages**:

- ✅ Simple to implement (just add one loss term)
- ✅ Prevents physically impossible predictions (can't predict 0g dead when 30% of image is yellow)
- ✅ Doesn't require architecture changes

---

### 🎯 Strategy 3: Visibility-Stratified Training

**Concept**: Split training data into visibility cohorts and train specialized heads

```python
# In dataset/feature engineering:
df['dead_visibility'] = df['dead_hsv_score'].apply(
    lambda x: 'high' if x > 0.15 else ('medium' if x > 0.05 else 'low')
)

# In model:
class DeadPredictionHead(nn.Module):
    def __init__(self):
        self.high_visibility_head = nn.Linear(...)
        self.low_visibility_head = nn.Linear(...)
        self.visibility_router = nn.Linear(...)  # Learns to route

    def forward(self, features, dead_hsv_score):
        router_logits = self.visibility_router(features)
        router_weights = F.softmax(router_logits, dim=1)

        pred_high = self.high_visibility_head(features)
        pred_low = self.low_visibility_head(features)

        dead_pred = router_weights[:, 0] * pred_high + router_weights[:, 1] * pred_low
        return dead_pred
```

**Advantages**:

- ✅ Specialized expertise for different visibility conditions
- ✅ Can learn different feature importance (high-vis → RGB features, low-vis → Total/Green/Clover)

**Disadvantages**:

- ⚠️ More complex architecture
- ⚠️ Requires sufficient data in each visibility cohort

---

### 🎯 Strategy 4: Enhanced HSV Dead Detection

**Concept**: Improve the HSV dead mask to capture more senescent material

**Current HSV Range** (lines 1063-1068):

```python
dead_lower = np.array([10, 40, 40])  # Hue: 10-25
dead_upper = np.array([25, 200, 255])
```

**Proposed Improvements**:

1. **Expand Hue Range** to capture pale/white dead material:

   ```python
   # Dead matter can be very pale (almost white) or dark brown
   dead_lower_1 = np.array([10, 40, 40])   # Yellow-brown
   dead_upper_1 = np.array([30, 200, 255])

   dead_lower_2 = np.array([0, 0, 180])    # Very pale/white (high value, low sat)
   dead_upper_2 = np.array([180, 30, 255])

   dead_mask = cv2.bitwise_or(
       cv2.inRange(hsv, dead_lower_1, dead_upper_1),
       cv2.inRange(hsv, dead_lower_2, dead_upper_2)
   )
   ```

2. **Texture-Based Dead Detection** (advanced):
   - Dead grass has distinct texture (dry, brittle, less uniform)
   - Use Gabor filters or local binary patterns (LBP) to detect texture
   - Combine with HSV for robust detection

3. **Multi-Scale Dead Detection**:
   - Dead material at ground level appears smaller/darker
   - Apply HSV detection at multiple image scales
   - Aggregate scores with spatial weighting

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 hours)

1. ✅ **Implement Strategy 2** (HSV-Guided Minimum Constraint)
   - Add `loss_dead_hsv_constraint` to training loop
   - Tune scaling factor (start with 50.0, adjust based on validation)

2. ✅ **Enhance HSV Dead Detection** (Strategy 4.1)
   - Expand hue range to [10, 30]
   - Add pale/white dead detection
   - Visualize masks to verify improvement

### Phase 2: Architecture Refactor (4-6 hours)

3. ✅ **Implement Strategy 1** (Dual-Path Prediction)
   - Add `dead_derived` pathway to model
   - Implement visibility gate using `dead_hsv` feature
   - Add consistency loss between direct and derived predictions

### Phase 3: Advanced Optimization (8+ hours)

4. ⚠️ **Implement Strategy 3** (Visibility-Stratified Heads) - **ONLY if Phase 1-2 don't reach 0.68+ R²**
5. ⚠️ **Texture-Based Dead Detection** (Strategy 4.2) - **Research task**

---

## Code Refactoring Checklist

### Files to Modify

#### 1. `src/training/common.py`

- [ ] Update `get_hsv_biomass_scores()` to expand dead detection range
- [ ] Add `get_dead_visibility_category()` helper function
- [ ] Add pale/white dead mask detection

#### 2. `src/training/models.py`

- [ ] Add `dead_derived` computation in `forward()`
- [ ] Add `visibility_gate` layer (optional for Strategy 1)
- [ ] Modify biomass_head output to include both direct and derived dead

#### 3. `src/training/train_unified_holdout.py`

- [ ] Add `loss_dead_hsv_constraint` in `train_one_epoch()` and `validate()`
- [ ] Add `loss_dead_consistency` for dual-path (Strategy 1)
- [ ] Log new loss components

#### 4. `src/training/config/config.yaml`

- [ ] Add `dead_hsv_constraint_weight: 2.0`
- [ ] Add `dead_consistency_weight: 1.0`
- [ ] Add `dead_visibility_threshold: 0.10`
- [ ] Update HSV ranges in `features.hsv_biomass_scores.dead_matter`

#### 5. `src/training/config/schemas.py`

- [ ] Add schema validation for new config parameters

---

## Expected Performance Improvements

### Baseline (Current)

- Dead R²: ~0.45-0.55 (estimated from "hardest target" comment)
- Overall R²: ~0.60-0.65

### After Phase 1 (HSV Constraint + Enhanced Detection)

- Dead R²: **0.55-0.65** (+0.10)
- Overall R²: **0.63-0.67** (+0.03)

### After Phase 2 (Dual-Path Prediction)

- Dead R²: **0.65-0.75** (+0.20)
- Overall R²: **0.66-0.70** (+0.06)

### Target

- Dead R²: **0.70+**
- Overall R²: **0.68+** ✅ Kaggle competitive

---

## Physics Constraints to Enforce

```python
# Hard constraints (always true):
1. Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g
2. GDM_g = Dry_Green_g + Dry_Clover_g
3. All biomass >= 0

# Soft constraints (usually true, enforce with loss):
4. dead_pred >= dead_hsv_score * k  (k ≈ 50g per 10% visible dead)
5. If dead_hsv_score > 0.2 → dead_pred should be "high confidence" (low variance)
6. If dead_hsv_score < 0.05 → dead_pred should rely on Total - (Green + Clover)
```

---

## Validation Strategy

### Metrics to Track

1. **Dead R² by Visibility Cohort**:
   - High visibility (dead_hsv > 0.15): Should be easiest
   - Low visibility (dead_hsv < 0.05): Currently hardest, biggest opportunity

2. **Physics Violation Rate**:
   - % of samples where `dead_pred < dead_hsv_score * 30` (impossible)
   - % of samples where `Total != Green + Dead + Clover` (within 5% tolerance)

3. **Consistency Score**:
   - Correlation between `dead_direct` and `dead_derived` predictions
   - Should be high (>0.8) for well-calibrated model

### Ablation Tests

- [ ] Baseline (current model)
- [ ] - HSV constraint only
- [ ] - Enhanced HSV detection only
- [ ] - Both (Phase 1 complete)
- [ ] - Dual-path (Phase 2 complete)

---

## Key Insights from Kaggle Discussion

> "Dead means whatever is already dead naturally on the ground"

**Implications**:

1. Dead is **not** just "dry" (Dry_Green exists separately)
2. Dead is **ground-level** → often occluded → HSV may miss it
3. Dead is **senescent** → specific color (yellow-white-brown) but variable
4. Dead is **passive** → doesn't correlate with NDVI/Height as strongly as Green

**Model should learn**:

- High NDVI + Low Dead HSV → Likely hidden dead under canopy
- Low NDVI + High Dead HSV → Visible senescent material
- Low NDVI + Low Dead HSV → Either bare soil OR dense dead (ambiguous)

---

## Questions for Further Investigation

1. **What's the distribution of `dead_hsv_score` in the training set?**
   - If most samples have low scores → hidden dead is dominant → prioritize derivation
   - If bimodal → visibility-stratified approach is ideal

2. **What's the correlation between Dead and other targets?**
   - Dead vs Total: Should be strong (Total includes Dead)
   - Dead vs Green: Should be negative (inverse relationship)
   - Dead vs Height: Should be weak/negative (dead is ground-level)

3. **Are there seasonal patterns in Dead visibility?**
   - Winter → More dead, more visible (less green canopy)
   - Spring → Less dead, more hidden (dense green growth)

---

## Next Steps

**Immediate Actions**:

1. Run EDA on `dead_hsv_score` distribution
2. Implement Phase 1 (Quick Wins)
3. Train and validate
4. If Dead R² improves by >0.05 → proceed to Phase 2
5. If Dead R² improves by <0.05 → investigate data quality or try Strategy 3

**Success Criteria**:

- Dead R² > 0.65
- Overall R² > 0.68
- Physics violation rate < 5%
- No degradation in Green/Clover/Total R²

---

## References

- Kaggle Discussion: "Dry Dead vs Dry Green" (Qiyu Liao, Competition Host)
- Current Config: `config.yaml` lines 35-37, 63-68
- Current HSV Implementation: `common.py` lines 1019-1095
- Current Model: `models.py` lines 124-136
