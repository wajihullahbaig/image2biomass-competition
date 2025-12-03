import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings

# --- Configuration ---
warnings.filterwarnings('ignore')
BASE_VIZ_PATH = 'visualizations_detailed'
os.makedirs(BASE_VIZ_PATH, exist_ok=True)

# Visual Settings for Publication Quality
sns.set_theme(style="ticks", context="notebook")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# --- Helper Functions ---
def get_season(month):
    if month in [12, 1, 2]: return 'Summer'
    elif month in [3, 4, 5]: return 'Autumn'
    elif month in [6, 7, 8]: return 'Winter'
    else: return 'Spring'

def process_data(filepath='train.csv'):
    print("1. Loading and Pivoting Data...")
    df = pd.read_csv(filepath)
    
    # 1. Clean IDs
    df['clean_id'] = df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
    
    # 2. Pivot to Wide Format (One row per physical plot)
    # We aggregate by max to merge rows, assuming constant metadata
    pivot_cols = ['clean_id', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    valid_cols = [c for c in pivot_cols if c in df.columns]
    
    # Pivot target values
    targets = df.pivot_table(index='clean_id', columns='target_name', values='target', aggfunc='max').reset_index()
    
    # Get Metadata
    meta = df[valid_cols].drop_duplicates(subset=['clean_id'])
    
    # Merge
    wide = pd.merge(meta, targets, on='clean_id', how='left')
    
    # 3. Fill Missing Targets with 0 (Standard assumption for biomass components)
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    for t in target_cols:
        if t not in wide.columns: wide[t] = 0.0
    wide[target_cols] = wide[target_cols].fillna(0.0)
    
    # 4. Feature Engineering
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'])
    wide['Month'] = wide['Sampling_Date'].dt.month
    wide['Season'] = wide['Month'].apply(get_season)
    
    # Log Transforms for Skewed Targets (for visualization)
    for t in target_cols:
        wide[f'Log_{t}'] = np.log1p(wide[t])

    # 5. Create Train/Validation Split for Distribution Analysis
    # We stratify by State to ensure representativeness
    train_idx, val_idx = train_test_split(wide.index, test_size=0.2, random_state=42, stratify=wide['State'])
    wide['Set'] = 'Train'
    wide.loc[val_idx, 'Set'] = 'Validation'
    
    print(f"   Data Shape: {wide.shape}")
    return wide, target_cols

# ==============================================================================
# 1. PHYSICS CONSISTENCY & COMPOSITION
# ==============================================================================
def viz_physics(df):
    print("2. Generating Physics Consistency Checks...")
    
    # A. The Summation Check
    df['Sum_Components'] = df['Dry_Clover_g'] + df['Dry_Dead_g'] + df['Dry_Green_g']
    df['Physics_Error'] = df['Dry_Total_g'] - df['Sum_Components']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Correlation of Sum vs Total
    sns.scatterplot(data=df, x='Sum_Components', y='Dry_Total_g', hue='Physics_Error', 
                    palette='coolwarm', alpha=0.6, ax=axes[0])
    max_val = max(df['Sum_Components'].max(), df['Dry_Total_g'].max())
    axes[0].plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect Physics (y=x)')
    axes[0].set_title("Constraint: Total = Clover + Dead + Green")
    axes[0].legend()
    
    # Plot 2: Composition by Season (Stacked)
    # Normalize to see percentage composition
    season_comp = df.groupby('Season')[['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']].mean()
    season_comp_pct = season_comp.div(season_comp.sum(axis=1), axis=0) * 100
    
    season_comp_pct = season_comp_pct.reindex(['Summer', 'Autumn', 'Winter', 'Spring']) # Order
    
    season_comp_pct.plot(kind='bar', stacked=True, color=['#d62728', '#7f7f7f', '#2ca02c'], ax=axes[1])
    axes[1].set_title("Biomass Composition Ratio by Season")
    axes[1].set_ylabel("Percentage of Total Mass (%)")
    axes[1].legend(title='Component', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '1_Physics_and_Composition.png'))
    plt.close()

# ==============================================================================
# 2. GROUPED DISTRIBUTIONS (TRAIN VS VALIDATION)
# ==============================================================================
def viz_distributions(df, target_cols):
    print("3. Generating Grouped Target Distributions...")
    
    # We focus on the most important target: Dry_Total_g
    # But checking if Train/Val have same distribution across categories
    
    groups = ['State', 'Season', 'Species']
    
    for group in groups:
        plt.figure(figsize=(14, 6))
        
        # We use Log scale for Y-axis because biomass is exponentially distributed
        # Boxen plots are better than boxplots for large datasets
        sns.boxenplot(
            data=df, 
            x=group, 
            y='Log_Dry_Total_g', 
            hue='Set',
            palette={'Train': '#3498db', 'Validation': '#e74c3c'}
        )
        
        plt.title(f"Train/Val Distribution Mismatch Check: Grouped by {group}", fontsize=14)
        plt.ylabel("Log(1 + Total Biomass)")
        plt.xticks(rotation=45)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_VIZ_PATH, f'2_Dist_by_{group}.png'))
        plt.close()

# ==============================================================================
# 3. CORRELATION MATRICES (LOWER TRIANGLE)
# ==============================================================================
def viz_correlations(df, target_cols):
    print("4. Generating Correlation Matrices...")
    
    # Select numeric features + targets
    cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI'] + target_cols
    corr = df[cols].corr()
    
    # Create Lower Triangle Mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap='RdBu_r', 
        center=0, 
        square=True, 
        linewidths=.5,
        cbar_kws={"shrink": .7}
    )
    
    plt.title("Feature & Target Correlations (Pearson)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '3_Correlation_Matrix.png'))
    plt.close()

# ==============================================================================
# 4. NON-LINEAR INTERACTIONS & EXPLAINABLE PHYSICS
# ==============================================================================
def viz_nonlinear(df):
    print("5. Generating Non-Linear Interaction Plots (Add/Ratio included)...")
    
    # --- Feature Engineering for Visualization ---
    
    # 1. Volumetric (Multiplication): Proxy for Mass ~ Volume * Density
    df['Interaction_Mul'] = df['Height_Ave_cm'] * df['Pre_GSHH_NDVI']
    
    # 2. Addition: (Note: Units differ, cm vs index, so this is dominated by Height)
    df['Interaction_Add'] = df['Height_Ave_cm'] + df['Pre_GSHH_NDVI']
    
    # 3. Ratio 1: Greenness per cm
    # Epsilon prevents division by zero for very short grass
    epsilon = 1e-3
    df['Interaction_Ratio_NDVI_H'] = df['Pre_GSHH_NDVI'] / (df['Height_Ave_cm'] + epsilon)
    
    # 4. Ratio 2: Height per unit Greenness (Structural Index)
    df['Interaction_Ratio_H_NDVI'] = df['Height_Ave_cm'] / (df['Pre_GSHH_NDVI'] + epsilon)
    
    # Subset for clearer plotting (Performance & Visual clutter reduction)
    # We take a random sample of 2000 points if the dataset is large
    plot_df = df.sample(n=min(2000, len(df)), random_state=42)
    
    # Layout: 2 Rows, 3 Columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten() # Flatten to easy indexing 0-5
    
    # --- Row 1: The Basics & The Physics Proxy ---
    
    # Plot 0: Base Height
    sns.regplot(data=plot_df, x='Height_Ave_cm', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'}, ax=axes[0])
    axes[0].set_title("Base: Height vs Biomass")
    
    # Plot 1: Base NDVI
    sns.regplot(data=plot_df, x='Pre_GSHH_NDVI', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'}, ax=axes[1])
    axes[1].set_title("Base: NDVI vs Biomass")
    
    # Plot 2: Multiplication (Usually the Strongest)
    r_mul, _ = pearsonr(plot_df['Interaction_Mul'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Mul', y='Log_Dry_Total_g', 
                order=1, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'green'}, ax=axes[2])
    axes[2].set_title(f"Multiplication (H * NDVI)\nR = {r_mul:.2f} (Volumetric Proxy)")
    
    # --- Row 2: The Arithmetic Interactions ---
    
    # Plot 3: Addition
    r_add, _ = pearsonr(plot_df['Interaction_Add'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Add', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'orange'}, ax=axes[3])
    axes[3].set_title(f"Addition (H + NDVI)\nR = {r_add:.2f}")
    
    # Plot 4: Ratio (NDVI / Height)
    r_rat1, _ = pearsonr(plot_df['Interaction_Ratio_NDVI_H'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Ratio_NDVI_H', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'purple'}, ax=axes[4])
    axes[4].set_title(f"Ratio (NDVI / Height)\nR = {r_rat1:.2f} (Green Density)")
    # Zoom in to 95th percentile to ignore extreme division artifacts
    axes[4].set_xlim(0, plot_df['Interaction_Ratio_NDVI_H'].quantile(0.95))
    
    # Plot 5: Ratio (Height / NDVI)
    r_rat2, _ = pearsonr(plot_df['Interaction_Ratio_H_NDVI'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Ratio_H_NDVI', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'purple'}, ax=axes[5])
    axes[5].set_title(f"Ratio (Height / NDVI)\nR = {r_rat2:.2f} (Structure Index)")
    axes[5].set_xlim(0, plot_df['Interaction_Ratio_H_NDVI'].quantile(0.95))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '4_NonLinear_Interactions_Full.png'))
    plt.close()


def viz_interaction_correlations(df, target_cols):
    print("6. Generating Interaction vs Target Correlation Matrix...")
    
    # 1. Ensure Features Exist (Re-calculating to be safe)
    epsilon = 1e-3
    # Base
    df['Interaction_Mul'] = df['Height_Ave_cm'] * df['Pre_GSHH_NDVI']
    df['Interaction_Add'] = df['Height_Ave_cm'] + df['Pre_GSHH_NDVI']
    df['Interaction_Ratio_NDVI_H'] = df['Pre_GSHH_NDVI'] / (df['Height_Ave_cm'] + epsilon)
    df['Interaction_Ratio_H_NDVI'] = df['Height_Ave_cm'] / (df['Pre_GSHH_NDVI'] + epsilon)
    
    # 2. Setup Lists
    # We use the LOG transformed targets because Pearson correlation assumes linearity,
    # and your previous plots proved the relationship is linear in Log space.
    log_targets = [f'Log_{t}' for t in target_cols]
    
    # Renaming for cleaner plot labels
    feature_map = {
        'Height_Ave_cm': 'Height (Base)',
        'Pre_GSHH_NDVI': 'NDVI (Base)',
        'Interaction_Mul': 'Multiplication (H * NDVI)',
        'Interaction_Add': 'Addition (H + NDVI)',
        'Interaction_Ratio_NDVI_H': 'Ratio (NDVI / H)',
        'Interaction_Ratio_H_NDVI': 'Ratio (H / NDVI)'
    }
    
    # 3. Calculate Correlation Matrix
    # We only want: Rows = Features, Cols = Targets
    features = list(feature_map.keys())
    
    # Calculate full correlation matrix then slice it
    full_corr = df[features + log_targets].corr()
    target_corr = full_corr.loc[features, log_targets]
    
    # Rename index for readability
    target_corr = target_corr.rename(index=feature_map)
    # Rename columns to remove "Log_" and "_g" for cleaner reading
    target_corr.columns = [c.replace('Log_', '').replace('_g', '') for c in target_corr.columns]

    # 4. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        target_corr,
        annot=True,
        fmt=".2f",
        cmap='RdBu_r', # Red = Negative, Blue = Positive
        center=0,
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Pearson Correlation (r)"},
        square=True
    )
    
    plt.title("Which Feature Predicts Which Target?\n(Correlation of Interactions vs Log-Biomass)", fontsize=16)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(BASE_VIZ_PATH, '5_Interaction_Target_Correlation.png')
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Saved to {save_path}")

def viz_inter_target_engineering(df, target_cols):
    """
    Explores engineered features FROM targets to predict difficult targets (like Dry_Dead).
    Creates auxiliary features using inter-target relationships.
    """
    print("7. Engineering Inter-Target Features for Dry_Dead Prediction...")
    
    # Focus on predicting Dry_Dead using other targets as proxies
    # Physical intuition: Dead material might correlate with ratios/differences of living material
    
    # === FEATURE ENGINEERING ===
    epsilon = 1e-3
    
    # 1. Residual-based features (What's "unexplained"?)
    df['Total_minus_Green'] = df['Dry_Total_g'] - df['Dry_Green_g']  # Should ≈ Dead + Clover
    df['Total_minus_Clover'] = df['Dry_Total_g'] - df['Dry_Clover_g']  # Should ≈ Dead + Green
    df['Green_plus_Clover'] = df['Dry_Green_g'] + df['Dry_Clover_g']  # Living biomass
    df['Unexplained_Mass'] = df['Dry_Total_g'] - df['Green_plus_Clover']  # Should ≈ Dead
    
    # 2. Ratio-based features (Composition signals)
    df['Ratio_Total_Green'] = df['Dry_Total_g'] / (df['Dry_Green_g'] + epsilon)
    df['Ratio_Total_Clover'] = df['Dry_Total_g'] / (df['Dry_Clover_g'] + epsilon)
    df['Ratio_Green_Clover'] = df['Dry_Green_g'] / (df['Dry_Clover_g'] + epsilon)
    df['Ratio_Clover_Total'] = df['Dry_Clover_g'] / (df['Dry_Total_g'] + epsilon)
    df['Ratio_Green_Total'] = df['Dry_Green_g'] / (df['Dry_Total_g'] + epsilon)
    
    # 3. Multiplicative interactions (Non-linear signals)
    df['Product_Green_Clover'] = df['Dry_Green_g'] * df['Dry_Clover_g']
    df['Product_Total_Green'] = df['Dry_Total_g'] * df['Dry_Green_g']
    
    # 4. Square/Power features (Capturing non-linear mass dynamics)
    df['Sqrt_Total'] = np.sqrt(df['Dry_Total_g'])
    df['Sqrt_Green'] = np.sqrt(df['Dry_Green_g'])
    df['Square_Green'] = df['Dry_Green_g'] ** 2
    
    # 5. GDM relationship (GDM often predicts Total well, residuals might help)
    df['GDM_minus_Total'] = df['GDM_g'] - df['Dry_Total_g']
    df['Ratio_GDM_Total'] = df['GDM_g'] / (df['Dry_Total_g'] + epsilon)
    
    # === CORRELATION ANALYSIS ===
    
    # List all engineered features
    engineered_features = [
        'Total_minus_Green', 'Total_minus_Clover', 'Green_plus_Clover', 'Unexplained_Mass',
        'Ratio_Total_Green', 'Ratio_Total_Clover', 'Ratio_Green_Clover', 
        'Ratio_Clover_Total', 'Ratio_Green_Total',
        'Product_Green_Clover', 'Product_Total_Green',
        'Sqrt_Total', 'Sqrt_Green', 'Square_Green',
        'GDM_minus_Total', 'Ratio_GDM_Total'
    ]
    
    # Target we care about
    focus_target = 'Dry_Dead_g'
    
    # Calculate correlations with Dry_Dead
    correlations = {}
    for feat in engineered_features:
        # Handle infinite/nan from division
        valid_mask = np.isfinite(df[feat]) & np.isfinite(df[focus_target])
        if valid_mask.sum() > 10:  # Need sufficient data
            r, _ = pearsonr(df.loc[valid_mask, feat], df.loc[valid_mask, focus_target])
            correlations[feat] = r
        else:
            correlations[feat] = 0.0
    
    # Sort by absolute correlation strength
    sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # === VISUALIZATION ===
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- Plot 1: Correlation Bar Chart ---
    feature_names = [f[0] for f in sorted_features]
    feature_corrs = [f[1] for f in sorted_features]
    
    colors = ['#d62728' if c > 0 else '#1f77b4' for c in feature_corrs]
    
    axes[0, 0].barh(feature_names, feature_corrs, color=colors, alpha=0.7)
    axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].set_xlabel('Pearson Correlation with Dry_Dead_g')
    axes[0, 0].set_title('Engineered Feature Strength for Predicting Dry_Dead', fontsize=13)
    axes[0, 0].invert_yaxis()
    
    # --- Plot 2: Top Feature Scatter (Best predictor) ---
    top_feature = sorted_features[0][0]
    top_corr = sorted_features[0][1]
    
    plot_df = df.sample(n=min(1500, len(df)), random_state=42)
    valid_mask = np.isfinite(plot_df[top_feature]) & np.isfinite(plot_df[focus_target])
    plot_df_valid = plot_df[valid_mask]
    
    sns.scatterplot(data=plot_df_valid, x=top_feature, y=focus_target, 
                    alpha=0.4, s=20, color='darkred', ax=axes[0, 1])
    sns.regplot(data=plot_df_valid, x=top_feature, y=focus_target, 
                scatter=False, lowess=True, line_kws={'color': 'black', 'linewidth': 2}, ax=axes[0, 1])
    axes[0, 1].set_title(f'Best Predictor: {top_feature}\nR = {top_corr:.3f}', fontsize=13)
    axes[0, 1].set_ylabel('Dry_Dead_g (Target)')
    
    # Zoom to 95th percentile if ratio feature
    if 'Ratio' in top_feature:
        axes[0, 1].set_xlim(0, plot_df_valid[top_feature].quantile(0.95))
    
    # --- Plot 3: Heatmap of Top Features vs All Targets ---
    top_n = 8
    top_feature_list = [f[0] for f in sorted_features[:top_n]]
    
    # Add original targets for comparison
    all_targets_for_heatmap = target_cols
    
    # Build correlation matrix
    heatmap_data = []
    for feat in top_feature_list:
        row = []
        for target in all_targets_for_heatmap:
            valid_mask = np.isfinite(df[feat]) & np.isfinite(df[target])
            if valid_mask.sum() > 10:
                r, _ = pearsonr(df.loc[valid_mask, feat], df.loc[valid_mask, target])
                row.append(r)
            else:
                row.append(0.0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(
        heatmap_data, 
        index=top_feature_list,
        columns=[t.replace('_g', '') for t in all_targets_for_heatmap]
    )
    
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt='.2f', 
        cmap='RdBu_r', 
        center=0, 
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Correlation (r)"},
        ax=axes[1, 0]
    )
    axes[1, 0].set_title(f'Top {top_n} Features vs All Targets', fontsize=13)
    axes[1, 0].set_ylabel('')
    
    # --- Plot 4: Residual Analysis (Physics Check) ---
    # If we predict Dead using best feature, what's the error pattern?
    
    # Simple linear fit
    from sklearn.linear_model import LinearRegression
    valid_mask = np.isfinite(df[top_feature]) & np.isfinite(df[focus_target])
    X_fit = df.loc[valid_mask, top_feature].values.reshape(-1, 1)
    y_fit = df.loc[valid_mask, focus_target].values
    
    model = LinearRegression()
    model.fit(X_fit, y_fit)
    y_pred = model.predict(X_fit)
    residuals = y_fit - y_pred
    
    axes[1, 1].scatter(y_pred, residuals, alpha=0.3, s=10, color='purple')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel(f'Predicted Dry_Dead (from {top_feature})')
    axes[1, 1].set_ylabel('Residual (Actual - Predicted)')
    axes[1, 1].set_title('Residual Pattern Analysis\n(Check for systematic bias)', fontsize=13)
    
    # Add std bands
    std_resid = np.std(residuals)
    axes[1, 1].axhline(std_resid, color='red', linestyle=':', alpha=0.5, label=f'±1 Std ({std_resid:.2f})')
    axes[1, 1].axhline(-std_resid, color='red', linestyle=':', alpha=0.5)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '6_Inter_Target_Engineering_for_Dead.png'))
    plt.close()
    
    # === PRINT RECOMMENDATIONS ===
    print("\n" + "="*60)
    print("💡 FEATURE ENGINEERING RECOMMENDATIONS FOR DRY_DEAD:")
    print("="*60)
    for i, (feat, corr) in enumerate(sorted_features[:5], 1):
        print(f"  {i}. {feat:30s} | R = {corr:+.3f}")
    print("="*60)


if __name__ == '__main__':
    # Load
    df, targets = process_data('train.csv')
    
    # Execute Modules
    viz_physics(df)
    viz_distributions(df, targets)
    viz_correlations(df, targets)
    viz_nonlinear(df)
    viz_interaction_correlations(df, targets)
    
    # NEW: Inter-target engineering for Dry_Dead
    viz_inter_target_engineering(df, targets)
    
    print("="*60)
    print(f"Done. Visualizations saved to: {BASE_VIZ_PATH}")
    print("="*60)