import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings

# --- Configuration ---
warnings.filterwarnings('ignore')
BASE_VIZ_PATH = 'visualizations_detailed_v2'
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
    pivot_cols = ['clean_id', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    valid_cols = [c for c in pivot_cols if c in df.columns]
    
    # Pivot target values
    targets = df.pivot_table(index='clean_id', columns='target_name', values='target', aggfunc='max').reset_index()
    
    # Get Metadata
    meta = df[valid_cols].drop_duplicates(subset=['clean_id'])
    
    # Merge
    wide = pd.merge(meta, targets, on='clean_id', how='left')
    
    # 3. Fill Missing Targets with 0
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    for t in target_cols:
        if t not in wide.columns: wide[t] = 0.0
    wide[target_cols] = wide[target_cols].fillna(0.0)
    
    # 4. Feature Engineering - EXPANDED
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'])
    wide['Month'] = wide['Sampling_Date'].dt.month
    wide['Season'] = wide['Month'].apply(get_season)
    
    # === NEW: Advanced Feature Transforms ===
    epsilon = 1e-3
    
    # NDVI transforms
    wide['NDVI_Squared'] = wide['Pre_GSHH_NDVI'] ** 2
    wide['NDVI_Sqrt'] = np.sqrt(wide['Pre_GSHH_NDVI'].clip(lower=0))
    
    # Height transforms
    wide['Height_Log'] = np.log1p(wide['Height_Ave_cm'])
    
    # Original interactions (keep these)
    wide['Interaction_Mul'] = wide['Height_Ave_cm'] * wide['Pre_GSHH_NDVI']
    wide['Interaction_Add'] = wide['Height_Ave_cm'] + wide['Pre_GSHH_NDVI']
    wide['Interaction_Ratio_NDVI_H'] = wide['Pre_GSHH_NDVI'] / (wide['Height_Ave_cm'] + epsilon)
    wide['Interaction_Ratio_H_NDVI'] = wide['Height_Ave_cm'] / (wide['Pre_GSHH_NDVI'] + epsilon)
    
    # === NEW: Target Inter-Engineering (Create as FEATURES) ===
    # These will be used as additional features/targets to analyze
    wide['Total_minus_Green'] = wide['Dry_Total_g'] - wide['Dry_Green_g']
    wide['Total_minus_Clover'] = wide['Dry_Total_g'] - wide['Dry_Clover_g']
    wide['Green_plus_Clover'] = wide['Dry_Green_g'] + wide['Dry_Clover_g']
    wide['Unexplained_Mass'] = wide['Dry_Total_g'] - wide['Green_plus_Clover']
    
    wide['Ratio_Total_Green'] = wide['Dry_Total_g'] / (wide['Dry_Green_g'] + epsilon)
    wide['Ratio_Total_Clover'] = wide['Dry_Total_g'] / (wide['Dry_Clover_g'] + epsilon)
    wide['Ratio_Green_Clover'] = wide['Dry_Green_g'] / (wide['Dry_Clover_g'] + epsilon)
    wide['Ratio_Clover_Total'] = wide['Dry_Clover_g'] / (wide['Dry_Total_g'] + epsilon)
    wide['Ratio_Green_Total'] = wide['Dry_Green_g'] / (wide['Dry_Total_g'] + epsilon)
    
    wide['Product_Green_Clover'] = wide['Dry_Green_g'] * wide['Dry_Clover_g']
    wide['Product_Total_Green'] = wide['Dry_Total_g'] * wide['Dry_Green_g']
    
    wide['Sqrt_Total'] = np.sqrt(wide['Dry_Total_g'])
    wide['Sqrt_Green'] = np.sqrt(wide['Dry_Green_g'])
    wide['Square_Green'] = wide['Dry_Green_g'] ** 2
    
    wide['GDM_minus_Total'] = wide['GDM_g'] - wide['Dry_Total_g']
    wide['Ratio_GDM_Total'] = wide['GDM_g'] / (wide['Dry_Total_g'] + epsilon)
    
    # === NEW: Dry_Dead_g PROXY FEATURES ===
    # Composition-based (Dead matter proxies)
    wide['Total_minus_Living'] = wide['Dry_Total_g'] - (wide['Dry_Green_g'] + wide['Dry_Clover_g'])
    wide['Dead_approx_Total_minus_Green_Clover'] = wide['Dry_Total_g'] - wide['Green_plus_Clover']
    wide['NonGreen_Mass'] = wide['Dry_Total_g'] - wide['Dry_Green_g']
    
    # Ratio features (Dead relative to other components)
    wide['Ratio_Dead_Total'] = wide['Dry_Dead_g'] / (wide['Dry_Total_g'] + epsilon)
    wide['Ratio_Dead_Green'] = wide['Dry_Dead_g'] / (wide['Dry_Green_g'] + epsilon)
    wide['Ratio_Dead_Clover'] = wide['Dry_Dead_g'] / (wide['Dry_Clover_g'] + epsilon)
    wide['Ratio_Total_Dead'] = wide['Dry_Total_g'] / (wide['Dry_Dead_g'] + epsilon)
    wide['Ratio_Green_Dead'] = wide['Dry_Green_g'] / (wide['Dry_Dead_g'] + epsilon)
    
    # Interaction products
    wide['Product_Dead_Green'] = wide['Dry_Dead_g'] * wide['Dry_Green_g']
    wide['Product_Dead_Clover'] = wide['Dry_Dead_g'] * wide['Dry_Clover_g']
    wide['Product_Dead_Total'] = wide['Dry_Dead_g'] * wide['Dry_Total_g']
    
    # Transforms of Dead
    wide['Sqrt_Dead'] = np.sqrt(wide['Dry_Dead_g'].clip(lower=0))
    wide['Square_Dead'] = wide['Dry_Dead_g'] ** 2
    wide['Log_Dead'] = np.log1p(wide['Dry_Dead_g'].clip(lower=0))
    
    # Dead with NDVI/Height interactions
    wide['Dead_times_NDVI'] = wide['Dry_Dead_g'] * wide['Pre_GSHH_NDVI']
    wide['Dead_times_Height'] = wide['Dry_Dead_g'] * wide['Height_Ave_cm']
    wide['Dead_div_NDVI'] = wide['Dry_Dead_g'] / (wide['Pre_GSHH_NDVI'] + epsilon)
    wide['Dead_div_Height'] = wide['Dry_Dead_g'] / (wide['Height_Ave_cm'] + epsilon)
    
    # Complex proxies
    wide['Living_to_Dead_Ratio'] = wide['Green_plus_Clover'] / (wide['Dry_Dead_g'] + epsilon)
    wide['Dead_fraction_of_NonGreen'] = wide['Dry_Dead_g'] / (wide['NonGreen_Mass'] + epsilon)
    
    # Log Transforms for ALL targets (original + engineered)
    all_target_cols = target_cols + [
        'Total_minus_Green', 'Total_minus_Clover', 'Green_plus_Clover', 'Unexplained_Mass',
        'Total_minus_Living', 'Dead_approx_Total_minus_Green_Clover', 'NonGreen_Mass'
    ]
    
    for t in all_target_cols:
        if t in wide.columns:
            wide[f'Log_{t}'] = np.log1p(wide[t].clip(lower=0))
    
    # 5. Create Train/Validation Split
    train_idx, val_idx = train_test_split(wide.index, test_size=0.2, random_state=42, stratify=wide['State'])
    wide['Set'] = 'Train'
    wide.loc[val_idx, 'Set'] = 'Validation'
    
    print(f"   Data Shape: {wide.shape}")
    print(f"   Total Columns: {len(wide.columns)}")
    return wide, target_cols

# ==============================================================================
# 1. PHYSICS CONSISTENCY & COMPOSITION
# ==============================================================================
def viz_physics(df):
    print("2. Generating Physics Consistency Checks...")
    
    df['Sum_Components'] = df['Dry_Clover_g'] + df['Dry_Dead_g'] + df['Dry_Green_g']
    df['Physics_Error'] = df['Dry_Total_g'] - df['Sum_Components']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(data=df, x='Sum_Components', y='Dry_Total_g', hue='Physics_Error', 
                    palette='coolwarm', alpha=0.6, ax=axes[0])
    max_val = max(df['Sum_Components'].max(), df['Dry_Total_g'].max())
    axes[0].plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect Physics (y=x)')
    axes[0].set_title("Constraint: Total = Clover + Dead + Green")
    axes[0].legend()
    
    season_comp = df.groupby('Season')[['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']].mean()
    season_comp_pct = season_comp.div(season_comp.sum(axis=1), axis=0) * 100
    season_comp_pct = season_comp_pct.reindex(['Summer', 'Autumn', 'Winter', 'Spring'])
    
    season_comp_pct.plot(kind='bar', stacked=True, color=['#d62728', '#7f7f7f', '#2ca02c'], ax=axes[1])
    axes[1].set_title("Biomass Composition Ratio by Season")
    axes[1].set_ylabel("Percentage of Total Mass (%)")
    axes[1].legend(title='Component', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '1_Physics_and_Composition.png'))
    plt.close()

# ==============================================================================
# 2. GROUPED DISTRIBUTIONS
# ==============================================================================
def viz_distributions(df, target_cols):
    print("3. Generating Grouped Target Distributions...")
    
    groups = ['State', 'Season', 'Species']
    
    for group in groups:
        plt.figure(figsize=(14, 6))
        
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
# 3. ENHANCED CORRELATION MATRIX - With New Features
# ==============================================================================
def viz_correlations(df, target_cols):
    print("4. Generating Enhanced Correlation Matrix...")
    
    # Include new transforms
    feature_cols = [
        'Height_Ave_cm', 'Height_Log',
        'Pre_GSHH_NDVI', 'NDVI_Squared', 'NDVI_Sqrt'
    ]
    
    cols = feature_cols + target_cols
    corr = df[cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(12, 10))
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
    
    plt.title("Enhanced Feature & Target Correlations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '3_Correlation_Matrix_Enhanced.png'))
    plt.close()

# ==============================================================================
# 4. EXPANDED NON-LINEAR INTERACTIONS
# ==============================================================================
def viz_nonlinear(df):
    print("5. Generating Expanded Non-Linear Interaction Plots...")
    
    plot_df = df.sample(n=min(2000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    # Row 1: Base features
    sns.regplot(data=plot_df, x='Height_Ave_cm', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'}, ax=axes[0])
    axes[0].set_title("Base: Height vs Biomass")
    
    sns.regplot(data=plot_df, x='Pre_GSHH_NDVI', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'}, ax=axes[1])
    axes[1].set_title("Base: NDVI vs Biomass")
    
    r_mul, _ = pearsonr(plot_df['Interaction_Mul'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Mul', y='Log_Dry_Total_g', 
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'green'}, ax=axes[2])
    axes[2].set_title(f"Multiplication (H * NDVI)\nR = {r_mul:.2f}")
    
    # Row 2: New transforms
    r_log_h, _ = pearsonr(plot_df['Height_Log'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Height_Log', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'purple'}, ax=axes[3])
    axes[3].set_title(f"Log(Height)\nR = {r_log_h:.2f}")
    
    r_ndvi_sq, _ = pearsonr(plot_df['NDVI_Squared'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='NDVI_Squared', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'orange'}, ax=axes[4])
    axes[4].set_title(f"NDVI²\nR = {r_ndvi_sq:.2f}")
    
    r_ndvi_sqrt, _ = pearsonr(plot_df['NDVI_Sqrt'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='NDVI_Sqrt', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'brown'}, ax=axes[5])
    axes[5].set_title(f"√NDVI\nR = {r_ndvi_sqrt:.2f}")
    
    # Row 3: Ratios
    r_add, _ = pearsonr(plot_df['Interaction_Add'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Add', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'cyan'}, ax=axes[6])
    axes[6].set_title(f"Addition (H + NDVI)\nR = {r_add:.2f}")
    
    r_rat1, _ = pearsonr(plot_df['Interaction_Ratio_NDVI_H'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Ratio_NDVI_H', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'magenta'}, ax=axes[7])
    axes[7].set_title(f"Ratio (NDVI / H)\nR = {r_rat1:.2f}")
    axes[7].set_xlim(0, plot_df['Interaction_Ratio_NDVI_H'].quantile(0.95))
    
    r_rat2, _ = pearsonr(plot_df['Interaction_Ratio_H_NDVI'], plot_df['Log_Dry_Total_g'])
    sns.regplot(data=plot_df, x='Interaction_Ratio_H_NDVI', y='Log_Dry_Total_g', 
                lowess=True, scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'teal'}, ax=axes[8])
    axes[8].set_title(f"Ratio (H / NDVI)\nR = {r_rat2:.2f}")
    axes[8].set_xlim(0, plot_df['Interaction_Ratio_H_NDVI'].quantile(0.95))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '4_NonLinear_Interactions_Expanded.png'))
    plt.close()

# ==============================================================================
# 5. COMPREHENSIVE INTERACTION CORRELATION - All Features vs All Targets
# ==============================================================================
def viz_interaction_correlations(df, target_cols):
    print("6. Generating Comprehensive Feature-Target Correlation Matrix...")
    
    # All features including new transforms
    features = [
        'Height_Ave_cm', 'Height_Log',
        'Pre_GSHH_NDVI', 'NDVI_Squared', 'NDVI_Sqrt',
        'Interaction_Mul', 'Interaction_Add',
        'Interaction_Ratio_NDVI_H', 'Interaction_Ratio_H_NDVI'
    ]
    
    # Include engineered targets alongside original
    extended_targets = target_cols + [
        'Total_minus_Green', 'Total_minus_Clover', 
        'Green_plus_Clover', 'Unexplained_Mass',
        'Total_minus_Living', 'Dead_approx_Total_minus_Green_Clover', 'NonGreen_Mass'
    ]
    
    log_targets = [f'Log_{t}' for t in extended_targets if f'Log_{t}' in df.columns]
    
    full_corr = df[features + log_targets].corr()
    target_corr = full_corr.loc[features, log_targets]
    
    # Clean column names
    target_corr.columns = [c.replace('Log_', '').replace('_g', '') for c in target_corr.columns]
    
    # Clean row names
    feature_map = {
        'Height_Ave_cm': 'Height',
        'Height_Log': 'Log(Height)',
        'Pre_GSHH_NDVI': 'NDVI',
        'NDVI_Squared': 'NDVI²',
        'NDVI_Sqrt': '√NDVI',
        'Interaction_Mul': 'H × NDVI',
        'Interaction_Add': 'H + NDVI',
        'Interaction_Ratio_NDVI_H': 'NDVI / H',
        'Interaction_Ratio_H_NDVI': 'H / NDVI'
    }
    target_corr = target_corr.rename(index=feature_map)
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        target_corr,
        annot=True,
        fmt=".2f",
        cmap='RdBu_r',
        center=0,
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Pearson Correlation (r)"},
        square=False
    )
    
    plt.title("Comprehensive Feature-Target Correlation Matrix\n(All Features vs All Targets in Log Space)", fontsize=16)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(BASE_VIZ_PATH, '5_Comprehensive_Correlation.png'))
    plt.close()

# ==============================================================================
# 6. DEEP DIVE: Dry_Dead Analysis with All Features
# ==============================================================================
def viz_dry_dead_deep_dive(df):
    print("7. Deep Dive Analysis: Predicting Dry_Dead with All Features...")
    
    # Target of interest
    focus_target = 'Dry_Dead_g'
    
    # All predictive features (base + transforms + engineered targets + Dead proxies)
    all_features = [
        'Height_Ave_cm', 'Height_Log',
        'Pre_GSHH_NDVI', 'NDVI_Squared', 'NDVI_Sqrt',
        'Interaction_Mul', 'Interaction_Add',
        'Interaction_Ratio_NDVI_H', 'Interaction_Ratio_H_NDVI',
        'Total_minus_Green', 'Total_minus_Clover', 
        'Green_plus_Clover', 'Unexplained_Mass',
        'Ratio_Total_Green', 'Ratio_Total_Clover', 
        'Ratio_Green_Clover', 'Ratio_Clover_Total', 'Ratio_Green_Total',
        'Product_Green_Clover', 'Product_Total_Green',
        'Sqrt_Total', 'Sqrt_Green', 'Square_Green',
        'GDM_minus_Total', 'Ratio_GDM_Total',
        # NEW: Dead-specific proxy features
        'Total_minus_Living', 'Dead_approx_Total_minus_Green_Clover', 'NonGreen_Mass',
        'Ratio_Dead_Total', 'Ratio_Dead_Green', 'Ratio_Dead_Clover',
        'Ratio_Total_Dead', 'Ratio_Green_Dead',
        'Product_Dead_Green', 'Product_Dead_Clover', 'Product_Dead_Total',
        'Sqrt_Dead', 'Square_Dead', 'Log_Dead',
        'Dead_times_NDVI', 'Dead_times_Height', 'Dead_div_NDVI', 'Dead_div_Height',
        'Living_to_Dead_Ratio', 'Dead_fraction_of_NonGreen'
    ]
    
    # Calculate correlations
    correlations = {}
    for feat in all_features:
        if feat in df.columns:
            valid_mask = np.isfinite(df[feat]) & np.isfinite(df[focus_target])
            if valid_mask.sum() > 10:
                r, _ = pearsonr(df.loc[valid_mask, feat], df.loc[valid_mask, focus_target])
                correlations[feat] = r
    
    sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # === VISUALIZATION ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot 1: Correlation ranking
    top_n = min(20, len(sorted_features))
    feature_names = [f[0] for f in sorted_features[:top_n]]
    feature_corrs = [f[1] for f in sorted_features[:top_n]]
    colors = ['#d62728' if c > 0 else '#1f77b4' for c in feature_corrs]
    
    axes[0].barh(feature_names, feature_corrs, color=colors, alpha=0.7)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_title(f'Top {top_n} Features for Predicting Dry_Dead', fontsize=12)
    axes[0].invert_yaxis()
    
    # Plot 2-4: Top 3 features scatter plots
    for idx in range(3):
        if idx < len(sorted_features):
            feat_name = sorted_features[idx][0]
            feat_corr = sorted_features[idx][1]
            
            plot_df = df.sample(n=min(1500, len(df)), random_state=42)
            valid_mask = np.isfinite(plot_df[feat_name]) & np.isfinite(plot_df[focus_target])
            plot_df_valid = plot_df[valid_mask]
            
            sns.scatterplot(data=plot_df_valid, x=feat_name, y=focus_target,
                          alpha=0.4, s=15, color='darkred', ax=axes[idx+1])
            sns.regplot(data=plot_df_valid, x=feat_name, y=focus_target,
                       scatter=False, lowess=True, 
                       line_kws={'color': 'black', 'linewidth': 2}, ax=axes[idx+1])
            axes[idx+1].set_title(f'#{idx+1}: {feat_name}\nR = {feat_corr:.3f}', fontsize=11)
            axes[idx+1].set_ylabel('Dry_Dead_g')
            
            if 'Ratio' in feat_name:
                axes[idx+1].set_xlim(0, plot_df_valid[feat_name].quantile(0.95))
    
    # Plot 5: Feature category breakdown
    categories = {
        'Base Features': ['Height_Ave_cm', 'Pre_GSHH_NDVI'],
        'Transforms': ['Height_Log', 'NDVI_Squared', 'NDVI_Sqrt'],
        'Interactions': ['Interaction_Mul', 'Interaction_Add', 
                        'Interaction_Ratio_NDVI_H', 'Interaction_Ratio_H_NDVI'],
        'Dead Proxies': ['Total_minus_Living', 'Dead_approx_Total_minus_Green_Clover', 'NonGreen_Mass',
                        'Ratio_Dead_Total', 'Ratio_Dead_Green', 'Ratio_Dead_Clover',
                        'Ratio_Total_Dead', 'Ratio_Green_Dead',
                        'Product_Dead_Green', 'Product_Dead_Clover', 'Product_Dead_Total',
                        'Sqrt_Dead', 'Square_Dead', 'Log_Dead',
                        'Dead_times_NDVI', 'Dead_times_Height', 'Dead_div_NDVI', 'Dead_div_Height',
                        'Living_to_Dead_Ratio', 'Dead_fraction_of_NonGreen'],
        'Other Engineering': [f for f in all_features if f not in 
                              ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'Height_Log', 
                               'NDVI_Squared', 'NDVI_Sqrt', 'Interaction_Mul', 
                               'Interaction_Add', 'Interaction_Ratio_NDVI_H', 
                               'Interaction_Ratio_H_NDVI',
                               'Total_minus_Living', 'Dead_approx_Total_minus_Green_Clover', 'NonGreen_Mass',
                               'Ratio_Dead_Total', 'Ratio_Dead_Green', 'Ratio_Dead_Clover',
                               'Ratio_Total_Dead', 'Ratio_Green_Dead',
                               'Product_Dead_Green', 'Product_Dead_Clover', 'Product_Dead_Total',
                               'Sqrt_Dead', 'Square_Dead', 'Log_Dead',
                               'Dead_times_NDVI', 'Dead_times_Height', 'Dead_div_NDVI', 'Dead_div_Height',
                               'Living_to_Dead_Ratio', 'Dead_fraction_of_NonGreen']]
    }
    
    cat_scores = {}
    for cat, feats in categories.items():
        scores = [abs(correlations.get(f, 0)) for f in feats if f in correlations]
        cat_scores[cat] = np.mean(scores) if scores else 0
    
    axes[4].bar(cat_scores.keys(), cat_scores.values(), 
               color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
    axes[4].set_ylabel('Mean |Correlation|')
    axes[4].set_title('Feature Category Performance', fontsize=12)
    axes[4].tick_params(axis='x', rotation=45)
    
    # Plot 6: Best feature residual analysis
    if sorted_features:
        best_feat = sorted_features[0][0]
        valid_mask = np.isfinite(df[best_feat]) & np.isfinite(df[focus_target])
        X_fit = df.loc[valid_mask, best_feat].values.reshape(-1, 1)
        y_fit = df.loc[valid_mask, focus_target].values
        
        model = LinearRegression()
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_fit)
        residuals = y_fit - y_pred
        
        axes[5].scatter(y_pred, residuals, alpha=0.3, s=10, color='purple')
        axes[5].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[5].set_xlabel(f'Predicted Dry_Dead')
        axes[5].set_ylabel('Residual')
        axes[5].set_title(f'Residual Analysis\n(Best Feature: {best_feat})', fontsize=11)
        
        std_resid = np.std(residuals)
        axes[5].axhline(std_resid, color='red', linestyle=':', alpha=0.5, label=f'±1σ ({std_resid:.2f})')
        axes[5].axhline(-std_resid, color='red', linestyle=':', alpha=0.5)
        axes[5].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '6_Dry_Dead_Deep_Dive.png'))
    plt.close()
    
    # Print recommendations
    print("\n" + "="*70)
    print("💡 TOP FEATURES FOR PREDICTING DRY_DEAD:")
    print("="*70)
    for i, (feat, corr) in enumerate(sorted_features[:10], 1):
        print(f"  {i:2d}. {feat:35s} | R = {corr:+.3f}")
    print("="*70)

# ==============================================================================
# 7. Multi-Target Heatmap: Which Features Predict Which Engineered Targets
# ==============================================================================
def viz_comprehensive_target_heatmap(df):
    print("8. Creating Comprehensive Target Prediction Heatmap...")
    
    all_features = [
        'Height_Ave_cm', 'Height_Log',
        'Pre_GSHH_NDVI', 'NDVI_Squared', 'NDVI_Sqrt',
        'Interaction_Mul', 'Interaction_Add',
        'Interaction_Ratio_NDVI_H', 'Interaction_Ratio_H_NDVI'
    ]
    
    # Original + engineered targets (including Dead proxies)
    all_targets = [
        'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g',
        'Total_minus_Green', 'Total_minus_Clover', 
        'Green_plus_Clover', 'Unexplained_Mass',
        'Total_minus_Living', 'Dead_approx_Total_minus_Green_Clover', 'NonGreen_Mass'
    ]
    
    # Build correlation matrix
    heatmap_data = []
    for feat in all_features:
        row = []
        for target in all_targets:
            if feat in df.columns and target in df.columns:
                valid_mask = np.isfinite(df[feat]) & np.isfinite(df[target])
                if valid_mask.sum() > 10:
                    r, _ = pearsonr(df.loc[valid_mask, feat], df.loc[valid_mask, target])
                    row.append(r)
                else:
                    row.append(0.0)
            else:
                row.append(0.0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=all_features,
        columns=[t.replace('_g', '') for t in all_targets]
    )
    
    # Clean feature names
    feature_labels = {
        'Height_Ave_cm': 'Height',
        'Height_Log': 'Log(H)',
        'Pre_GSHH_NDVI': 'NDVI',
        'NDVI_Squared': 'NDVI²',
        'NDVI_Sqrt': '√NDVI',
        'Interaction_Mul': 'H×NDVI',
        'Interaction_Add': 'H+NDVI',
        'Interaction_Ratio_NDVI_H': 'NDVI/H',
        'Interaction_Ratio_H_NDVI': 'H/NDVI'
    }
    
    heatmap_df = heatmap_df.rename(index=feature_labels)
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Correlation (r)"},
        vmin=-0.8,
        vmax=0.8
    )
    
    plt.title('Feature-Target Correlation Matrix\n(All Features vs Original + Engineered Targets)', fontsize=16)
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(BASE_VIZ_PATH, '7_Comprehensive_Target_Heatmap.png'))
    plt.close()

# ==============================================================================
# 8. TARGET VARIABILITY ANALYSIS - Understanding Spread & Uncertainty
# ==============================================================================
def viz_target_variability(df, target_cols):
    """
    Analyzes the variance, standard deviation, and coefficient of variation
    for each target variable across different groupings.
    
    This helps identify:
    - Which targets are most/least variable
    - How variability changes across seasons/species/states
    - Heteroscedasticity patterns (variance changing with mean)
    """
    print("8. Generating Target Variability Analysis...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # --- PLOT 1: Overall Variability Comparison ---
    ax1 = fig.add_subplot(gs[0, :])
    
    variability_stats = []
    for target in target_cols:
        stats = {
            'Target': target.replace('_g', ''),
            'Mean': df[target].mean(),
            'Std': df[target].std(),
            'CV (%)': (df[target].std() / df[target].mean() * 100) if df[target].mean() > 0 else 0,
            'IQR': df[target].quantile(0.75) - df[target].quantile(0.25),
            'Range': df[target].max() - df[target].min()
        }
        variability_stats.append(stats)
    
    var_df = pd.DataFrame(variability_stats)
    
    x = np.arange(len(var_df))
    width = 0.35
    
    ax1_twin = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, var_df['Std'], width, 
                    label='Std Dev (g)', color='steelblue', alpha=0.8)
    bars2 = ax1_twin.bar(x + width/2, var_df['CV (%)'], width, 
                         label='CV (%)', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Target Variable', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Standard Deviation (g)', fontsize=11, color='steelblue')
    ax1_twin.set_ylabel('Coefficient of Variation (%)', fontsize=11, color='coral')
    ax1.set_title('Target Variability Comparison: Std Dev vs Coefficient of Variation', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(var_df['Target'], rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1_twin.tick_params(axis='y', labelcolor='coral')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.1f}', ha='center', va='bottom', fontsize=9, color='steelblue')
        ax1_twin.text(bar2.get_x() + bar2.get_width()/2., height2,
                     f'{height2:.0f}%', ha='center', va='bottom', fontsize=9, color='coral')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # --- PLOT 2: Variability by Season ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    season_std = df.groupby('Season')[target_cols].std().reindex(season_order)
    season_std.columns = [c.replace('_g', '') for c in season_std.columns]
    
    season_std.plot(kind='bar', ax=ax2, width=0.8, colormap='Set2')
    ax2.set_title('Std Dev by Season', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Season', fontsize=10)
    ax2.set_ylabel('Standard Deviation (g)', fontsize=10)
    ax2.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # --- PLOT 3: Variability by Species (Top 5) ---
    ax3 = fig.add_subplot(gs[1, 1])
    
    top_species = df['Species'].value_counts().head(5).index
    species_std = df[df['Species'].isin(top_species)].groupby('Species')[target_cols].std()
    species_std.columns = [c.replace('_g', '') for c in species_std.columns]
    
    species_std.plot(kind='bar', ax=ax3, width=0.8, colormap='viridis')
    ax3.set_title('Std Dev by Species (Top 5)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Species', fontsize=10)
    ax3.set_ylabel('Standard Deviation (g)', fontsize=10)
    ax3.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # --- PLOT 4: Variability by State ---
    ax4 = fig.add_subplot(gs[1, 2])
    
    state_std = df.groupby('State')[target_cols].std()
    state_std.columns = [c.replace('_g', '') for c in state_std.columns]
    
    state_std.plot(kind='bar', ax=ax4, width=0.8, colormap='Paired')
    ax4.set_title('Std Dev by State', fontsize=12, fontweight='bold')
    ax4.set_xlabel('State', fontsize=10)
    ax4.set_ylabel('Standard Deviation (g)', fontsize=10)
    ax4.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # --- PLOT 5: Heteroscedasticity Check (Mean vs Std) ---
    ax5 = fig.add_subplot(gs[2, 0])
    
    for target in target_cols:
        # Group by bins of the target value
        df_sorted = df.sort_values(target)
        n_bins = 10
        bin_size = len(df_sorted) // n_bins
        
        means = []
        stds = []
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(df_sorted)
            bin_data = df_sorted.iloc[start:end][target]
            means.append(bin_data.mean())
            stds.append(bin_data.std())
        
        ax5.plot(means, stds, 'o-', label=target.replace('_g', ''), alpha=0.7, markersize=6)
    
    ax5.set_xlabel('Mean Value (g)', fontsize=10)
    ax5.set_ylabel('Standard Deviation (g)', fontsize=10)
    ax5.set_title('Heteroscedasticity Check\n(Mean vs Std in Decile Bins)', 
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # --- PLOT 6: Distribution of Standardized Residuals ---
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Standardize each target (z-score)
    standardized = {}
    for target in target_cols:
        z = (df[target] - df[target].mean()) / df[target].std()
        standardized[target.replace('_g', '')] = z
    
    std_df = pd.DataFrame(standardized)
    
    std_df.plot(kind='box', ax=ax6, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax6.set_ylabel('Standardized Value (z-score)', fontsize=10)
    ax6.set_title('Standardized Target Distribution\n(Box Plot Comparison)', 
                  fontsize=12, fontweight='bold')
    ax6.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    
    # --- PLOT 7: Variance Ratio Across Groups ---
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Calculate variance ratio (between-group / within-group) for each target
    variance_ratios = []
    
    for target in target_cols:
        # Season-based variance decomposition
        overall_var = df[target].var()
        
        # Between-group variance
        season_means = df.groupby('Season')[target].mean()
        season_counts = df.groupby('Season').size()
        grand_mean = df[target].mean()
        between_var = ((season_means - grand_mean) ** 2 * season_counts).sum() / len(df)
        
        # Within-group variance
        within_var = overall_var - between_var
        
        variance_ratios.append({
            'Target': target.replace('_g', ''),
            'Between-Group Var': between_var,
            'Within-Group Var': within_var,
            'Ratio': between_var / within_var if within_var > 0 else 0
        })
    
    vr_df = pd.DataFrame(variance_ratios)
    
    x = np.arange(len(vr_df))
    width = 0.35
    
    ax7.bar(x - width/2, vr_df['Between-Group Var'], width, 
            label='Between-Season', color='darkgreen', alpha=0.7)
    ax7.bar(x + width/2, vr_df['Within-Group Var'], width, 
            label='Within-Season', color='orange', alpha=0.7)
    
    ax7.set_xlabel('Target Variable', fontsize=10)
    ax7.set_ylabel('Variance Component', fontsize=10)
    ax7.set_title('Variance Decomposition by Season\n(Between vs Within Group)', 
                  fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(vr_df['Target'], rotation=45, ha='right')
    ax7.legend(fontsize=9)
    ax7.grid(axis='y', alpha=0.3)
    
    # Add ratio annotations
    for i, ratio in enumerate(vr_df['Ratio']):
        ax7.text(i, max(vr_df['Between-Group Var'].max(), vr_df['Within-Group Var'].max()) * 1.05,
                f'F={ratio:.2f}', ha='center', fontsize=8, fontweight='bold')
    
    plt.savefig(os.path.join(BASE_VIZ_PATH, '8_Target_Variability_Analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PRINT STATISTICAL SUMMARY ---
    print("\n" + "="*70)
    print("📊 TARGET VARIABILITY SUMMARY:")
    print("="*70)
    print(f"{'Target':<15} {'Mean':>8} {'Std':>8} {'CV(%)':>8} {'Min':>8} {'Max':>8}")
    print("-"*70)
    for _, row in var_df.iterrows():
        target_name = row['Target']
        mean_val = row['Mean']
        std_val = row['Std']
        cv_val = row['CV (%)']
        
        # Get min/max from original data
        target_full = [t for t in target_cols if t.replace('_g', '') == target_name][0]
        min_val = df[target_full].min()
        max_val = df[target_full].max()
        
        print(f"{target_name:<15} {mean_val:>8.1f} {std_val:>8.1f} {cv_val:>8.1f} {min_val:>8.1f} {max_val:>8.1f}")
    print("="*70)
    
    print("\n💡 KEY INSIGHTS:")
    print("-"*70)
    
    # Identify most/least variable
    most_variable = var_df.loc[var_df['CV (%)'].idxmax()]
    least_variable = var_df.loc[var_df['CV (%)'].idxmin()]
    
    print(f"  • Most Variable: {most_variable['Target']} (CV = {most_variable['CV (%)']:.1f}%)")
    print(f"  • Least Variable: {least_variable['Target']} (CV = {least_variable['CV (%)']:.1f}%)")
    
    # Identify targets with high between-group variance (good for stratification)
    print("\n  • Variance Decomposition (Between-Season / Within-Season):")
    for _, row in vr_df.iterrows():
        if row['Ratio'] > 0.5:
            print(f"    - {row['Target']}: F-ratio = {row['Ratio']:.2f} (High between-group variance → good for stratification)")
        elif row['Ratio'] < 0.1:
            print(f"    - {row['Target']}: F-ratio = {row['Ratio']:.2f} (Low between-group variance → less predictable by season)")
    
    print("="*70 + "\n")
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ENHANCED EDA WITH FEATURE TRANSFORMS & TARGET ENGINEERING")
    print("="*70 + "\n")
    
    # Load and process
    df, targets = process_data('train.csv')
    
    # Execute all visualization modules
    viz_physics(df)
    viz_distributions(df, targets)
    viz_correlations(df, targets)
    viz_nonlinear(df)
    viz_interaction_correlations(df, targets)
    viz_dry_dead_deep_dive(df)
    viz_comprehensive_target_heatmap(df)
    viz_target_variability(df, targets)

    
    print("\n" + "="*70)
    print(f"✅ Complete! All visualizations saved to: {BASE_VIZ_PATH}/")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. 1_Physics_and_Composition.png")
    print("  2. 2_Dist_by_[State/Season/Species].png")
    print("  3. 3_Correlation_Matrix_Enhanced.png")
    print("  4. 4_NonLinear_Interactions_Expanded.png")
    print("  5. 5_Comprehensive_Correlation.png")
    print("  6. 6_Dry_Dead_Deep_Dive.png")
    print("  7. 7_Comprehensive_Target_Heatmap.png")
    print("  8. 8_Target_Variability_Analysis.png")
    print("="*70 + "\n")