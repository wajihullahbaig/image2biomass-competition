#!/usr/bin/env python3
"""
Ratio Correlation Analysis
Analyzes correlations between biomass targets and derived ratio features,
specifically focusing on Dead Biomass relationships with Height and Total Biomass.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_CSV = Path('./wide.csv')
OUTPUT_DIR = Path('./analysis_results/ratio_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and prepare data."""
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Please ensure data is prepared.")
        return None
        
    df = pd.read_csv(INPUT_CSV)
    
    # Ensure numeric columns
    cols_to_numeric = ['Height_Ave_cm', 'Pre_GSHH_NDVI', 
                       'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def compute_derived_features(df):
    """Compute derived features for analysis."""
    print("Computing derived features...")
    
    # 1. Height Transformations
    # Ensure no zero/negative heights for division/log
    df['Height_Clean'] = df['Height_Ave_cm'].clip(lower=0.1)
    df['Height_Log'] = np.log1p(df['Height_Clean'])
    
    # 2. Ratio Features (Focus on Dead Biomass)
    # Dead / Height
    df['Dead_per_cm_Height'] = df['Dry_Dead_g'] / df['Height_Clean']
    
    # Dead / Total (Handle zero division)
    df['Dead_to_Total_Ratio'] = df['Dry_Dead_g'] / (df['Dry_Total_g'] + 1e-6)
    
    # Dead / GDM
    df['Dead_to_GDM_Ratio'] = df['Dry_Dead_g'] / (df['GDM_g'] + 1e-6)
    
    return df

def plot_target_analysis(df, target_base, predictors):
    """
    Analyzes which version of the target (Raw, Density, Ratio) 
    correlates best with the predictors.
    """
    print(f"Analyzing target: {target_base}...")
    
    # 1. Create Derived Versions of this Target
    # Raw
    t_raw = df[target_base]
    
    # Density (Target / Height)
    t_density = df[target_base] / df['Height_Clean']
    
    # Ratio to Total (Target / Total)
    # Avoid self-division for Total
    if target_base == 'Dry_Total_g':
        t_ratio_total = pd.Series(np.ones(len(df)), index=df.index) # Trivial
    else:
        t_ratio_total = df[target_base] / (df['Dry_Total_g'] + 1e-6)
        
    # Ratio to GDM (Target / GDM)
    if target_base == 'GDM_g':
        t_ratio_gdm = pd.Series(np.ones(len(df)), index=df.index)
    else:
        t_ratio_gdm = df[target_base] / (df['GDM_g'] + 1e-6)

    # 2. Build DataFrame for Correlation
    data = pd.DataFrame({
        f'{target_base} (Raw)': t_raw,
        f'{target_base} / Height (Density)': t_density,
        f'{target_base} / Total (Ratio)': t_ratio_total,
        f'{target_base} / GDM (Ratio)': t_ratio_gdm
    })
    
    # Remove trivial columns (e.g. Total/Total)
    if target_base == 'Dry_Total_g':
        data.drop(columns=[f'{target_base} / Total (Ratio)'], inplace=True)
    if target_base == 'GDM_g':
        data.drop(columns=[f'{target_base} / GDM (Ratio)'], inplace=True)
        
    # 3. Calculate Correlations with Predictors
    # Rows: Derived Targets
    # Cols: Predictors
    corr_data = []
    
    for derived_name in data.columns:
        row_corrs = []
        for pred in predictors:
            # Drop NaNs
            valid = data[derived_name].notna() & df[pred].notna()
            if valid.sum() > 10:
                r, _ = stats.pearsonr(data.loc[valid, derived_name], df.loc[valid, pred])
                row_corrs.append(r)
            else:
                row_corrs.append(np.nan)
        corr_data.append(row_corrs)
        
    corr_df = pd.DataFrame(corr_data, index=data.columns, columns=predictors)
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                linewidths=1, linecolor='white')
    
    plt.title(f'Which version of {target_base} is easiest to predict?', fontsize=14, fontweight='bold')
    plt.ylabel('Target Variations')
    plt.xlabel('Predictors')
    plt.tight_layout()
    
    safe_name = target_base.replace('_', '')
    save_path = OUTPUT_DIR / f'heatmap_{safe_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   [OK] Saved {save_path}")

def plot_correlation_heatmaps(df):
    """
    Orchestrator for the 5 separate heatmaps.
    """
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    predictors = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Height_Log']
    
    # Ensure predictors exist
    for p in predictors:
        if p not in df.columns:
            print(f"Warning: Predictor {p} not found in dataframe.")
            return

    for target in targets:
        if target in df.columns:
            plot_target_analysis(df, target, predictors)
        else:
            print(f"Warning: Target {target} not found in dataframe.")

def main():
    print("="*60)
    print("RATIO CORRELATION ANALYSIS (5 Targets)")
    print("="*60)
    
    df = load_data()
    if df is None:
        return
        
    df = compute_derived_features(df)
    
    plot_correlation_heatmaps(df) # Updated main call
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
