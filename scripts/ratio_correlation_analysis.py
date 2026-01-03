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

def plot_correlation_heatmap(df):
    """
    Plot heatmap for all 5 target variables against:
    NDVI, Height, Height(Log), and Ratio Features.
    """
    print("Generating correlation heatmap...")
    
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    features = [
        'Pre_GSHH_NDVI', 
        'Height_Clean', 
        'Height_Log',
        'Dead_per_cm_Height',
        'Dead_to_Total_Ratio',
        'Dead_to_GDM_Ratio'
    ]
    
    # Filter for columns that exist
    valid_targets = [t for t in targets if t in df.columns]
    valid_features = [f for f in features if f in df.columns]
    
    # Calculate Correlation Matrix
    # We want rows=Targets, cols=Features
    corr_matrix = pd.DataFrame(index=valid_targets, columns=valid_features)
    
    for target in valid_targets:
        for feature in valid_features:
            # Drop NaNs for valid correlation
            valid_mask = df[[target, feature]].notna().all(axis=1)
            if valid_mask.sum() > 10:
                r, _ = stats.pearsonr(df.loc[valid_mask, target], df.loc[valid_mask, feature])
                corr_matrix.loc[target, feature] = r
            else:
                corr_matrix.loc[target, feature] = np.nan
                
    corr_matrix = corr_matrix.astype(float)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', 
                linewidths=1, linecolor='white')
    
    plt.title('Correlation: Biomass Targets vs Derived Features', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Features (Derived & Raw)', fontsize=12, fontweight='bold')
    plt.ylabel('Biomass Targets', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'ratio_correlation_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   [OK] Heatmap saved to {save_path}")
    
    # Save CSV
    corr_matrix.to_csv(OUTPUT_DIR / 'ratio_correlation_matrix.csv')
    print("   [OK] Correlation matrix saved to CSV")

def main():
    print("="*60)
    print("RATIO CORRELATION ANALYSIS (Requested Steps)")
    print("="*60)
    
    df = load_data()
    if df is None:
        return
        
    df = compute_derived_features(df)
    
    plot_correlation_heatmap(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
