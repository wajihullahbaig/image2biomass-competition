#!/usr/bin/env python3
"""
Interaction Correlation Analysis
Analyzes correlations between biomass targets and feature interactions,
specifically focusing on NDVI and Height combinations.
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
OUTPUT_DIR = Path('./analysis_results/interaction_analysis')
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

def compute_interaction_features(df):
    """Compute interaction features for analysis."""
    print("Computing interaction features...")
    
    # 1. Clean Height & Log Space
    # Ensure no zero/negative heights for log
    df['Height_Clean'] = df['Height_Ave_cm'].clip(lower=0.1)
    df['Height_Log'] = np.log1p(df['Height_Clean'])
    
    # Systematic Interaction Generation
    epsilon = 1e-6
    
    base_features = {
        'NDVI': df['Pre_GSHH_NDVI'],
        'H': df['Height_Clean'],
        'logH': df['Height_Log']
    }
    
    # 2. Generate Pairs: (NDVI, H) and (NDVI, logH)
    pairs = [('NDVI', 'H'), ('NDVI', 'logH')]
    
    for name1, name2 in pairs:
        s1 = base_features[name1]
        s2 = base_features[name2]
        
        # Multiply (*)
        df[f'{name1}*{name2}'] = s1 * s2
        
        # Add (+)
        df[f'{name1}+{name2}'] = s1 + s2
        
        # Subtract (-) (Both ways)
        df[f'{name1}-{name2}'] = s1 - s2
        df[f'{name2}-{name1}'] = s2 - s1
        
        # Divide (/) (Both ways, protected)
        df[f'{name1}_div_{name2}'] = s1 / (s2 + epsilon)
        df[f'{name2}_div_{name1}'] = s2 / (s1 + epsilon)

    return df

def plot_giant_heatmap(df):
    """
    Plot a giant heatmap for all 5 target variables against
    NDVI, Height, and their interactions.
    """
    print("Generating giant interaction heatmap...")
    
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Identify all feature columns
    # We look for base features + generated interaction columns
    base_feats = ['Pre_GSHH_NDVI', 'Height_Clean', 'Height_Log']
    
    interaction_cols = [c for c in df.columns if any(x in c for x in ['+', '-', '*', '_div_'])]
    
    # Combine and sort for nicer plotting
    features = base_feats + sorted(interaction_cols)
    
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
    
    # Plotting - Adjust size dynamically based on number of features
    width = max(12, len(valid_features) * 0.8)
    height = max(8, len(valid_targets) * 1.5)
    
    plt.figure(figsize=(width, height))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', 
                linewidths=1, linecolor='white')
    
    plt.title('Feature Interaction Correlations\nTargets vs (NDVI, Height, Interactions)', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Features & Interactions', fontsize=12, fontweight='bold')
    plt.ylabel('Biomass Targets', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'interaction_correlation_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   [OK] Heatmap saved to {save_path}")
    
    # Save CSV
    corr_matrix.to_csv(OUTPUT_DIR / 'interaction_correlation_matrix.csv')
    print("   [OK] Correlation matrix saved to CSV")

def main():
    print("="*60)
    print("INTERACTION CORRELATION ANALYSIS")
    print("="*60)
    
    df = load_data()
    if df is None:
        return
        
    df = compute_interaction_features(df)
    
    plot_giant_heatmap(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
