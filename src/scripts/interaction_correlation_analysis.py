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
    df['Height_Clean_Log'] = np.log1p(df['Height_Clean'])
    
    # Systematic Interaction Generation
    epsilon = 1e-6
    
    base_features = {
        'NDVI': df['Pre_GSHH_NDVI'],
        'H': df['Height_Clean'],
        'logH': df['Height_Clean_Log']
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

def print_best_derived_targets(corr_df):
    """
    Analyzes the correlation matrix to find the best derived form for each target.
    Prints a sorted list of top correlations for each target.
    """
    print("\n" + "="*80)
    print("BEST DERIVED TARGETS SUMMARY (Top 5 per Target)")
    print("="*80)
    
    base_targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    for base in base_targets:
        print(f"\nTarget: {base}")
        print(f"{'-'*60}")
        print(f"{'Correlation':<12} | {'Derived Form':<40} | {'Predictor'}")
        print(f"{'-'*60}")
        
        # Filter rows for this base target
        relevant_rows = [idx for idx in corr_df.index if base in idx]
        if not relevant_rows:
            continue
            
        sub_df = corr_df.loc[relevant_rows]
        
        # Flatten and sort
        correlations = []
        for derived_t in sub_df.index:
            for feat in sub_df.columns:
                val = sub_df.loc[derived_t, feat]
                if pd.notna(val):
                    correlations.append({
                        'derived': derived_t,
                        'predictor': feat,
                        'corr': val,
                        'abs_corr': abs(val)
                    })
        
        # Sort by absolute correlation desc
        correlations.sort(key=lambda x: x['abs_corr'], reverse=True)
        
        # Save to CSV
        results_df = pd.DataFrame(correlations)
        csv_name = f'best_derived_{base}.csv'
        results_df.to_csv(OUTPUT_DIR / csv_name, index=False)
        print(f"   [Saved] {csv_name}")
        
        # Print top 5
        for item in correlations[:5]:
            print(f"{item['corr']:<12.4f} | {item['derived']:<40} | {item['predictor']}")

def plot_giant_heatmap(df):
    """
    Plot a giant heatmap looking for correlations between:
    Rows: Derived Targets (Target <op> Feature)
    Cols: Features
    """
    print("Generating giant interaction heatmap...")
    
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # 1. Identify Feature Columns
    base_feats = ['Pre_GSHH_NDVI', 'Height_Clean', 'Height_Clean_Log']
    interaction_cols = [c for c in df.columns if any(x in c for x in ['+', '-', '*', '_div_'])]
    feature_cols = base_feats + sorted(interaction_cols)
    
    # Filter for validity
    valid_targets = [t for t in targets if t in df.columns]
    valid_features = [f for f in feature_cols if f in df.columns]
    
    if not valid_targets or not valid_features:
        print("No valid targets or features found.")
        return

    print(f"Features: {len(valid_features)}")
    print(f"Targets: {len(valid_targets)}")
    
    # 2. Build Rows (Derived Targets)
    # We will build a list of dictionaries to construct the correlation DataFrame
    epsilon = 1e-6
    correlation_rows = []
    row_labels = []
    
    for target in valid_targets:
        t_series = df[target]
        
        # Also include the Raw Target itself
        corrs = []
        for feat in valid_features:
            f_series = df[feat]
            # Correlation(Target, Feature)
            # Standard Pearson
            valid = t_series.notna() & f_series.notna()
            if valid.sum() > 10:
                r, _ = stats.pearsonr(t_series[valid], f_series[valid])
            else:
                r = np.nan
            corrs.append(r)
        
        correlation_rows.append(corrs)
        row_labels.append(f"RAW: {target}")
        
        # Now interactions
        for inter_feat in valid_features:
            f_inter = df[inter_feat]
            
            # Operations
            ops = {
                f'{target} * {inter_feat}': t_series * f_inter,
                f'{target} / {inter_feat}': t_series / (f_inter + epsilon),
                f'{target} + {inter_feat}': t_series + f_inter,
                f'{target} - {inter_feat}': t_series - f_inter
            }
            
            for op_name, derived_series in ops.items():
                # Correlate this derived series against ALL features
                row_corrs = []
                for feat in valid_features:
                    f_col = df[feat]
                    valid = derived_series.notna() & f_col.notna()
                    
                    if valid.sum() > 10:
                        r, _ = stats.pearsonr(derived_series[valid], f_col[valid])
                    else:
                        r = np.nan
                    row_corrs.append(r)
                
                correlation_rows.append(row_corrs)
                row_labels.append(op_name)

    # 3. Create DataFrame
    corr_df = pd.DataFrame(correlation_rows, index=row_labels, columns=valid_features)
    corr_df = corr_df.astype(float)
    
    # 3b. Print Top Correlation Summary
    print_best_derived_targets(corr_df)
    
    # 4. Plotting
    # This matrix is HUGE. Rows ~ 5 * (1 + 11*4) = 225. Cols = 11.
    n_rows = len(corr_df)
    n_cols = len(corr_df.columns)
    
    # Dynamic Height: 0.25 inch per row is comfortable
    fig_height = max(10, n_rows * 0.25)
    fig_width = max(12, n_cols * 1.2)
    
    print(f"Plotting heatmap ({n_rows} rows x {n_cols} cols). Output size: {fig_width:.1f}x{fig_height:.1f} inches.")
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Use a diverging colormap to show +/- 1 correlations clearly
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, fmt='.2f', 
                linewidths=0.5, linecolor='white', cbar_kws={"shrink": 0.5})
    
    plt.title('Giant Interaction Correlation Heatmap\n(Rows: Derived Targets, Cols: Features)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Base & Interaction Features', fontsize=14, fontweight='bold')
    plt.ylabel('Derived Targets (Target <op> Feature)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'giant_interaction_heatmap.png'
    # Increase limit for large images correlation map
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight') # Lower DPI for huge image to save memory
        print(f"   [OK] Heatmap saved to {save_path}")
    except Exception as e:
        print(f"   [Error] Failed to save image: {e}")
    
    plt.close()
    
    # Save CSV
    corr_df.to_csv(OUTPUT_DIR / 'giant_interaction_matrix.csv')
    print("   [OK] Matrix saved to CSV")

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
