#!/usr/bin/env python3
"""
Advanced Correlation Visualizations
Additional analyses and visualizations for deeper insights.
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
OUTPUT_DIR = Path('./analysis_results/correlation_analysis_advanced')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and prepare data."""
    df = pd.read_csv(INPUT_CSV)
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    
    # Add dominant species
    species_cols = [col for col in df.columns if col.startswith('Species_')]
    df['dominant_species'] = df[species_cols].idxmax(axis=1).str.replace('Species_', '')
    
    return df

def plot_correlation_by_state(df):
    """Compare correlations across different states."""
    print('\n State-Specific Correlation Analysis:')
    print('='*80)
    
    states = sorted(df['State'].unique())
    features = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Interaction_Mul']
    targets = ['Dry_Total_g', 'GDM_g']
    
    results = []
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, target in enumerate(targets):
        for jdx, feature in enumerate(features[:2]):  # NDVI and Height only
            ax = axes[idx, jdx]
            
            for state in states:
                state_df = df[df['State'] == state]
                
                if len(state_df) < 10:
                    continue
                
                # Calculate correlation
                r, p = stats.pearsonr(state_df[feature], state_df[target])
                
                results.append({
                    'State': state,
                    'Feature': feature,
                    'Target': target,
                    'Correlation': r,
                    'P_Value': p,
                    'N_Samples': len(state_df)
                })
                
                # Plot scatter
                ax.scatter(state_df[feature], state_df[target], 
                          label=f'{state} (r={r:.2f}, n={len(state_df)})',
                          alpha=0.5, s=50)
            
            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel(target, fontsize=11, fontweight='bold')
            ax.set_title(f'{feature} vs {target}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'state_specific_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] state_specific_correlations.png')
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'state_correlations.csv', index=False)
    print('   [OK] state_correlations.csv')
    
    return results_df

def plot_partial_correlations(df):
    """Calculate and visualize partial correlations."""
    print('\n Partial Correlation Analysis:')
    print('='*80)
    print('   (Correlation between two variables after removing the effect of others)')
    
    from sklearn.linear_model import LinearRegression
    
    features = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    targets = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    
    partial_corrs = {}
    
    for target in targets:
        partial_corrs[target] = {}
        
        for i, feat1 in enumerate(features):
            # Calculate partial correlation of feat1 with target, controlling for other features
            other_features = [f for f in features if f != feat1]
            
            # Regress out other features from feat1
            X_other = df[other_features].values
            y_feat1 = df[feat1].values
            model1 = LinearRegression()
            model1.fit(X_other, y_feat1)
            residuals_feat1 = y_feat1 - model1.predict(X_other)
            
            # Regress out other features from target
            y_target = df[target].values
            model2 = LinearRegression()
            model2.fit(X_other, y_target)
            residuals_target = y_target - model2.predict(X_other)
            
            # Correlation between residuals is the partial correlation
            partial_r, partial_p = stats.pearsonr(residuals_feat1, residuals_target)
            
            # Also get regular correlation for comparison
            regular_r, _ = stats.pearsonr(df[feat1], df[target])
            
            partial_corrs[target][feat1] = {
                'partial_r': partial_r,
                'regular_r': regular_r,
                'p_value': partial_p
            }
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for plotting
    feat_names = features
    
    for idx, target in enumerate(targets[:2]):  # Plot first 2 targets
        ax = axes[idx] if idx < 2 else None
        if ax is None:
            continue
        
        regular_vals = [partial_corrs[target][f]['regular_r'] for f in feat_names]
        partial_vals = [partial_corrs[target][f]['partial_r'] for f in feat_names]
        
        x = np.arange(len(feat_names))
        width = 0.35
        
        ax.bar(x - width/2, regular_vals, width, label='Regular Correlation', alpha=0.8)
        ax.bar(x + width/2, partial_vals, width, label='Partial Correlation', alpha=0.8)
        
        ax.set_xlabel('Feature', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
        ax.set_title(f'Correlations with {target}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('_', '\n') for f in feat_names], fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partial_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] partial_correlations.png')
    
    # Save detailed results
    rows = []
    for target in targets:
        for feat in features:
            rows.append({
                'Target': target,
                'Feature': feat,
                'Regular_Correlation': partial_corrs[target][feat]['regular_r'],
                'Partial_Correlation': partial_corrs[target][feat]['partial_r'],
                'P_Value': partial_corrs[target][feat]['p_value']
            })
    
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'partial_correlations.csv', index=False)
    print('   [OK] partial_correlations.csv')

def plot_correlation_by_height_bins(df):
    """Analyze how NDVI correlation changes across height ranges."""
    print('\n NDVI Correlation by Height Range:')
    print('='*80)
    
    # Create height bins
    df['height_bin'] = pd.qcut(df['Height_Ave_cm'], q=4, 
                                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    
    targets = ['Dry_Total_g', 'GDM_g']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    results = []
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        bins = df['height_bin'].cat.categories
        correlations = []
        sample_counts = []
        
        for bin_name in bins:
            bin_df = df[df['height_bin'] == bin_name]
            r, p = stats.pearsonr(bin_df['Pre_GSHH_NDVI'], bin_df[target])
            correlations.append(r)
            sample_counts.append(len(bin_df))
            
            results.append({
                'Height_Bin': bin_name,
                'Target': target,
                'NDVI_Correlation': r,
                'P_Value': p,
                'N_Samples': len(bin_df),
                'Height_Range': f'{bin_df["Height_Ave_cm"].min():.1f}-{bin_df["Height_Ave_cm"].max():.1f} cm'
            })
        
        # Bar plot
        x = np.arange(len(bins))
        bars = ax.bar(x, correlations, alpha=0.7, color='steelblue')
        
        # Add sample counts on bars
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Height Quartile', fontsize=11, fontweight='bold')
        ax.set_ylabel('NDVI Correlation', fontsize=11, fontweight='bold')
        ax.set_title(f'NDVI Correlation with {target}\nby Height Quartile', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bins, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ndvi_correlation_by_height.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] ndvi_correlation_by_height.png')
    
    # Save results
    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'ndvi_by_height_bins.csv', index=False)
    print('   [OK] ndvi_by_height_bins.csv')

def plot_correlation_by_ndvi_bins(df):
    """Analyze how Height correlation changes across NDVI ranges."""
    print('\n Height Correlation by NDVI Range:')
    print('='*80)
    
    # Create NDVI bins
    df['ndvi_bin'] = pd.qcut(df['Pre_GSHH_NDVI'], q=4, 
                              labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    
    targets = ['Dry_Total_g', 'GDM_g']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    results = []
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        bins = df['ndvi_bin'].cat.categories
        correlations = []
        sample_counts = []
        
        for bin_name in bins:
            bin_df = df[df['ndvi_bin'] == bin_name]
            r, p = stats.pearsonr(bin_df['Height_Ave_cm'], bin_df[target])
            correlations.append(r)
            sample_counts.append(len(bin_df))
            
            results.append({
                'NDVI_Bin': bin_name,
                'Target': target,
                'Height_Correlation': r,
                'P_Value': p,
                'N_Samples': len(bin_df),
                'NDVI_Range': f'{bin_df["Pre_GSHH_NDVI"].min():.2f}-{bin_df["Pre_GSHH_NDVI"].max():.2f}'
            })
        
        # Bar plot
        x = np.arange(len(bins))
        bars = ax.bar(x, correlations, alpha=0.7, color='forestgreen')
        
        # Add sample counts on bars
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('NDVI Quartile', fontsize=11, fontweight='bold')
        ax.set_ylabel('Height Correlation', fontsize=11, fontweight='bold')
        ax.set_title(f'Height Correlation with {target}\nby NDVI Quartile', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bins, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'height_correlation_by_ndvi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] height_correlation_by_ndvi.png')
    
    # Save results
    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'height_by_ndvi_bins.csv', index=False)
    print('   [OK] height_by_ndvi_bins.csv')

def plot_residual_analysis(df):
    """Analyze residuals to understand model fit quality."""
    print('\n Residual Analysis:')
    print('='*80)
    
    from sklearn.linear_model import LinearRegression
    
    # Build simple linear models and analyze residuals
    features = [['Pre_GSHH_NDVI'], ['Height_Ave_cm'], ['Pre_GSHH_NDVI', 'Height_Ave_cm']]
    feature_names = ['NDVI Only', 'Height Only', 'NDVI + Height']
    target = 'Dry_Total_g'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (feat_list, feat_name) in enumerate(zip(features, feature_names)):
        X = df[feat_list].values
        y = df[target].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # R² score
        r2 = model.score(X, y)
        
        # Plot 1: Residuals vs Fitted
        ax1 = axes[0, idx]
        ax1.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Fitted Values', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Residuals', fontsize=10, fontweight='bold')
        ax1.set_title(f'{feat_name}\nR² = {r2:.3f}', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot
        ax2 = axes[1, idx]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot: {feat_name}', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] residual_analysis.png')

def create_advanced_summary():
    """Create summary document for advanced analysis."""
    summary_path = OUTPUT_DIR / 'advanced_analysis_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write('ADVANCED CORRELATION ANALYSIS SUMMARY\n')
        f.write('='*80 + '\n\n')
        
        f.write('This analysis provides deeper insights into feature correlations:\n\n')
        
        f.write('1. STATE-SPECIFIC CORRELATIONS\n')
        f.write('   Different states show varying correlation patterns.\n')
        f.write('   This suggests geographic factors influence feature-target relationships.\n\n')
        
        f.write('2. PARTIAL CORRELATIONS\n')
        f.write('   Removes the effect of other variables to find true relationships.\n')
        f.write('   Useful for understanding unique contribution of each feature.\n\n')
        
        f.write('3. CONDITIONAL CORRELATIONS\n')
        f.write('   NDVI correlation varies by height range (and vice versa).\n')
        f.write('   Suggests non-linear interactions between features.\n\n')
        
        f.write('4. RESIDUAL ANALYSIS\n')
        f.write('   Checks model assumptions and fit quality.\n')
        f.write('   Q-Q plots assess normality of residuals.\n\n')
        
        f.write('GENERATED FILES:\n')
        f.write('   • state_specific_correlations.png\n')
        f.write('   • partial_correlations.png\n')
        f.write('   • ndvi_correlation_by_height.png\n')
        f.write('   • height_correlation_by_ndvi.png\n')
        f.write('   • residual_analysis.png\n')
        f.write('   • Various CSV files with detailed statistics\n\n')
        
        f.write('='*80 + '\n')
    
    print('   [OK] advanced_analysis_summary.txt')

def main():
    """Main execution."""
    print('='*80)
    print('ADVANCED CORRELATION ANALYSIS')
    print('='*80)
    
    # Load data
    print('\n Loading data...')
    df = load_data()
    print(f'   [OK] Loaded {len(df)} samples')
    
    # Run analyses
    plot_correlation_by_state(df)
    plot_partial_correlations(df)
    plot_correlation_by_height_bins(df)
    plot_correlation_by_ndvi_bins(df)
    plot_residual_analysis(df)
    create_advanced_summary()
    
    print('\n' + '='*80)
    print('[COMPLETE] ADVANCED ANALYSIS COMPLETE')
    print('='*80)
    print(f'\n All outputs saved to: {OUTPUT_DIR}/')
    print('='*80 + '\n')

if __name__ == '__main__':
    main()