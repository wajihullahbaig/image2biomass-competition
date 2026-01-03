#!/usr/bin/env python3
"""
Correlation Analysis for NDVI, Height Features, and Species
Analyzes relationships between features and target variables for engineered species.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Configuration
INPUT_CSV = Path('./wide.csv')
OUTPUT_DIR = Path('./analysis_results/correlation_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature groups
NDVI_FEATURES = ['Pre_GSHH_NDVI']
HEIGHT_FEATURES = ['Height_Ave_cm', 'Height_Ave_cm_log']
ENGINEERED_FEATURES = ['Interaction_Mul']  # NDVI * Height interaction
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

SPECIES_COLS = [
    'Species_clover', 'Species_whiteclover', 'Species_subcloverdalkeith', 
    'Species_subcloverlosa', 'Species_ryegrass', 'Species_phalaris', 
    'Species_fescue', 'Species_lucerne', 'Species_barleygrass', 
    'Species_silvergrass', 'Species_speargrass', 'Species_bromegrass',
    'Species_capeweed', 'Species_crumbweed'
]

def load_data():
    """Load and prepare data."""
    df = pd.read_csv(INPUT_CSV)
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    return df

def create_correlation_matrix_plot(df, features, title, filename, figsize=(12, 10)):
    """Create a comprehensive correlation heatmap."""
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   [OK] {filename}')
    
    return corr_matrix

def create_clustered_heatmap(df, features, title, filename):
    """Create hierarchically clustered correlation heatmap."""
    corr_matrix = df[features].corr()
    
    # Perform hierarchical clustering
    dissimilarity = 1 - abs(corr_matrix)
    Z = hierarchy.linkage(squareform(dissimilarity), method='average')
    
    # Create clustermap
    g = sns.clustermap(corr_matrix, method='average', cmap='RdBu_r', 
                       center=0, vmin=-1, vmax=1,
                       annot=True, fmt='.2f', 
                       figsize=(14, 12),
                       cbar_kws={"shrink": 0.8},
                       dendrogram_ratio=0.15)
    
    g.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   [OK] {filename}')

def analyze_feature_target_correlations(df):
    """Analyze correlations between features and target variables."""
    print('\n Feature-Target Correlations:')
    print('='*80)
    
    all_features = NDVI_FEATURES + HEIGHT_FEATURES + ENGINEERED_FEATURES + SPECIES_COLS
    
    results = []
    for feature in all_features:
        for target in TARGET_COLS:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(df[feature], df[target])
            
            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = stats.spearmanr(df[feature], df[target])
            
            results.append({
                'Feature': feature,
                'Target': target,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'Significant': '***' if min(pearson_p, spearman_p) < 0.001 
                              else '**' if min(pearson_p, spearman_p) < 0.01
                              else '*' if min(pearson_p, spearman_p) < 0.05
                              else ''
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Pearson_r', key=abs, ascending=False)
    
    # Save full results
    results_df.to_csv(OUTPUT_DIR / 'feature_target_correlations.csv', index=False)
    print('   [OK] feature_target_correlations.csv')
    
    # Print top correlations
    print('\n   Top 20 Strongest Feature-Target Correlations (by absolute Pearson r):')
    print('   ' + '-'*76)
    for _, row in results_df.head(20).iterrows():
        print(f'   {row["Feature"]:<35} -> {row["Target"]:<15} '
              f'r={row["Pearson_r"]:>6.3f} {row["Significant"]}')
    
    return results_df

def create_feature_target_heatmap(df):
    """Create heatmap of feature-target correlations."""
    all_features = NDVI_FEATURES + HEIGHT_FEATURES + ENGINEERED_FEATURES + SPECIES_COLS
    
    # Calculate correlations
    corr_data = []
    for feature in all_features:
        row = []
        for target in TARGET_COLS:
            r, _ = stats.pearsonr(df[feature], df[target])
            row.append(r)
        corr_data.append(row)
    
    corr_df = pd.DataFrame(corr_data, index=all_features, columns=TARGET_COLS)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature-Target Correlations (Pearson r)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_target_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] feature_target_heatmap.png')

def analyze_ndvi_height_interactions(df):
    """Analyze interaction between NDVI and Height."""
    print('\n NDVI × Height Interaction Analysis:')
    print('='*80)
    
    # Create scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGET_COLS):
        ax = axes[idx]
        
        # Create scatter with color gradient based on NDVI
        scatter = ax.scatter(df['Height_Ave_cm'], df[target], 
                           c=df['Pre_GSHH_NDVI'], cmap='viridis',
                           s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NDVI', fontsize=10)
        
        # Calculate correlations
        r_height, _ = stats.pearsonr(df['Height_Ave_cm'], df[target])
        r_ndvi, _ = stats.pearsonr(df['Pre_GSHH_NDVI'], df[target])
        r_interaction, _ = stats.pearsonr(df['Interaction_Mul'], df[target])
        
        ax.set_xlabel('Height (cm)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{target} (g)', fontsize=11, fontweight='bold')
        ax.set_title(f'{target}\nHeight r={r_height:.3f}, NDVI r={r_ndvi:.3f}, '
                    f'Interaction r={r_interaction:.3f}', 
                    fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    fig.suptitle('Height vs Targets (colored by NDVI)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ndvi_height_interaction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] ndvi_height_interaction.png')

def analyze_species_effects(df):
    """Analyze how species affect correlations."""
    print('\n Species-Specific Correlation Analysis:')
    print('='*80)
    
    # Get dominant species for each sample (species with value = 1.0)
    df['dominant_species'] = df[SPECIES_COLS].idxmax(axis=1).str.replace('Species_', '')
    
    # Only analyze species with at least 10 samples
    species_counts = df['dominant_species'].value_counts()
    common_species = species_counts[species_counts >= 10].index.tolist()
    
    results = []
    
    print(f'\n   Analyzing {len(common_species)} species with ≥10 samples:')
    print('   ' + '-'*76)
    
    for species in common_species:
        species_df = df[df['dominant_species'] == species]
        n_samples = len(species_df)
        
        # Calculate correlations for this species
        for feature in ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Interaction_Mul']:
            for target in ['Dry_Total_g', 'GDM_g']:
                r, p = stats.pearsonr(species_df[feature], species_df[target])
                
                results.append({
                    'Species': species,
                    'N_Samples': n_samples,
                    'Feature': feature,
                    'Target': target,
                    'Correlation': r,
                    'P_Value': p,
                    'Significant': '***' if p < 0.001 else '**' if p < 0.01 
                                  else '*' if p < 0.05 else ''
                })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['Species', 'Target', 'Correlation'], 
                                       ascending=[True, True, False])
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / 'species_specific_correlations.csv', index=False)
    print('   [OK] species_specific_correlations.csv')
    
    # Create visualization
    pivot_data = results_df[results_df['Target'] == 'Dry_Total_g'].pivot(
        index='Species', columns='Feature', values='Correlation'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Species-Specific Feature Correlations with Dry_Total_g', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Species', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'species_specific_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] species_specific_heatmap.png')
    
    return results_df

def analyze_multicollinearity(df):
    """Analyze multicollinearity among features using VIF."""
    print('\n Multicollinearity Analysis (VIF):')
    print('='*80)
    
    from sklearn.preprocessing import StandardScaler
    
    features = NDVI_FEATURES + HEIGHT_FEATURES + ENGINEERED_FEATURES
    X = df[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate VIF for each feature
    vif_data = []
    for i, feature in enumerate(features):
        # VIF calculation: 1 / (1 - R²)
        # R² from regression of feature i on all other features
        from sklearn.linear_model import LinearRegression
        
        # Get other features
        X_others = np.delete(X_scaled, i, axis=1)
        y = X_scaled[:, i]
        
        # Fit model
        model = LinearRegression()
        model.fit(X_others, y)
        r_squared = model.score(X_others, y)
        
        # Calculate VIF
        vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        
        vif_data.append({
            'Feature': feature,
            'VIF': vif,
            'Warning': '[WARNING] High' if vif > 10 else '[OK] OK' if vif < 5 else '[MODERATE] Moderate'
        })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    print('\n   VIF Scores (Variance Inflation Factor):')
    print('   ' + '-'*60)
    print('   VIF < 5: No multicollinearity concern')
    print('   VIF 5-10: Moderate multicollinearity')
    print('   VIF > 10: High multicollinearity (consider removing)')
    print('   ' + '-'*60)
    
    for _, row in vif_df.iterrows():
        print(f'   {row["Feature"]:<25} VIF = {row["VIF"]:>7.2f}  {row["Warning"]}')
    
    vif_df.to_csv(OUTPUT_DIR / 'vif_analysis.csv', index=False)
    print('\n   [OK] vif_analysis.csv')
    
    return vif_df

def create_pairplot(df):
    """Create pairplot of key features and targets."""
    print('\n Creating Pairplot:')
    print('='*80)
    
    # Select subset of columns for clarity
    plot_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Interaction_Mul', 
                 'Dry_Total_g', 'GDM_g']
    
    # Get dominant species for coloring
    df['dominant_species'] = df[SPECIES_COLS].idxmax(axis=1).str.replace('Species_', '')
    
    # Only use common species for coloring
    species_counts = df['dominant_species'].value_counts()
    common_species = species_counts[species_counts >= 10].index.tolist()[:5]  # Top 5
    
    plot_df = df[df['dominant_species'].isin(common_species)][plot_cols + ['dominant_species']].copy()
    
    # Create pairplot
    g = sns.pairplot(plot_df, hue='dominant_species', 
                     diag_kind='kde', plot_kws={'alpha': 0.6, 's': 40},
                     corner=True, height=2.5)
    
    g.fig.suptitle('Feature and Target Relationships (Top 5 Species)', 
                  fontsize=14, fontweight='bold', y=1.001)
    
    plt.savefig(OUTPUT_DIR / 'pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   [OK] pairplot.png')

def generate_summary_report(df, feature_target_corr, species_corr, vif_df):
    """Generate comprehensive summary report."""
    report_path = OUTPUT_DIR / 'correlation_summary_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('='*80 + '\n')
        f.write('CORRELATION ANALYSIS SUMMARY REPORT\n')
        f.write('='*80 + '\n\n')
        
        f.write('Dataset Information:\n')
        f.write(f'  Total Samples: {len(df)}\n')
        f.write(f'  Date Range: {df["Sampling_Date"].min().date()} to {df["Sampling_Date"].max().date()}\n')
        f.write(f'  States: {", ".join(sorted(df["State"].unique()))}\n')
        f.write(f'  Species Combinations: {df["Species"].nunique()}\n\n')
        
        f.write('-'*80 + '\n')
        f.write('TOP 10 STRONGEST FEATURE-TARGET CORRELATIONS\n')
        f.write('-'*80 + '\n')
        top_corr = feature_target_corr.head(10)
        for _, row in top_corr.iterrows():
            f.write(f'{row["Feature"]:<35} -> {row["Target"]:<15} '
                   f'r={row["Pearson_r"]:>6.3f} {row["Significant"]}\n')
        
        f.write('\n' + '-'*80 + '\n')
        f.write('KEY INSIGHTS\n')
        f.write('-'*80 + '\n\n')
        
        # NDVI insights
        ndvi_corrs = feature_target_corr[feature_target_corr['Feature'] == 'Pre_GSHH_NDVI']
        avg_ndvi_corr = ndvi_corrs['Pearson_r'].mean()
        f.write(f'1. NDVI Correlations:\n')
        f.write(f'   Average correlation with targets: {avg_ndvi_corr:.3f}\n')
        strongest = ndvi_corrs.loc[ndvi_corrs['Pearson_r'].abs().idxmax()]
        f.write(f'   Strongest: {strongest["Target"]} (r={strongest["Pearson_r"]:.3f})\n\n')
        
        # Height insights
        height_corrs = feature_target_corr[feature_target_corr['Feature'] == 'Height_Ave_cm']
        avg_height_corr = height_corrs['Pearson_r'].mean()
        f.write(f'2. Height Correlations:\n')
        f.write(f'   Average correlation with targets: {avg_height_corr:.3f}\n')
        strongest = height_corrs.loc[height_corrs['Pearson_r'].abs().idxmax()]
        f.write(f'   Strongest: {strongest["Target"]} (r={strongest["Pearson_r"]:.3f})\n\n')
        
        # Interaction insights
        int_corrs = feature_target_corr[feature_target_corr['Feature'] == 'Interaction_Mul']
        avg_int_corr = int_corrs['Pearson_r'].mean()
        f.write(f'3. Interaction Feature (NDVI × Height):\n')
        f.write(f'   Average correlation with targets: {avg_int_corr:.3f}\n')
        strongest = int_corrs.loc[int_corrs['Pearson_r'].abs().idxmax()]
        f.write(f'   Strongest: {strongest["Target"]} (r={strongest["Pearson_r"]:.3f})\n\n')
        
        # Multicollinearity
        f.write(f'4. Multicollinearity (VIF):\n')
        for _, row in vif_df.iterrows():
            f.write(f'   {row["Feature"]:<25} VIF={row["VIF"]:>7.2f}  {row["Warning"]}\n')
        f.write('\n')
        
        # Species-specific
        f.write(f'5. Species-Specific Correlations:\n')
        f.write(f'   Analyzed {species_corr["Species"].nunique()} species with ≥10 samples\n')
        f.write(f'   Range of NDVI->Dry_Total_g correlations: ')
        ndvi_dry = species_corr[(species_corr['Feature']=='Pre_GSHH_NDVI') & 
                                (species_corr['Target']=='Dry_Total_g')]
        f.write(f'{ndvi_dry["Correlation"].min():.3f} to {ndvi_dry["Correlation"].max():.3f}\n\n')
        
        f.write('-'*80 + '\n')
        f.write('RECOMMENDATIONS\n')
        f.write('-'*80 + '\n\n')
        
        # Generate recommendations
        if vif_df['VIF'].max() > 10:
            high_vif = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
            f.write(f'[WARNING]  High multicollinearity detected: {", ".join(high_vif)}\n')
            f.write(f'   Consider removing one of the correlated features.\n\n')
        
        if avg_int_corr > max(avg_ndvi_corr, avg_height_corr):
            f.write(f'[OK] Interaction feature shows stronger correlations than individual features.\n')
            f.write(f'   Keep this engineered feature in the model.\n\n')
        
        f.write(f'[OK] Use species-specific correlations to guide model architecture.\n')
        f.write(f'   Consider species-aware feature engineering or stratification.\n\n')
        
        f.write('='*80 + '\n')
        f.write('END OF REPORT\n')
        f.write('='*80 + '\n')
    
    print(f'   [OK] correlation_summary_report.txt')

def main():
    """Main execution function."""
    print('='*80)
    print('CORRELATION ANALYSIS: NDVI, HEIGHT, AND SPECIES')
    print('='*80)
    
    # Load data
    print('\n Loading data...')
    df = load_data()
    print(f'   [OK] Loaded {len(df)} samples')
    
    # 1. Full correlation matrix (all features)
    print('\n Creating Correlation Matrices:')
    print('='*80)
    
    all_features = NDVI_FEATURES + HEIGHT_FEATURES + ENGINEERED_FEATURES + TARGET_COLS
    create_correlation_matrix_plot(
        df, all_features,
        'Full Correlation Matrix: Features + Targets',
        'full_correlation_matrix.png',
        figsize=(12, 10)
    )
    
    # 2. Species correlation matrix
    species_and_targets = SPECIES_COLS + TARGET_COLS
    create_clustered_heatmap(
        df, species_and_targets,
        'Clustered Correlation: Species + Targets',
        'species_correlation_clustered.png'
    )
    
    # 3. Feature-target correlations
    feature_target_corr = analyze_feature_target_correlations(df)
    create_feature_target_heatmap(df)
    
    # 4. NDVI × Height interaction analysis
    analyze_ndvi_height_interactions(df)
    
    # 5. Species-specific analysis
    species_corr = analyze_species_effects(df)
    
    # 6. Multicollinearity analysis
    vif_df = analyze_multicollinearity(df)
    
    # 7. Pairplot
    create_pairplot(df)
    
    # 8. Generate summary report
    print('\n Generating Summary Report:')
    print('='*80)
    generate_summary_report(df, feature_target_corr, species_corr, vif_df)
    
    print('\n' + '='*80)
    print('[COMPLETE] CORRELATION ANALYSIS COMPLETE')
    print('='*80)
    print(f'\n📁 All outputs saved to: {OUTPUT_DIR}/')
    print('\nGenerated files:')
    print('   • full_correlation_matrix.png')
    print('   • species_correlation_clustered.png')
    print('   • feature_target_heatmap.png')
    print('   • ndvi_height_interaction.png')
    print('   • species_specific_heatmap.png')
    print('   • pairplot.png')
    print('   • feature_target_correlations.csv')
    print('   • species_specific_correlations.csv')
    print('   • vif_analysis.csv')
    print('   • correlation_summary_report.txt')
    print('='*80 + '\n')

if __name__ == '__main__':
    main()