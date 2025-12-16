import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, 
    StratifiedGroupKFold, train_test_split
)
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configuration
N_FOLDS = 5
BASE_VIZ_PATH = 'cv_strategy_analysis'
os.makedirs(BASE_VIZ_PATH, exist_ok=True)

# Visual Settings
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Helper Functions
def get_season(month):
    if month in [12, 1, 2]: return 'Summer'
    elif month in [3, 4, 5]: return 'Autumn'
    elif month in [6, 7, 8]: return 'Winter'
    else: return 'Spring'

def load_and_prepare_data(filepath='train.csv'):
    """Load and prepare data for CV analysis"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Clean sample IDs
    df['sample_id'] = df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
    
    # Pivot to wide format
    targets = df.pivot_table(
        index='sample_id', 
        columns='target_name', 
        values='target', 
        aggfunc='max'
    ).reset_index()
    
    # Get metadata
    meta_cols = ['sample_id', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    meta = df[meta_cols].drop_duplicates(subset=['sample_id'])
    
    # Merge
    wide = pd.merge(meta, targets, on='sample_id', how='left')
    
    # Fill missing targets
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    for t in target_cols:
        if t not in wide.columns:
            wide[t] = 0.0
    wide[target_cols] = wide[target_cols].fillna(0.0)
    
    # Feature engineering
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'])
    wide['month'] = wide['Sampling_Date'].dt.month
    wide['season'] = wide['month'].apply(get_season)
    wide['year'] = wide['Sampling_Date'].dt.year
    
    # Create species groups (group rare species)
    species_counts = wide['Species'].value_counts()
    wide['species_grouped'] = wide['Species'].apply(
        lambda x: x if species_counts[x] >= 10 else 'Other'
    )
    
    print(f"Total unique samples: {len(wide)}")
    print(f"Columns: {list(wide.columns)}")
    
    return wide, target_cols

def calculate_fold_statistics(df, fold_assignments, target_cols, strategy_name):
    """Calculate detailed statistics for each fold"""
    stats_list = []
    
    for fold_idx in sorted(df['fold'].unique()):
        fold_mask = df['fold'] == fold_idx
        fold_data = df[fold_mask]
        
        fold_stats = {
            'Strategy': strategy_name,
            'Fold': fold_idx,
            'N_Samples': len(fold_data),
            'N_States': fold_data['State'].nunique(),
            'N_Species': fold_data['Species'].nunique(),
            'N_Seasons': fold_data['season'].nunique()
        }
        
        # Target statistics
        for target in target_cols:
            fold_stats[f'{target}_mean'] = fold_data[target].mean()
            fold_stats[f'{target}_std'] = fold_data[target].std()
            fold_stats[f'{target}_min'] = fold_data[target].min()
            fold_stats[f'{target}_max'] = fold_data[target].max()
        
        stats_list.append(fold_stats)
    
    return pd.DataFrame(stats_list)

def analyze_cv_strategy(df, strategy_name, splitter, group_col=None, stratify_col=None):
    """Analyze a single CV strategy"""
    print(f"\nAnalyzing: {strategy_name}")
    
    df_copy = df.copy()
    df_copy['fold'] = -1
    
    # Prepare inputs for splitter
    X = df_copy.index.values
    y = df_copy['season'].values if stratify_col == 'season' else df_copy.index.values
    groups = df_copy[group_col].values if group_col else None
    
    # Assign folds
    try:
        if isinstance(splitter, GroupKFold):
            splits = splitter.split(X, y, groups)
        elif isinstance(splitter, StratifiedKFold):
            splits = splitter.split(X, y)
        else:  # KFold
            splits = splitter.split(X)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            df_copy.loc[val_idx, 'fold'] = fold_idx
        
        # Check if all samples assigned
        if (df_copy['fold'] == -1).any():
            print(f"  ⚠️  Warning: {(df_copy['fold'] == -1).sum()} samples not assigned to any fold")
            return None
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None
    
    return df_copy

def plot_fold_distributions(results_dict, target_cols):
    """Create comprehensive visualization of fold distributions"""
    print("\nGenerating fold distribution plots...")
    
    n_strategies = len(results_dict)
    fig = plt.figure(figsize=(24, 6 * n_strategies))
    
    for strat_idx, (strategy_name, df) in enumerate(results_dict.items()):
        if df is None:
            continue
        
        # Row for this strategy
        base_idx = strat_idx * 4
        
        # Plot 1: Sample counts per fold
        ax1 = plt.subplot(n_strategies, 4, base_idx + 1)
        fold_counts = df['fold'].value_counts().sort_index()
        bars = ax1.bar(fold_counts.index, fold_counts.values, color='steelblue', alpha=0.7)
        ax1.axhline(len(df) / N_FOLDS, color='red', linestyle='--', 
                    label=f'Expected ({len(df)/N_FOLDS:.0f})', linewidth=2)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Sample Count')
        ax1.set_title(f'{strategy_name}\nSample Distribution')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: State distribution
        ax2 = plt.subplot(n_strategies, 4, base_idx + 2)
        state_dist = pd.crosstab(df['fold'], df['State'], normalize='index') * 100
        state_dist.plot(kind='bar', stacked=True, ax=ax2, colormap='Set3')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('State Distribution per Fold')
        ax2.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        
        # Plot 3: Season distribution
        ax3 = plt.subplot(n_strategies, 4, base_idx + 3)
        season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
        season_dist = pd.crosstab(df['fold'], df['season'], normalize='index') * 100
        season_dist = season_dist.reindex(columns=season_order, fill_value=0)
        season_dist.plot(kind='bar', stacked=True, ax=ax3, 
                        color=['#FF6B6B', '#FFA500', '#4ECDC4', '#95E77D'])
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Season Distribution per Fold')
        ax3.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
        
        # Plot 4: Target mean distribution (Dry_Total_g)
        ax4 = plt.subplot(n_strategies, 4, base_idx + 4)
        fold_means = df.groupby('fold')['Dry_Total_g'].agg(['mean', 'std']).reset_index()
        bars = ax4.bar(fold_means['fold'], fold_means['mean'], 
                      yerr=fold_means['std'], capsize=5, 
                      color='coral', alpha=0.7, error_kw={'linewidth': 2})
        overall_mean = df['Dry_Total_g'].mean()
        ax4.axhline(overall_mean, color='darkgreen', linestyle='--', 
                   label=f'Overall Mean ({overall_mean:.1f}g)', linewidth=2)
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Dry Total (g)')
        ax4.set_title('Target Distribution (Mean ± Std)')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '1_Fold_Distributions_Overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_variance_analysis(results_dict, target_cols):
    """Analyze variance between and within folds"""
    print("\nGenerating variance analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    strategy_names = list(results_dict.keys())
    
    # Plot 1: Between-fold variance for each target
    ax1 = axes[0, 0]
    variance_data = []
    
    for strategy_name, df in results_dict.items():
        if df is None:
            continue
        for target in target_cols:
            fold_means = df.groupby('fold')[target].mean()
            between_var = fold_means.var()
            variance_data.append({
                'Strategy': strategy_name,
                'Target': target.replace('_g', ''),
                'Between_Fold_Var': between_var
            })
    
    var_df = pd.DataFrame(variance_data)
    var_pivot = var_df.pivot(index='Strategy', columns='Target', values='Between_Fold_Var')
    var_pivot.plot(kind='bar', ax=ax1, colormap='viridis')
    ax1.set_ylabel('Between-Fold Variance')
    ax1.set_title('Between-Fold Variance by Target\n(Lower is Better = More Balanced)')
    ax1.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Coefficient of Variation across folds
    ax2 = axes[0, 1]
    cv_data = []
    
    for strategy_name, df in results_dict.items():
        if df is None:
            continue
        for target in target_cols:
            fold_means = df.groupby('fold')[target].mean()
            cv = (fold_means.std() / fold_means.mean() * 100) if fold_means.mean() > 0 else 0
            cv_data.append({
                'Strategy': strategy_name,
                'Target': target.replace('_g', ''),
                'CV (%)': cv
            })
    
    cv_df = pd.DataFrame(cv_data)
    cv_pivot = cv_df.pivot(index='Strategy', columns='Target', values='CV (%)')
    cv_pivot.plot(kind='bar', ax=ax2, colormap='plasma')
    ax2.set_ylabel('Coefficient of Variation (%)')
    ax2.set_title('CV of Fold Means\n(Lower = More Consistent)')
    ax2.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Sample size variance
    ax3 = axes[1, 0]
    size_variance = []
    
    for strategy_name, df in results_dict.items():
        if df is None:
            continue
        fold_sizes = df['fold'].value_counts().sort_index()
        size_variance.append({
            'Strategy': strategy_name,
            'Mean_Size': fold_sizes.mean(),
            'Std_Size': fold_sizes.std(),
            'CV (%)': (fold_sizes.std() / fold_sizes.mean() * 100)
        })
    
    size_df = pd.DataFrame(size_variance)
    x = np.arange(len(size_df))
    bars = ax3.bar(x, size_df['Std_Size'], color='steelblue', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(size_df['Strategy'], rotation=45, ha='right')
    ax3.set_ylabel('Std Dev of Fold Sizes')
    ax3.set_title('Fold Size Consistency\n(Lower = More Balanced)')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add CV labels
    for i, (bar, cv) in enumerate(zip(bars, size_df['CV (%)'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'CV={cv:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Distribution balance score (combined metric)
    ax4 = axes[1, 1]
    balance_scores = []
    
    for strategy_name, df in results_dict.items():
        if df is None:
            continue
        
        # Calculate multiple balance metrics
        fold_sizes = df['fold'].value_counts()
        size_cv = fold_sizes.std() / fold_sizes.mean() * 100
        
        # State balance (entropy-based)
        state_entropy = []
        for fold in df['fold'].unique():
            fold_data = df[df['fold'] == fold]
            state_dist = fold_data['State'].value_counts(normalize=True)
            entropy = stats.entropy(state_dist)
            state_entropy.append(entropy)
        avg_state_entropy = np.mean(state_entropy)
        
        # Season balance
        season_entropy = []
        for fold in df['fold'].unique():
            fold_data = df[df['fold'] == fold]
            season_dist = fold_data['season'].value_counts(normalize=True)
            entropy = stats.entropy(season_dist)
            season_entropy.append(entropy)
        avg_season_entropy = np.mean(season_entropy)
        
        # Target variance (lower is better)
        target_cv = []
        for target in target_cols:
            fold_means = df.groupby('fold')[target].mean()
            cv = fold_means.std() / fold_means.mean() * 100 if fold_means.mean() > 0 else 0
            target_cv.append(cv)
        avg_target_cv = np.mean(target_cv)
        
        # Composite score (normalized)
        # Higher entropy = better diversity
        # Lower CV = better balance
        balance_score = (avg_state_entropy + avg_season_entropy) * 100 - (size_cv + avg_target_cv) / 2
        
        balance_scores.append({
            'Strategy': strategy_name,
            'Balance_Score': balance_score,
            'Size_CV': size_cv,
            'State_Entropy': avg_state_entropy,
            'Season_Entropy': avg_season_entropy,
            'Target_CV': avg_target_cv
        })
    
    balance_df = pd.DataFrame(balance_scores).sort_values('Balance_Score', ascending=False)
    
    bars = ax4.barh(balance_df['Strategy'], balance_df['Balance_Score'], 
                    color=sns.color_palette('RdYlGn', len(balance_df)))
    ax4.set_xlabel('Balance Score (Higher is Better)')
    ax4.set_title('Overall CV Strategy Balance Score\n(Composite Metric)')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, balance_df['Balance_Score'])):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '2_Variance_Analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return balance_df

def plot_detailed_target_distributions(results_dict, target_cols):
    """Detailed box plots for each target across folds"""
    print("\nGenerating detailed target distributions...")
    
    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    n_targets = len(target_cols)
    n_strategies = len(valid_results)
    
    if n_strategies == 0:
        print("  ⚠️  No valid strategies to plot")
        return
    
    fig, axes = plt.subplots(n_targets, n_strategies, 
                            figsize=(6 * n_strategies, 4 * n_targets))
    
    # Handle single strategy case
    if n_strategies == 1:
        axes = axes.reshape(-1, 1)
    elif n_targets == 1:
        axes = axes.reshape(1, -1)
    
    for col_idx, (strategy_name, df) in enumerate(valid_results.items()):
        for row_idx, target in enumerate(target_cols):
            ax = axes[row_idx, col_idx]
            
            # Box plot
            fold_data = [df[df['fold'] == fold][target].values 
                        for fold in sorted(df['fold'].unique())]
            
            bp = ax.boxplot(fold_data, labels=sorted(df['fold'].unique()),
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            # Overall mean line
            overall_mean = df[target].mean()
            ax.axhline(overall_mean, color='darkgreen', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Overall Mean')
            
            ax.set_xlabel('Fold')
            ax.set_ylabel(target.replace('_g', ' (g)'))
            ax.set_title(f'{strategy_name}\n{target.replace("_g", "")}')
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '3_Target_Distributions_Detailed.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results_dict, balance_df, target_cols):
    """Generate text summary report"""
    print("\n" + "="*80)
    print("CROSS-VALIDATION STRATEGY ANALYSIS REPORT")
    print("="*80)
    
    for strategy_name, df in results_dict.items():
        if df is None:
            print(f"\n❌ {strategy_name}: FAILED")
            continue
        
        print(f"\n{'='*80}")
        print(f"📊 {strategy_name}")
        print(f"{'='*80}")
        
        # Basic stats
        print(f"\n1. FOLD SIZE DISTRIBUTION:")
        fold_sizes = df['fold'].value_counts().sort_index()
        print(f"   Mean: {fold_sizes.mean():.1f} samples")
        print(f"   Std:  {fold_sizes.std():.1f} samples")
        print(f"   Min:  {fold_sizes.min()} samples")
        print(f"   Max:  {fold_sizes.max()} samples")
        print(f"   CV:   {(fold_sizes.std() / fold_sizes.mean() * 100):.2f}%")
        
        # State distribution
        print(f"\n2. STATE DISTRIBUTION:")
        for fold in sorted(df['fold'].unique()):
            fold_data = df[df['fold'] == fold]
            states = fold_data['State'].value_counts()
            print(f"   Fold {fold}: {dict(states)}")
        
        # Season distribution
        print(f"\n3. SEASON DISTRIBUTION:")
        for fold in sorted(df['fold'].unique()):
            fold_data = df[df['fold'] == fold]
            seasons = fold_data['season'].value_counts()
            print(f"   Fold {fold}: {dict(seasons)}")
        
        # Target statistics
        print(f"\n4. TARGET STATISTICS (Mean ± Std):")
        for target in target_cols:
            print(f"\n   {target}:")
            fold_means = df.groupby('fold')[target].mean()
            fold_stds = df.groupby('fold')[target].std()
            overall_mean = df[target].mean()
            overall_std = df[target].std()
            
            for fold in sorted(df['fold'].unique()):
                mean_val = fold_means[fold]
                std_val = fold_stds[fold]
                diff_pct = ((mean_val - overall_mean) / overall_mean * 100) if overall_mean > 0 else 0
                print(f"      Fold {fold}: {mean_val:6.1f} ± {std_val:5.1f} g  "
                      f"(Δ {diff_pct:+5.1f}% from overall)")
            
            cv = (fold_means.std() / fold_means.mean() * 100) if fold_means.mean() > 0 else 0
            print(f"      Fold Mean CV: {cv:.2f}%")
    
    # Rankings
    print(f"\n{'='*80}")
    print("🏆 STRATEGY RANKINGS (by Balance Score)")
    print(f"{'='*80}")
    
    for rank, row in balance_df.iterrows():
        print(f"\n{rank+1}. {row['Strategy']}")
        print(f"   Balance Score:    {row['Balance_Score']:.2f}")
        print(f"   Size CV:          {row['Size_CV']:.2f}%")
        print(f"   State Entropy:    {row['State_Entropy']:.3f}")
        print(f"   Season Entropy:   {row['Season_Entropy']:.3f}")
        print(f"   Avg Target CV:    {row['Target_CV']:.2f}%")
    
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    best_strategy = balance_df.iloc[0]
    print(f"\n🎯 Best Overall: {best_strategy['Strategy']}")
    print(f"   → Most balanced across size, diversity, and target consistency")
    
    # Specific recommendations
    print(f"\n📌 Specific Considerations:")
    print(f"   • For temporal generalization: Use Season-based GroupKFold")
    print(f"   • For geographic generalization: Use State-based GroupKFold")
    print(f"   • For species generalization: Use Species-based GroupKFold")
    print(f"   • For maximum data efficiency: Use Stratified KFold")
    print(f"   • For simplicity: Use standard KFold")
    
    print(f"\n⚠️  Important Notes:")
    print(f"   • With only 357 samples, any grouping will create imbalanced folds")
    print(f"   • Consider using fewer folds (3-4) for better fold sizes")
    print(f"   • GroupKFold ensures no leakage within groups but may sacrifice balance")
    print(f"   • StratifiedKFold provides better target balance but may split related samples")
    
    print(f"\n{'='*80}\n")

def main():
    # Load data
    df, target_cols = load_and_prepare_data('train.csv')
    
    print(f"\n{'='*80}")
    print(f"Dataset Overview:")
    print(f"  Total Samples: {len(df)}")
    print(f"  States: {df['State'].nunique()} ({list(df['State'].unique())})")
    print(f"  Seasons: {df['season'].nunique()} ({list(df['season'].unique())})")
    print(f"  Species: {df['Species'].nunique()}")
    print(f"  Top 5 Species: {list(df['Species'].value_counts().head().index)}")
    print(f"{'='*80}\n")
    
    # Define CV strategies
    strategies = {
        '1. Standard KFold': (
            KFold(n_splits=N_FOLDS, shuffle=True, random_state=42),
            None, None
        ),
        '2. Stratified KFold (Season)': (
            StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42),
            None, 'season'
        ),
        '3. Stratified KFold (Species)': (
            StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42),
            None, 'Species'
        ),
        '4. GroupKFold (Season)': (
            GroupKFold(n_splits=N_FOLDS),
            'season', None
        ),
        '5. GroupKFold (State)': (
            GroupKFold(n_splits=N_FOLDS),
            'State', None
        ),
        '6. GroupKFold (Species Grouped)': (
            GroupKFold(n_splits=N_FOLDS),
            'species_grouped', None
        ),
        '7. GroupKFold (Sample ID)': (
            GroupKFold(n_splits=N_FOLDS),
            'sample_id', None
        )
    }
    
    # Analyze each strategy
    results = {}
    for strategy_name, (splitter, group_col, stratify_col) in strategies.items():
        result_df = analyze_cv_strategy(df, strategy_name, splitter, group_col, stratify_col)
        results[strategy_name] = result_df
    
    # Generate visualizations
    plot_fold_distributions(results, target_cols)
    balance_df = plot_variance_analysis(results, target_cols)
    plot_detailed_target_distributions(results, target_cols)
    
    # Generate report
    generate_summary_report(results, balance_df, target_cols)
    
    print(f"\n✅ Analysis complete! Files saved to: {BASE_VIZ_PATH}/")
    print(f"   - 1_Fold_Distributions_Overview.png")
    print(f"   - 2_Variance_Analysis.png")
    print(f"   - 3_Target_Distributions_Detailed.png")

if __name__ == '__main__':
    main()