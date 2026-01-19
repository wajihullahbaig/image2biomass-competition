import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def visualize_data_splits(train_df, val_df, holdout_df, session_dir, fold=None, group_col='SessionID'):
    """
    Comprehensive visualization of data splits to detect leakage and understand stratification.
    
    Args:
        group_col: Column name used for grouping in the split (e.g. 'SessionID', 'State_Species')
    """
    plots_dir = os.path.join(session_dir, 'split_analysis')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Combine all data with split labels
    train_df = train_df.copy()
    val_df = val_df.copy() 
    holdout_df = holdout_df.copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    holdout_df['split'] = 'holdout'
    
    all_data = pd.concat([train_df, val_df, holdout_df], ignore_index=True)
    
    # Define consistent color mapping
    COLOR_MAP = {'train': 'green', 'validation': 'blue', 'holdout': 'red'}
    split_order = ['train', 'validation', 'holdout']
    plot_colors = [COLOR_MAP[s] for s in split_order]
    
    # 1. TEMPORAL LEAKAGE CHECK
    plt.figure(figsize=(24, 12))  # Increased size for better readability
    plt.style.use('default')  # Ensure clean styling
    
    # Convert sampling dates
    all_data['Sampling_Date'] = pd.to_datetime(all_data['Sampling_Date'])
    
    # Timeline plot
    plt.subplot(2, 4, 1)
    for split in split_order:
        color = COLOR_MAP[split]
        split_data = all_data[all_data['split'] == split]
        dates = split_data['Sampling_Date']
        y_vals = np.random.normal(0, 0.1, len(dates))  # Add jitter
        plt.scatter(dates, y_vals, alpha=0.7, label=f'{split} (n={len(dates)})', color=color, s=25)
    
    plt.title('Temporal Distribution of Splits', fontsize=12, fontweight='bold')
    plt.xlabel('Sampling Date', fontsize=10)
    plt.ylabel('Random Jitter', fontsize=10)
    plt.legend(fontsize=9)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 2. SPECIES × SEASON COVERAGE MATRIX (Training only)
    plt.subplot(2, 4, 2)
    if 'Species' in all_data.columns and 'Season' in all_data.columns:
        # Get training data coverage
        train_data = all_data[all_data['split'] == 'train']
        coverage_matrix = pd.crosstab(train_data['Species'], train_data['Season'])
        
        # Ensure all seasons are present (in order)
        season_order = ['summer', 'autumn', 'winter', 'spring']
        for s in season_order:
            if s not in coverage_matrix.columns:
                coverage_matrix[s] = 0
        coverage_matrix = coverage_matrix[season_order]
        
        # Create heatmap
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list('coverage', ['#ffcccc', '#66ff66', '#006600'])
        
        im = plt.imshow(coverage_matrix.values, cmap=cmap, aspect='auto')
        plt.colorbar(im, label='Train Samples')
        
        # Labels
        plt.xticks(range(len(coverage_matrix.columns)), coverage_matrix.columns, fontsize=9)
        plt.yticks(range(len(coverage_matrix.index)), 
                   [s[:12] + '..' if len(s) > 12 else s for s in coverage_matrix.index], 
                   fontsize=7)
        
        # Add count annotations
        for i in range(len(coverage_matrix.index)):
            for j in range(len(coverage_matrix.columns)):
                val = coverage_matrix.iloc[i, j]
                color = 'white' if val > coverage_matrix.values.max() * 0.6 else 'black'
                plt.text(j, i, str(val), ha='center', va='center', fontsize=7, color=color)
        
        # Count coverage gaps
        zero_cells = (coverage_matrix.values == 0).sum()
        total_cells = coverage_matrix.size
        coverage_pct = 100 * (total_cells - zero_cells) / total_cells
        
        plt.title(f'Species×Season Coverage (Train)\n{coverage_pct:.0f}% covered ({zero_cells} gaps)', 
                  fontsize=11, fontweight='bold')
        plt.xlabel('Season', fontsize=10)
        plt.ylabel('Species', fontsize=10)
    else:
        plt.text(0.5, 0.5, 'Species/Season columns not found', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=11, style='italic', color='red')
        plt.title('Coverage Matrix - Columns Missing', fontsize=12, fontweight='bold', color='red')
    
    # 3. SPECIES DISTRIBUTION
    plt.subplot(2, 4, 3)
    species_cross = pd.crosstab(all_data['Species'], all_data['split'])
    # Reorder columns to ensure correct coloring
    cols = [c for c in split_order if c in species_cross.columns]
    species_cross = species_cross[cols]
    current_colors = [COLOR_MAP[c] for c in cols]
    
    ax3 = species_cross.plot(kind='bar', ax=plt.gca(), color=current_colors, alpha=0.8)
    plt.title('Species Distribution Across Splits', fontsize=12, fontweight='bold')
    plt.xlabel('Species', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(rotation=45, fontsize=9, ha='right')
    plt.yticks(fontsize=9)
    plt.legend([s.capitalize() for s in cols], fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. STATE DISTRIBUTION
    plt.subplot(2, 4, 4)
    state_cross = pd.crosstab(all_data['State'], all_data['split'])
    cols = [c for c in split_order if c in state_cross.columns]
    state_cross = state_cross[cols]
    current_colors = [COLOR_MAP[c] for c in cols]
    
    ax4 = state_cross.plot(kind='bar', ax=plt.gca(), color=current_colors, alpha=0.8)
    plt.title('State Distribution Across Splits', fontsize=12, fontweight='bold')
    plt.xlabel('State', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(rotation=45, fontsize=9, ha='right')
    plt.yticks(fontsize=9)
    plt.legend([s.capitalize() for s in cols], fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. STRATIFICATION QUALITY (State_Species)
    plt.subplot(2, 4, 5)
    strat_key = 'State_Species'
    if strat_key in all_data.columns:
        strat_cross = pd.crosstab(all_data[strat_key], all_data['split'], margins=True)
        # Show top 15 most common groups
        top_groups = strat_cross.iloc[:-1, -1].sort_values(ascending=False).head(15)
        # Reorder columns to ensure correct coloring
        cols = [c for c in split_order if c in strat_cross.columns]
        strat_subset = strat_cross.loc[top_groups.index, cols]
        current_colors = [COLOR_MAP[c] for c in cols]
        
        ax5 = strat_subset.plot(kind='bar', ax=plt.gca(), color=current_colors, alpha=0.8)
        plt.title(f'Top 15 {strat_key} Groups', fontsize=12, fontweight='bold')
        plt.xlabel(strat_key, fontsize=10)
        plt.ylabel('Count', fontsize=10)
        # Truncate long labels for readability
        labels = [label.get_text()[:15] + '...' if len(label.get_text()) > 15 else label.get_text() 
                 for label in ax5.get_xticklabels()]
        plt.xticks(range(len(labels)), labels, rotation=45, fontsize=8, ha='right')
        plt.yticks(fontsize=9)
        plt.legend([s.capitalize() for s in cols], fontsize=9)
        plt.grid(True, alpha=0.3, axis='y')
    
    # 6. ALL 5 BIOMASS TARGETS DISTRIBUTION (log+1 scale)
    plt.subplot(2, 4, 6)
    biomass_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    available_cols = [c for c in biomass_cols if c in all_data.columns]
    
    if available_cols:
        # Plot training data only with different colors per target
        train_data = all_data[all_data['split'] == 'train']
        target_colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#3498db', '#f39c12']  # Green, Red, Purple, Blue, Orange
        labels = ['Green', 'Dead', 'Clover', 'GDM', 'Total']
        
        for i, (col, color, label) in enumerate(zip(available_cols, target_colors, labels)):
            if col in train_data.columns:
                values = np.log1p(train_data[col].fillna(0))
                plt.hist(values, bins=25, alpha=0.5, color=color, label=label, density=True)
        
        plt.title('All Targets Distribution (Train, log+1)', fontsize=11, fontweight='bold')
        plt.xlabel('log(biomass_g + 1)', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.legend(fontsize=8, loc='upper right')
        plt.grid(True, alpha=0.3)
    
    # 7. SEASON DISTRIBUTION
    plt.subplot(2, 4, 7)
    if 'Season' in all_data.columns:
        season_cross = pd.crosstab(all_data['Season'], all_data['split'])
        cols = [c for c in split_order if c in season_cross.columns]
        season_cross = season_cross[cols]
        current_colors = [COLOR_MAP[c] for c in cols]
        
        ax7 = season_cross.plot(kind='bar', ax=plt.gca(), color=current_colors, alpha=0.8)
        plt.title('Season Distribution Across Splits', fontsize=12, fontweight='bold')
        plt.xlabel('Season', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.xticks(rotation=45, fontsize=9, ha='right')
        plt.yticks(fontsize=9)
        plt.legend([s.capitalize() for s in cols], fontsize=9)
        plt.grid(True, alpha=0.3, axis='y')
    
    # 8. DATE RANGE SUMMARY
    plt.subplot(2, 4, 8)
    # Ensure splits are in order
    date_ranges = all_data.groupby('split')['Sampling_Date'].agg(['min', 'max'])
    date_ranges = date_ranges.reindex(split_order).dropna()
    
    # Create a timeline showing date ranges
    y_pos = range(len(date_ranges))
    
    for i, (split, row) in enumerate(date_ranges.iterrows()):
        color = COLOR_MAP[split]
        plt.barh(i, (row['max'] - row['min']).days, 
                left=row['min'], color=color, alpha=0.8, 
                label=f"{split}: {row['min'].strftime('%m/%d/%y')} to {row['max'].strftime('%m/%d/%y')}")
    
    plt.yticks(y_pos, date_ranges.index, fontsize=9)
    plt.title('Date Ranges by Split', fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=10)
    plt.xticks(rotation=45, fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Improve layout and spacing
    plt.tight_layout(pad=3.0)
    
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    plt.savefig(os.path.join(plots_dir, f'split_analysis{fold_suffix}.png'), 
               dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # NEW: Generate the 5-target distribution report requested by user
    visualize_multi_target_distributions(all_data, plots_dir, fold)
    
    generate_leakage_report(all_data, plots_dir, fold, group_col)

def visualize_multi_target_distributions(all_data, plots_dir, fold=None):
    """Generate a separate figure showing distributions for all 5 targets across splits."""
    biomass_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    available_cols = [c for c in biomass_cols if c in all_data.columns]
    
    if not available_cols:
        return

    # Create a 1x5 or 5x1 or 2x3 grid. 5x1 stack is good for vertical comparison.
    # Let's use 2x3 to keep it readable.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # User requested colors: train:green, validation:blue, holdout:red
    colors = {'train': 'green', 'validation': 'blue', 'holdout': 'red'}
    split_labels = {'train': 'Train', 'validation': 'Validation', 'holdout': 'Holdout'}
    
    for i, col in enumerate(available_cols):
        ax = axes[i]
        for split in ['train', 'validation', 'holdout']:
            split_data = all_data[all_data['split'] == split]
            if not split_data.empty:
                vals = np.log1p(split_data[col].fillna(0))
                ax.hist(vals, bins=25, alpha=0.5, color=colors[split], 
                        label=f"{split_labels[split]} (n={len(split_data)})", density=True)
        
        ax.set_title(f'Target: {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('log(grams + 1)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide the 6th empty subplot if only 5 columns
    if len(available_cols) < 6:
        axes[5].axis('off')

    fold_str = f"Fold {fold} " if fold is not None else ""
    plt.suptitle(f'{fold_str}Biomass Target Distributions (Log Scale)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    save_path = os.path.join(plots_dir, f'split_analysis{fold_suffix}_targets.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_leakage_report(all_data, plots_dir, fold=None, group_col='SessionID'):
    """Generate detailed text report of potential leakage issues.
    
    Args:
        group_col: Column name used for grouping in the split (e.g. 'SessionID', 'State_Species')
    """
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    report_path = os.path.join(plots_dir, f'leakage_report{fold_suffix}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DATA SPLIT LEAKAGE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # 1. Group leakage (using the specified grouping column)
        if group_col is None or group_col == 'None' or str(group_col).lower() == 'null':
            f.write(f"1. GROUP LEAKAGE ANALYSIS:\n")
            f.write(f"   Group-based splitting disabled (coverage-aware mode)\n")
            f.write(f"   ✓ Using coverage-first splitting strategy\n")
        elif group_col in all_data.columns:
            group_splits = all_data.groupby(group_col)['split'].apply(lambda x: sorted(x.unique())).reset_index()
            leaky_groups = group_splits[group_splits['split'].apply(len) > 1]
            
            f.write(f"1. GROUP LEAKAGE ({group_col}):\n")
            f.write(f"   Total {group_col}s: {len(group_splits)}\n")
            f.write(f"   {group_col}s with leakage: {len(leaky_groups)}\n")
            if len(leaky_groups) > 0:
                f.write(f"   ALERT: {group_col}s appear in multiple splits!\n")
                for _, row in leaky_groups.head(10).iterrows():
                    f.write(f"     {row[group_col]}: {row['split']}\n")
            else:
                f.write(f"   ✓ No {group_col} leakage detected\n")
        else:
            f.write(f"1. GROUP LEAKAGE ({group_col}):\n")
            f.write(f"   Note: Column '{group_col}' not in data (may be intentional for coverage-aware split)\n")
        f.write("\n")
        
        # 2. Temporal leakage
        f.write(f"2. TEMPORAL ANALYSIS:\n")
        date_ranges = all_data.groupby('split')['Sampling_Date'].agg(['min', 'max', 'count'])
        for split, row in date_ranges.iterrows():
            f.write(f"   {split.upper()}:\n")
            f.write(f"     Date range: {row['min'].strftime('%Y-%m-%d')} to {row['max'].strftime('%Y-%m-%d')}\n")
            f.write(f"     Sample count: {row['count']}\n")
        
        # Check for temporal overlap
        train_dates = set(all_data[all_data['split'] == 'train']['Sampling_Date'])
        val_dates = set(all_data[all_data['split'] == 'validation']['Sampling_Date']) 
        holdout_dates = set(all_data[all_data['split'] == 'holdout']['Sampling_Date'])
        
        overlap_train_val = train_dates & val_dates
        overlap_train_holdout = train_dates & holdout_dates
        overlap_val_holdout = val_dates & holdout_dates
        
        if overlap_train_val:
            f.write(f"   ALERT: {len(overlap_train_val)} dates overlap between train/validation\n")
        if overlap_train_holdout:
            f.write(f"   ALERT: {len(overlap_train_holdout)} dates overlap between train/holdout\n") 
        if overlap_val_holdout:
            f.write(f"   ALERT: {len(overlap_val_holdout)} dates overlap between validation/holdout\n")
        
        if not (overlap_train_val or overlap_train_holdout or overlap_val_holdout):
            f.write(f"   ✓ No temporal overlap detected\n")
        f.write("\n")
        
        # 3. Stratification quality
        f.write(f"3. STRATIFICATION QUALITY:\n")
        strat_key = 'State_Species'
        if strat_key in all_data.columns:
            strat_dist = all_data.groupby([strat_key, 'split']).size().unstack(fill_value=0)
            
            # Groups that appear in only one split
            single_split_groups = strat_dist[(strat_dist > 0).sum(axis=1) == 1]
            f.write(f"   Groups appearing in only one split: {len(single_split_groups)}\n")
            
            # Groups well represented across splits
            multi_split_groups = strat_dist[(strat_dist > 0).sum(axis=1) > 1]
            f.write(f"   Groups appearing in multiple splits: {len(multi_split_groups)}\n")
            
            if len(single_split_groups) > 0:
                f.write(f"   Groups with limited representation:\n")
                for group, row in single_split_groups.head(10).iterrows():
                    split_with_data = row[row > 0].index[0]
                    f.write(f"     {group}: only in {split_with_data} ({row[split_with_data]} samples)\n")
        f.write("\n")
        
        # 4. Summary
        f.write(f"4. SUMMARY:\n")
        total_samples = len(all_data)
        train_pct = len(all_data[all_data['split'] == 'train']) / total_samples * 100
        val_pct = len(all_data[all_data['split'] == 'validation']) / total_samples * 100
        holdout_pct = len(all_data[all_data['split'] == 'holdout']) / total_samples * 100
        
        f.write(f"   Total samples: {total_samples}\n")
        f.write(f"   Train: {train_pct:.1f}% ({len(all_data[all_data['split'] == 'train'])} samples)\n")
        f.write(f"   Validation: {val_pct:.1f}% ({len(all_data[all_data['split'] == 'validation'])} samples)\n") 
        f.write(f"   Holdout: {holdout_pct:.1f}% ({len(all_data[all_data['split'] == 'holdout'])} samples)\n")

def visualize_cross_validation_splits(fold_splits, full_df, session_dir):
    """
    Visualize all cross-validation folds to check consistency.
    fold_splits: list of (train_idx, val_idx) tuples from KFold
    """
    plots_dir = os.path.join(session_dir, 'cv_analysis')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 12))
    
    # Prepare date information
    full_df = full_df.copy()
    full_df['Sampling_Date'] = pd.to_datetime(full_df['Sampling_Date'])
    full_df['date_ordinal'] = full_df['Sampling_Date'].map(lambda x: x.toordinal())
    
    n_folds = len(fold_splits)
    
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        plt.subplot(n_folds, 3, fold*3 + 1)
        
        # Temporal distribution for this fold
        train_dates = full_df.iloc[train_idx]['Sampling_Date']
        val_dates = full_df.iloc[val_idx]['Sampling_Date']
        
        plt.hist(train_dates, bins=20, alpha=0.6, label=f'Train (n={len(train_idx)})', color='green')
        plt.hist(val_dates, bins=20, alpha=0.6, label=f'Val (n={len(val_idx)})', color='blue') 
        plt.title(f'Fold {fold+1}: Temporal Distribution')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(n_folds, 3, fold*3 + 2)
        
        # Species distribution for this fold
        train_species = full_df.iloc[train_idx]['Species'].value_counts()
        val_species = full_df.iloc[val_idx]['Species'].value_counts()
        
        all_species = sorted(set(train_species.index) | set(val_species.index))
        train_counts = [train_species.get(sp, 0) for sp in all_species]
        val_counts = [val_species.get(sp, 0) for sp in all_species]
        
        x = np.arange(len(all_species))
        width = 0.35
        
        plt.bar(x - width/2, train_counts, width, label='Train', color='green', alpha=0.7)
        plt.bar(x + width/2, val_counts, width, label='Val', color='blue', alpha=0.7)
        plt.title(f'Fold {fold+1}: Species Distribution')
        plt.xticks(x, all_species, rotation=45)
        plt.legend()
        
        plt.subplot(n_folds, 3, fold*3 + 3)
        
        # State distribution for this fold
        train_states = full_df.iloc[train_idx]['State'].value_counts()
        val_states = full_df.iloc[val_idx]['State'].value_counts()
        
        all_states = sorted(set(train_states.index) | set(val_states.index))
        train_state_counts = [train_states.get(st, 0) for st in all_states]
        val_state_counts = [val_states.get(st, 0) for st in all_states]
        
        x = np.arange(len(all_states))
        plt.bar(x - width/2, train_state_counts, width, label='Train', color='green', alpha=0.7)
        plt.bar(x + width/2, val_state_counts, width, label='Val', color='blue', alpha=0.7)
        plt.title(f'Fold {fold+1}: State Distribution')
        plt.xticks(x, all_states)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cv_folds_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

# Example usage function for integration into training pipeline
def analyze_splits_in_training(train_df, val_df, holdout_df, session_dir, fold=None, group_col='SessionID'):
    """
    Call this function after creating your train/val/holdout splits to analyze them.
    
    Args:
        group_col: Column name used for grouping in the split (e.g. 'SessionID', 'State_Species')
    """
    print(f"Generating split analysis visualizations (grouping by {group_col})...")
    visualize_data_splits(train_df, val_df, holdout_df, session_dir, fold, group_col)
    print(f"Split analysis saved to {session_dir}/split_analysis/")