import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = 'groupkfold_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target variables to analyze
TARGET_COLS = ['Dry_Total_g', 'Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g']

# Models to test
MODELS = {
    'Ridge': Ridge(alpha=1.0),
    'Ridge_Strong': Ridge(alpha=10.0),
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
}

# Number of folds
N_SPLITS = 5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def compute_metrics(y_true, y_pred):
    """Compute comprehensive regression metrics"""
    # Handle zero/negative predictions for MAPE
    mask = (y_true > 0) & (y_pred > 0)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if mask.sum() > 0 else np.nan,
        'Pearson_R': pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan,
        'Mean_Pred': np.mean(y_pred),
        'Std_Pred': np.std(y_pred),
        'Mean_True': np.mean(y_true),
        'Std_True': np.std(y_true)
    }
    return metrics

def analyze_group_distribution(df, group_col):
    """Analyze distribution of samples across groups"""
    group_counts = df[group_col].value_counts()
    
    stats = {
        'n_groups': df[group_col].nunique(),
        'min_samples': group_counts.min(),
        'max_samples': group_counts.max(),
        'mean_samples': group_counts.mean(),
        'median_samples': group_counts.median(),
        'std_samples': group_counts.std()
    }
    
    return stats, group_counts

def create_composite_group(df, cols):
    """Create composite grouping column"""
    return df[cols].astype(str).apply(lambda x: '_'.join(x), axis=1)

# ============================================================================
# MAIN ANALYSIS CLASS
# ============================================================================
class GroupKFoldAnalyzer:
    def __init__(self, df, target_cols, feature_cols, group_configs, n_splits=5):
        self.df = df.copy()
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.group_configs = group_configs
        self.n_splits = n_splits
        self.results = {}
        
    def run_analysis(self):
        """Run complete GroupKFold analysis for all configurations"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPREHENSIVE GROUP K-FOLD ANALYSIS")
        print("="*80)
        
        for config_name, group_cols in self.group_configs.items():
            print(f"\n{'='*80}")
            print(f"📊 Analyzing: {config_name}")
            print(f"   Grouping by: {group_cols}")
            print(f"{'='*80}")
            
            # Create composite group if multiple columns
            if len(group_cols) == 1:
                group_col = group_cols[0]
                groups = self.df[group_col].values
            else:
                group_col = '_'.join(group_cols)
                groups = create_composite_group(self.df, group_cols)
                self.df[group_col] = groups
                groups = groups.values
            
            # Analyze group distribution
            stats, counts = analyze_group_distribution(self.df, group_col)
            print(f"\n📈 Group Distribution Statistics:")
            print(f"   • Number of unique groups: {stats['n_groups']}")
            print(f"   • Samples per group - Min: {stats['min_samples']}, Max: {stats['max_samples']}")
            print(f"   • Samples per group - Mean: {stats['mean_samples']:.1f}, Median: {stats['median_samples']:.1f}")
            print(f"   • Samples per group - Std: {stats['std_samples']:.2f}")
            
            # Check if we have enough groups for k-fold
            if stats['n_groups'] < self.n_splits:
                print(f"\n⚠️  WARNING: Only {stats['n_groups']} groups available, reducing splits to {stats['n_groups']}")
                n_splits = stats['n_groups']
            else:
                n_splits = self.n_splits
            
            # Run cross-validation
            config_results = self.cross_validate(groups, n_splits, config_name, group_col)
            self.results[config_name] = {
                'cv_results': config_results,
                'group_stats': stats,
                'group_counts': counts,
                'group_col': group_col
            }
            
        return self.results
    
    def cross_validate(self, groups, n_splits, config_name, group_col):
        """Perform grouped cross-validation"""
        results = {target: {model_name: [] for model_name in MODELS.keys()} 
                  for target in self.target_cols}
        
        fold_details = []
        
        gkf = GroupKFold(n_splits=n_splits)
        
        # Prepare features
        X = self.df[self.feature_cols].fillna(0).values
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
            print(f"\n   Fold {fold_idx + 1}/{n_splits}:")
            print(f"      Train size: {len(train_idx)}, Val size: {len(val_idx)}")
            
            # Track which groups are in train vs validation
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            overlap = train_groups & val_groups
            
            print(f"      Train groups: {len(train_groups)}, Val groups: {len(val_groups)}")
            print(f"      Group overlap: {len(overlap)} (should be 0!)")
            
            if len(overlap) > 0:
                print(f"      ⚠️  WARNING: Group leakage detected! Overlapping groups: {overlap}")
            
            fold_info = {
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_groups': len(train_groups),
                'val_groups': len(val_groups),
                'overlap': len(overlap)
            }
            
            X_train, X_val = X[train_idx], X[val_idx]
            
            # Train and evaluate for each target
            for target in self.target_cols:
                y = self.df[target].values
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train each model
                for model_name, model in MODELS.items():
                    try:
                        # Clone model for this fold
                        from sklearn.base import clone
                        model_clone = clone(model)
                        
                        # Train
                        model_clone.fit(X_train, y_train)
                        
                        # Predict
                        y_pred = model_clone.predict(X_val)
                        
                        # Clip negative predictions to 0
                        y_pred = np.clip(y_pred, 0, None)
                        
                        # Compute metrics
                        metrics = compute_metrics(y_val, y_pred)
                        results[target][model_name].append(metrics)
                        
                    except Exception as e:
                        print(f"      ⚠️  Error with {model_name} on {target}: {str(e)}")
                        results[target][model_name].append(None)
            
            fold_details.append(fold_info)
        
        return {
            'results': results,
            'fold_details': fold_details
        }
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("📋 GENERATING COMPREHENSIVE SUMMARY REPORT")
        print("="*80)
        
        summary_data = []
        
        for config_name, config_data in self.results.items():
            cv_results = config_data['cv_results']['results']
            group_stats = config_data['group_stats']
            fold_details = config_data['cv_results']['fold_details']
            
            # Calculate average metrics across folds
            for target in self.target_cols:
                for model_name in MODELS.keys():
                    fold_metrics = cv_results[target][model_name]
                    
                    # Filter out None values (failed folds)
                    valid_metrics = [m for m in fold_metrics if m is not None]
                    
                    if len(valid_metrics) > 0:
                        # Aggregate metrics
                        avg_metrics = {
                            'Config': config_name,
                            'Target': target.replace('_g', ''),
                            'Model': model_name,
                            'RMSE_mean': np.mean([m['RMSE'] for m in valid_metrics]),
                            'RMSE_std': np.std([m['RMSE'] for m in valid_metrics]),
                            'MAE_mean': np.mean([m['MAE'] for m in valid_metrics]),
                            'MAE_std': np.std([m['MAE'] for m in valid_metrics]),
                            'R2_mean': np.mean([m['R2'] for m in valid_metrics]),
                            'R2_std': np.std([m['R2'] for m in valid_metrics]),
                            'MAPE_mean': np.nanmean([m['MAPE'] for m in valid_metrics]),
                            'Pearson_mean': np.mean([m['Pearson_R'] for m in valid_metrics]),
                            'N_Groups': group_stats['n_groups'],
                            'Samples_per_Group_mean': group_stats['mean_samples'],
                            'N_Folds': len(valid_metrics),
                            'Group_Overlap_mean': np.mean([f['overlap'] for f in fold_details])
                        }
                        summary_data.append(avg_metrics)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = os.path.join(OUTPUT_DIR, 'summary_report.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary saved to: {summary_path}")
        
        return summary_df
    
    def visualize_results(self, summary_df):
        """Create comprehensive visualizations"""
        print("\n📊 Generating visualizations...")
        
        # 1. Overall Performance Comparison
        self._plot_overall_comparison(summary_df)
        
        # 2. Per-Target Performance
        self._plot_per_target_performance(summary_df)
        
        # 3. Model Comparison
        self._plot_model_comparison(summary_df)
        
        # 4. Group Distribution Analysis
        self._plot_group_distributions()
        
        # 5. Stability Analysis (std across folds)
        self._plot_stability_analysis(summary_df)
        
        print(f"✅ All visualizations saved to: {OUTPUT_DIR}/")
    
    def _plot_overall_comparison(self, summary_df):
        """Plot overall R2 comparison across configurations"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Average across all targets and models
        config_avg = summary_df.groupby('Config').agg({
            'R2_mean': 'mean',
            'RMSE_mean': 'mean'
        }).reset_index()
        
        config_avg = config_avg.sort_values('R2_mean', ascending=False)
        
        # R2 comparison
        colors = ['#2ecc71' if r2 > 0.5 else '#e74c3c' if r2 < 0.3 else '#f39c12' 
                 for r2 in config_avg['R2_mean']]
        
        axes[0].barh(config_avg['Config'], config_avg['R2_mean'], color=colors, alpha=0.8)
        axes[0].set_xlabel('Mean R² Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Overall Performance by Configuration\n(Averaged across all targets & models)', 
                         fontsize=14, fontweight='bold')
        axes[0].axvline(0.5, color='green', linestyle='--', alpha=0.5, label='Good (0.5)')
        axes[0].axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='Fair (0.3)')
        axes[0].legend()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(config_avg.iterrows()):
            axes[0].text(row['R2_mean'] + 0.01, i, f"{row['R2_mean']:.3f}", 
                        va='center', fontweight='bold')
        
        # RMSE comparison
        axes[1].barh(config_avg['Config'], config_avg['RMSE_mean'], color='steelblue', alpha=0.8)
        axes[1].set_xlabel('Mean RMSE', fontsize=12, fontweight='bold')
        axes[1].set_title('Mean RMSE by Configuration\n(Lower is better)', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(config_avg.iterrows()):
            axes[1].text(row['RMSE_mean'] + 0.5, i, f"{row['RMSE_mean']:.2f}", 
                        va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '1_overall_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_target_performance(self, summary_df):
        """Plot performance for each target separately"""
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        axes = axes.flatten()
        
        for idx, target in enumerate(self.target_cols):
            target_data = summary_df[summary_df['Target'] == target.replace('_g', '')]
            
            if len(target_data) == 0:
                continue
            
            # Group by config and take best model
            best_by_config = target_data.loc[target_data.groupby('Config')['R2_mean'].idxmax()]
            best_by_config = best_by_config.sort_values('R2_mean', ascending=True)
            
            colors = ['#2ecc71' if r2 > 0.5 else '#e74c3c' if r2 < 0.3 else '#f39c12' 
                     for r2 in best_by_config['R2_mean']]
            
            axes[idx].barh(best_by_config['Config'], best_by_config['R2_mean'], 
                          color=colors, alpha=0.8)
            axes[idx].errorbar(best_by_config['R2_mean'], range(len(best_by_config)), 
                              xerr=best_by_config['R2_std'], fmt='none', 
                              ecolor='black', capsize=5, alpha=0.5)
            
            axes[idx].set_xlabel('R² Score', fontsize=10)
            axes[idx].set_title(f'{target}\n(Best model per config)', fontsize=11, fontweight='bold')
            axes[idx].axvline(0.5, color='green', linestyle='--', alpha=0.3)
            axes[idx].axvline(0.3, color='orange', linestyle='--', alpha=0.3)
            axes[idx].grid(axis='x', alpha=0.3)
            
            # Add model names and values
            for i, (_, row) in enumerate(best_by_config.iterrows()):
                axes[idx].text(row['R2_mean'] + 0.02, i, 
                             f"{row['Model'][:6]}: {row['R2_mean']:.3f}±{row['R2_std']:.3f}", 
                             va='center', fontsize=8)
        
        # Remove extra subplot
        if len(self.target_cols) < 6:
            fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_per_target_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, summary_df):
        """Compare models across configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['R2_mean', 'RMSE_mean', 'MAE_mean', 'MAPE_mean']
        titles = ['R² Score (Higher is better)', 'RMSE (Lower is better)', 
                 'MAE (Lower is better)', 'MAPE (Lower is better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            pivot_data = summary_df.pivot_table(
                values=metric,
                index='Config',
                columns='Model',
                aggfunc='mean'
            )
            
            pivot_data.plot(kind='barh', ax=axes[idx], width=0.8)
            axes[idx].set_xlabel(metric.replace('_mean', ''), fontsize=11)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_group_distributions(self):
        """Visualize group distributions for each configuration"""
        n_configs = len(self.results)
        fig, axes = plt.subplots(n_configs, 2, figsize=(16, 5*n_configs))
        
        if n_configs == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (config_name, config_data) in enumerate(self.results.items()):
            group_counts = config_data['group_counts']
            group_stats = config_data['group_stats']
            
            # Histogram
            axes[idx, 0].hist(group_counts.values, bins=20, color='steelblue', 
                             alpha=0.7, edgecolor='black')
            axes[idx, 0].axvline(group_stats['mean_samples'], color='red', 
                                linestyle='--', linewidth=2, label=f"Mean: {group_stats['mean_samples']:.1f}")
            axes[idx, 0].axvline(group_stats['median_samples'], color='green', 
                                linestyle='--', linewidth=2, label=f"Median: {group_stats['median_samples']:.1f}")
            axes[idx, 0].set_xlabel('Samples per Group', fontsize=11)
            axes[idx, 0].set_ylabel('Frequency', fontsize=11)
            axes[idx, 0].set_title(f'{config_name}: Group Size Distribution', fontsize=12, fontweight='bold')
            axes[idx, 0].legend()
            axes[idx, 0].grid(alpha=0.3)
            
            # Top groups bar chart
            top_10 = group_counts.head(10).sort_values(ascending=True)
            axes[idx, 1].barh(range(len(top_10)), top_10.values, color='coral', alpha=0.8)
            axes[idx, 1].set_yticks(range(len(top_10)))
            axes[idx, 1].set_yticklabels(top_10.index, fontsize=9)
            axes[idx, 1].set_xlabel('Number of Samples', fontsize=11)
            axes[idx, 1].set_title(f'{config_name}: Top 10 Groups by Sample Count', fontsize=12, fontweight='bold')
            axes[idx, 1].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, val in enumerate(top_10.values):
                axes[idx, 1].text(val + 0.5, i, str(int(val)), va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_group_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_analysis(self, summary_df):
        """Analyze stability (consistency across folds)"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # R2 stability (lower std is better)
        stability_data = summary_df.groupby('Config').agg({
            'R2_std': 'mean',
            'RMSE_std': 'mean'
        }).reset_index()
        
        stability_data = stability_data.sort_values('R2_std', ascending=True)
        
        colors = ['#2ecc71' if std < 0.1 else '#e74c3c' if std > 0.2 else '#f39c12' 
                 for std in stability_data['R2_std']]
        
        axes[0].barh(stability_data['Config'], stability_data['R2_std'], 
                    color=colors, alpha=0.8)
        axes[0].set_xlabel('R² Standard Deviation Across Folds', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Stability: R² Variance\n(Lower = More Consistent)', 
                         fontsize=14, fontweight='bold')
        axes[0].axvline(0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
        axes[0].axvline(0.2, color='orange', linestyle='--', alpha=0.5, label='Fair (0.2)')
        axes[0].legend()
        axes[0].grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(stability_data.iterrows()):
            axes[0].text(row['R2_std'] + 0.005, i, f"{row['R2_std']:.3f}", 
                        va='center', fontweight='bold', fontsize=9)
        
        # RMSE stability
        stability_data_rmse = stability_data.sort_values('RMSE_std', ascending=True)
        
        axes[1].barh(stability_data_rmse['Config'], stability_data_rmse['RMSE_std'], 
                    color='steelblue', alpha=0.8)
        axes[1].set_xlabel('RMSE Standard Deviation Across Folds', fontsize=12, fontweight='bold')
        axes[1].set_title('Model Stability: RMSE Variance\n(Lower = More Consistent)', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(stability_data_rmse.iterrows()):
            axes[1].text(row['RMSE_std'] + 0.2, i, f"{row['RMSE_std']:.2f}", 
                        va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_stability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_recommendations(self, summary_df):
        """Generate recommendations based on analysis"""
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS FOR TRAINING REGIME SELECTION")
        print("="*80)
        
        recommendations = []
        
        # 1. Best overall configuration
        config_avg = summary_df.groupby('Config').agg({
            'R2_mean': 'mean',
            'R2_std': 'mean',
            'RMSE_mean': 'mean'
        }).reset_index()
        
        best_config = config_avg.loc[config_avg['R2_mean'].idxmax()]
        print(f"\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"   Configuration: {best_config['Config']}")
        print(f"   Mean R²: {best_config['R2_mean']:.4f}")
        print(f"   R² Stability (std): {best_config['R2_std']:.4f}")
        print(f"   Mean RMSE: {best_config['RMSE_mean']:.4f}")
        
        recommendations.append({
            'Category': 'Overall Best',
            'Configuration': best_config['Config'],
            'R2_mean': best_config['R2_mean'],
            'R2_std': best_config['R2_std'],
            'Reason': 'Highest average R² across all targets'
        })
        
        # 2. Most stable configuration
        most_stable = config_avg.loc[config_avg['R2_std'].idxmin()]
        print(f"\n🎯 MOST STABLE CONFIGURATION:")
        print(f"   Configuration: {most_stable['Config']}")
        print(f"   Mean R²: {most_stable['R2_mean']:.4f}")
        print(f"   R² Stability (std): {most_stable['R2_std']:.4f} ⭐ LOWEST VARIANCE")
        print(f"   Mean RMSE: {most_stable['RMSE_mean']:.4f}")
        
        recommendations.append({
            'Category': 'Most Stable',
            'Configuration': most_stable['Config'],
            'R2_mean': most_stable['R2_mean'],
            'R2_std': most_stable['R2_std'],
            'Reason': 'Lowest variance across folds - most consistent'
        })
        
        # 3. Best per target
        print(f"\n🎯 BEST CONFIGURATION PER TARGET:")
        for target in self.target_cols:
            target_data = summary_df[summary_df['Target'] == target.replace('_g', '')]
            if len(target_data) > 0:
                best_for_target = target_data.loc[target_data['R2_mean'].idxmax()]
                print(f"\n   {target}:")
                print(f"      Configuration: {best_for_target['Config']}")
                print(f"      Model: {best_for_target['Model']}")
                print(f"      R²: {best_for_target['R2_mean']:.4f} ± {best_for_target['R2_std']:.4f}")
                print(f"      RMSE: {best_for_target['RMSE_mean']:.4f} ± {best_for_target['RMSE_std']:.4f}")
                
                recommendations.append({
                    'Category': f'Best for {target}',
                    'Configuration': best_for_target['Config'],
                    'Model': best_for_target['Model'],
                    'R2_mean': best_for_target['R2_mean'],
                    'R2_std': best_for_target['R2_std'],
                    'Reason': f'Best R² for {target}'
                })
        
        # 4. Group leakage check
        print(f"\n🔒 GROUP LEAKAGE ANALYSIS:")
        for config_name, config_data in self.results.items():
            fold_details = config_data['cv_results']['fold_details']
            total_overlap = sum([f['overlap'] for f in fold_details])
            print(f"   {config_name}: Total group overlaps = {total_overlap}")
            if total_overlap > 0:
                print(f"      ⚠️  WARNING: Leakage detected!")
            else:
                print(f"      ✅ No leakage - perfect group separation")
        
        # 5. Sample efficiency
        print(f"\n📊 SAMPLE EFFICIENCY (Considering group sizes):")
        for config_name, config_data in self.results.items():
            group_stats = config_data['group_stats']
            config_perf = config_avg[config_avg['Config'] == config_name].iloc[0]
            
            efficiency = config_perf['R2_mean'] / group_stats['n_groups']
            print(f"   {config_name}:")
            print(f"      Groups: {group_stats['n_groups']}")
            print(f"      Avg samples/group: {group_stats['mean_samples']:.1f}")
            print(f"      R² per group: {efficiency:.4f}")
        
        # Save recommendations
        rec_df = pd.DataFrame(recommendations)
        rec_path = os.path.join(OUTPUT_DIR, 'recommendations.csv')
        rec_df.to_csv(rec_path, index=False)
        print(f"\n✅ Recommendations saved to: {rec_path}")
        
        print("\n" + "="*80)
        print("📝 SELECTION CRITERIA GUIDE:")
        print("="*80)
        print("""
        Choose configuration based on your priority:
        
        1️⃣  MAXIMUM PERFORMANCE → Use "Best Overall Configuration"
           - Highest R² across all targets
           - May have higher variance across folds
        
        2️⃣  CONSISTENCY/RELIABILITY → Use "Most Stable Configuration"  
           - Lower variance = more reliable estimates
           - Predictions will be more consistent
        
        3️⃣  SPECIFIC TARGET → Use "Best for [Target]" configuration
           - Optimized for one particular prediction task
        
        4️⃣  PREVENTING OVERFITTING → Prefer configurations with:
           - More groups (better generalization)
           - Lower group overlap (no leakage)
           - More balanced group sizes
        
        5️⃣  SMALL DATASET (357 samples) → Consider:
           - State_Species: Good balance (fewer groups, more samples/group)
           - Avoid too granular grouping (e.g., State_Sampling_Date might be too specific)
        """)
        print("="*80)
        
        return rec_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("🌾 PASTURE BIOMASS: GROUP K-FOLD CROSS-VALIDATION ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('./wide.csv')
    print(f"   Loaded {len(df)} samples")
    
    # Define features
    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Height_Ave_cm_log']
    
    # Add interaction features
    df['NDVI_Height_Interaction'] = df['Pre_GSHH_NDVI'] * df['Height_Ave_cm']
    df['NDVI_squared'] = df['Pre_GSHH_NDVI'] ** 2
    feature_cols.extend(['NDVI_Height_Interaction', 'NDVI_squared'])
    
    print(f"   Using {len(feature_cols)} features: {feature_cols}")
    
    # Define grouping configurations
    group_configs = {
        'State_Species': ['State', 'Species'],
        'State_Sampling_Date': ['State', 'Sampling_Date'],
        'State_Season': ['State', 'season']
    }
    
    # Initialize analyzer
    analyzer = GroupKFoldAnalyzer(
        df=df,
        target_cols=TARGET_COLS,
        feature_cols=feature_cols,
        group_configs=group_configs,
        n_splits=N_SPLITS
    )
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Generate summary
    summary_df = analyzer.generate_summary_report()
    
    # Create visualizations
    analyzer.visualize_results(summary_df)
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations(summary_df)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
    print(f"\n📊 Generated files:")
    print(f"   • summary_report.csv - Detailed metrics for all configurations")
    print(f"   • recommendations.csv - Actionable recommendations")
    print(f"   • 1_overall_comparison.png - Overall performance comparison")
    print(f"   • 2_per_target_performance.png - Per-target analysis")
    print(f"   • 3_model_comparison.png - Model comparison across configs")
    print(f"   • 4_group_distributions.png - Group size distributions")
    print(f"   • 5_stability_analysis.png - Fold-to-fold consistency")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()