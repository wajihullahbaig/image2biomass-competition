"""
Exploratory Data Analysis for Dead Biomass Visibility
Analyzes the distribution of dead_hsv_score and its correlation with Dry_Dead_g target
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from common import load_data, get_hsv_biomass_scores
from feature_transform import apply_deterministic_features

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dead_visibility():
    """Analyze dead biomass visibility patterns in the dataset"""
    
    # Load data
    logger.info("Loading data...")
    df = load_data(logger)
    df = apply_deterministic_features(df)
    
    # Calculate HSV scores for all images
    logger.info("Calculating HSV scores for all images...")
    dead_hsv_scores = []
    green_hsv_scores = []
    dry_green_hsv_scores = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing image {idx}/{len(df)}")
        
        try:
            img_path = row['image_path']
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            scores = get_hsv_biomass_scores(img_np)
            dead_hsv_scores.append(scores['dead_score'])
            green_hsv_scores.append(scores['green_score'])
            dry_green_hsv_scores.append(scores['dry_green_score'])
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            dead_hsv_scores.append(0.0)
            green_hsv_scores.append(0.0)
            dry_green_hsv_scores.append(0.0)
    
    df['dead_hsv_score'] = dead_hsv_scores
    df['green_hsv_score'] = green_hsv_scores
    df['dry_green_hsv_score'] = dry_green_hsv_scores
    
    # Save augmented dataframe
    df.to_csv('analysis_results/data_with_hsv_scores.csv', index=False)
    logger.info("Saved augmented data to data_with_hsv_scores.csv")
    
    # Create analysis directory
    os.makedirs('analysis_results/dead_visibility_analysis', exist_ok=True)
    
    # ===== ANALYSIS 1: Distribution of dead_hsv_score =====
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 1: Dead HSV Score Distribution")
    logger.info("="*60)
    
    print(f"\nDead HSV Score Statistics:")
    print(f"  Mean: {df['dead_hsv_score'].mean():.4f}")
    print(f"  Median: {df['dead_hsv_score'].median():.4f}")
    print(f"  Std: {df['dead_hsv_score'].std():.4f}")
    print(f"  Min: {df['dead_hsv_score'].min():.4f}")
    print(f"  Max: {df['dead_hsv_score'].max():.4f}")
    
    print(f"\nVisibility Cohorts:")
    high_vis = (df['dead_hsv_score'] > 0.15).sum()
    med_vis = ((df['dead_hsv_score'] > 0.05) & (df['dead_hsv_score'] <= 0.15)).sum()
    low_vis = (df['dead_hsv_score'] <= 0.05).sum()
    
    print(f"  High Visibility (>15%): {high_vis} samples ({high_vis/len(df)*100:.1f}%)")
    print(f"  Medium Visibility (5-15%): {med_vis} samples ({med_vis/len(df)*100:.1f}%)")
    print(f"  Low Visibility (<5%): {low_vis} samples ({low_vis/len(df)*100:.1f}%)")
    
    # Plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(df['dead_hsv_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0.05, color='orange', linestyle='--', label='Low/Med threshold')
    axes[0, 0].axvline(0.15, color='red', linestyle='--', label='Med/High threshold')
    axes[0, 0].set_xlabel('Dead HSV Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Dead HSV Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Log-scale histogram
    axes[0, 1].hist(df['dead_hsv_score'], bins=50, edgecolor='black', alpha=0.7, log=True)
    axes[0, 1].axvline(0.05, color='orange', linestyle='--', label='Low/Med threshold')
    axes[0, 1].axvline(0.15, color='red', linestyle='--', label='Med/High threshold')
    axes[0, 1].set_xlabel('Dead HSV Score')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].set_title('Distribution (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Cumulative distribution
    sorted_scores = np.sort(df['dead_hsv_score'])
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 0].plot(sorted_scores, cumulative, linewidth=2)
    axes[1, 0].axvline(0.05, color='orange', linestyle='--', label='5% threshold')
    axes[1, 0].axvline(0.15, color='red', linestyle='--', label='15% threshold')
    axes[1, 0].set_xlabel('Dead HSV Score')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Box plot by season
    if 'Season' in df.columns:
        df.boxplot(column='dead_hsv_score', by='Season', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Season')
        axes[1, 1].set_ylabel('Dead HSV Score')
        axes[1, 1].set_title('Dead Visibility by Season')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_results/dead_visibility_analysis/dead_hsv_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: analysis_results/dead_visibility_analysis/dead_hsv_distribution.png")
    
    # ===== ANALYSIS 2: Correlation with Dry_Dead_g =====
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 2: Dead HSV Score vs Dry_Dead_g Correlation")
    logger.info("="*60)
    
    correlation = df[['dead_hsv_score', 'Dry_Dead_g']].corr().iloc[0, 1]
    print(f"\nPearson Correlation: {correlation:.4f}")
    
    # Scatter plot with regression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].scatter(df['dead_hsv_score'], df['Dry_Dead_g'], alpha=0.3, s=10)
    axes[0].set_xlabel('Dead HSV Score')
    axes[0].set_ylabel('Dry_Dead_g (grams)')
    axes[0].set_title(f'Dead HSV vs Actual Dead Biomass (r={correlation:.3f})')
    axes[0].grid(alpha=0.3)
    
    # Add regression line
    z = np.polyfit(df['dead_hsv_score'], df['Dry_Dead_g'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['dead_hsv_score'].min(), df['dead_hsv_score'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'y={z[0]:.1f}x+{z[1]:.1f}')
    axes[0].legend()
    
    # Log scale
    axes[1].scatter(df['dead_hsv_score'], np.log1p(df['Dry_Dead_g']), alpha=0.3, s=10)
    axes[1].set_xlabel('Dead HSV Score')
    axes[1].set_ylabel('log1p(Dry_Dead_g)')
    axes[1].set_title('Dead HSV vs Log Dead Biomass')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_results/dead_visibility_analysis/dead_correlation.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: analysis_results/dead_visibility_analysis/dead_correlation.png")
    
    # ===== ANALYSIS 3: Dead by Visibility Cohort =====
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 3: Dead Biomass by Visibility Cohort")
    logger.info("="*60)
    
    df['visibility_cohort'] = pd.cut(
        df['dead_hsv_score'], 
        bins=[-0.001, 0.05, 0.15, 1.0],
        labels=['Low (<5%)', 'Medium (5-15%)', 'High (>15%)']
    )
    
    cohort_stats = df.groupby('visibility_cohort')['Dry_Dead_g'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\nDead Biomass by Visibility Cohort:")
    print(cohort_stats)
    
    # Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Dry_Dead_g', by='visibility_cohort', ax=ax)
    ax.set_xlabel('Visibility Cohort')
    ax.set_ylabel('Dry_Dead_g (grams)')
    ax.set_title('Dead Biomass Distribution by Visibility')
    plt.sca(ax)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('analysis_results/dead_visibility_analysis/dead_by_cohort.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: analysis_results/dead_visibility_analysis/dead_by_cohort.png")
    
    # ===== ANALYSIS 4: Correlation Matrix =====
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 4: Correlation Matrix")
    logger.info("="*60)
    
    corr_cols = ['Dry_Dead_g', 'Dry_Green_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g',
                 'dead_hsv_score', 'green_hsv_score', 'dry_green_hsv_score',
                 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    corr_matrix = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Biomass Targets and HSV Scores')
    plt.tight_layout()
    plt.savefig('analysis_results/dead_visibility_analysis/correlation_matrix.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: analysis_results/dead_visibility_analysis/correlation_matrix.png")
    
    print("\nKey Correlations with Dry_Dead_g:")
    dead_corr = corr_matrix['Dry_Dead_g'].sort_values(ascending=False)
    print(dead_corr)
    
    # ===== ANALYSIS 5: Physics Constraint Validation =====
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 5: Physics Constraint Validation")
    logger.info("="*60)
    
    # Check: Total = Green + Dead + Clover
    df['total_computed'] = df['Dry_Green_g'] + df['Dry_Dead_g'] + df['Dry_Clover_g']
    df['total_error'] = df['Dry_Total_g'] - df['total_computed']
    df['total_error_pct'] = (df['total_error'] / (df['Dry_Total_g'] + 1e-6)) * 100
    
    print(f"\nPhysics Constraint: Total = Green + Dead + Clover")
    print(f"  Mean Error: {df['total_error'].mean():.2f}g")
    print(f"  Mean Error %: {df['total_error_pct'].mean():.2f}%")
    print(f"  Std Error: {df['total_error'].std():.2f}g")
    print(f"  Max Error: {df['total_error'].abs().max():.2f}g")
    
    violations = (df['total_error'].abs() > 5.0).sum()
    print(f"  Violations (>5g error): {violations} samples ({violations/len(df)*100:.1f}%)")
    
    # Check: GDM = Green + Clover
    df['gdm_computed'] = df['Dry_Green_g'] + df['Dry_Clover_g']
    df['gdm_error'] = df['GDM_g'] - df['gdm_computed']
    df['gdm_error_pct'] = (df['gdm_error'] / (df['GDM_g'] + 1e-6)) * 100
    
    print(f"\nPhysics Constraint: GDM = Green + Clover")
    print(f"  Mean Error: {df['gdm_error'].mean():.2f}g")
    print(f"  Mean Error %: {df['gdm_error_pct'].mean():.2f}%")
    print(f"  Std Error: {df['gdm_error'].std():.2f}g")
    print(f"  Max Error: {df['gdm_error'].abs().max():.2f}g")
    
    # ===== ANALYSIS 6: HSV Score Reliability =====
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 6: HSV Score as Minimum Dead Predictor")
    logger.info("="*60)
    
    # Hypothesis: dead_hsv_score * k should be <= Dry_Dead_g
    # Find optimal k
    k_values = np.arange(10, 200, 10)
    violation_rates = []
    
    for k in k_values:
        min_dead_from_hsv = df['dead_hsv_score'] * k
        violations = (df['Dry_Dead_g'] < min_dead_from_hsv).sum()
        violation_rates.append(violations / len(df) * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, violation_rates, marker='o', linewidth=2)
    ax.set_xlabel('Scaling Factor k (dead_hsv_score * k = min_dead_g)')
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('HSV-Based Minimum Dead Constraint Violation Rate')
    ax.grid(alpha=0.3)
    ax.axhline(5, color='red', linestyle='--', label='5% acceptable violation rate')
    ax.legend()
    plt.tight_layout()
    plt.savefig('analysis_results/dead_visibility_analysis/hsv_constraint_calibration.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: analysis_results/dead_visibility_analysis/hsv_constraint_calibration.png")
    
    # Find optimal k (5% violation rate)
    optimal_k_idx = np.argmin(np.abs(np.array(violation_rates) - 5.0))
    optimal_k = k_values[optimal_k_idx]
    print(f"\nOptimal Scaling Factor k: {optimal_k}")
    print(f"  Violation Rate at k={optimal_k}: {violation_rates[optimal_k_idx]:.2f}%")
    
    # ===== SUMMARY REPORT =====
    logger.info("\n" + "="*60)
    logger.info("SUMMARY REPORT")
    logger.info("="*60)
    
    summary = f"""
    DEAD BIOMASS VISIBILITY ANALYSIS SUMMARY
    =========================================
    
    Dataset Size: {len(df)} samples
    
    1. VISIBILITY DISTRIBUTION:
       - High Visibility (>15%): {high_vis} samples ({high_vis/len(df)*100:.1f}%)
       - Medium Visibility (5-15%): {med_vis} samples ({med_vis/len(df)*100:.1f}%)
       - Low Visibility (<5%): {low_vis} samples ({low_vis/len(df)*100:.1f}%)
       
    2. DEAD HSV SCORE STATISTICS:
       - Mean: {df['dead_hsv_score'].mean():.4f}
       - Median: {df['dead_hsv_score'].median():.4f}
       - Std: {df['dead_hsv_score'].std():.4f}
       
    3. CORRELATION WITH DRY_DEAD_G:
       - Pearson r: {correlation:.4f}
       - Interpretation: {'Strong' if abs(correlation) > 0.5 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} correlation
       
    4. DEAD BIOMASS BY COHORT:
       - Low Visibility: {cohort_stats.loc['Low (<5%)', 'mean']:.2f}g ± {cohort_stats.loc['Low (<5%)', 'std']:.2f}g
       - Medium Visibility: {cohort_stats.loc['Medium (5-15%)', 'mean']:.2f}g ± {cohort_stats.loc['Medium (5-15%)', 'std']:.2f}g
       - High Visibility: {cohort_stats.loc['High (>15%)', 'mean']:.2f}g ± {cohort_stats.loc['High (>15%)', 'std']:.2f}g
       
    5. PHYSICS CONSTRAINTS:
       - Total = Green + Dead + Clover: {violations} violations ({violations/len(df)*100:.1f}%)
       - Mean error: {df['total_error'].mean():.2f}g
       
    6. HSV CONSTRAINT RECOMMENDATION:
       - Optimal scaling factor k: {optimal_k}
       - Formula: min_dead_g = dead_hsv_score * {optimal_k}
       - Violation rate: {violation_rates[optimal_k_idx]:.2f}%
       
    RECOMMENDATIONS:
    ================
    {'1. PRIORITY: Implement dual-path prediction (visible vs hidden dead)' if low_vis > len(df) * 0.5 else '1. HSV-based constraint should work well (most samples have visible dead)'}
    2. Use HSV constraint with k={optimal_k} in loss function
    3. {'Consider visibility-stratified training (large low-vis cohort)' if low_vis > len(df) * 0.3 else 'Single model should suffice (small low-vis cohort)'}
    4. Correlation is {'strong enough for direct prediction' if abs(correlation) > 0.5 else 'moderate - derivation may help'}
    """
    
    print(summary)
    
    # Save summary
    with open('analysis_results/dead_visibility_analysis/SUMMARY.txt', 'w') as f:
        f.write(summary)
    
    logger.info("\n" + "="*60)
    logger.info("Analysis complete! Check analysis_results/dead_visibility_analysis/ for results")
    logger.info("="*60)

if __name__ == '__main__':
    analyze_dead_visibility()
