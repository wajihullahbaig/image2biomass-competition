import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
BASE_VIZ_PATH = 'visualizations'
STAGE1_VIZ_PATH = os.path.join(BASE_VIZ_PATH, 'stage1')
STAGE2_VIZ_PATH = os.path.join(BASE_VIZ_PATH, 'stage2')

for path in [BASE_VIZ_PATH, STAGE1_VIZ_PATH, STAGE2_VIZ_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- Helper Functions ---

def get_season(month):
    """Maps month to Australian seasons."""
    if month in [12, 1, 2]: return 'Summer'
    elif month in [3, 4, 5]: return 'Autumn'
    elif month in [6, 7, 8]: return 'Winter'
    else: return 'Spring'

def prepare_stage1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares Stage 1 features (raw features used for Stage 1 model).
    """
    print("Preparing Stage 1 features...")
    wide_df = df.pivot_table(
        index=['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    # Date features
    wide_df['Sampling_Date'] = pd.to_datetime(wide_df['Sampling_Date'])
    wide_df['month'] = wide_df['Sampling_Date'].dt.month
    wide_df['season'] = wide_df['month'].apply(get_season)
    period = 12
    wide_df['month_sin'] = np.sin(2 * np.pi * wide_df['month'] / period)
    wide_df['month_cos'] = np.cos(2 * np.pi * wide_df['month'] / period)
    
    # Impute
    for col in ['Pre_GSHH_NDVI', 'Height_Ave_cm']:
        if wide_df[col].isnull().any():
            median_val = wide_df[col].median()
            wide_df[col] = wide_df[col].fillna(median_val)
    
    return wide_df.drop(columns=['Sampling_Date'])

def prepare_stage2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares Stage 2 features (includes all engineered features).
    """
    print("Preparing Stage 2 features...")
    wide_df = prepare_stage1_features(df)
    
    # Log transform height (matching stage2.py which overwrites Height_Ave_cm)
    wide_df['Height_Ave_cm'] = np.log1p(wide_df['Height_Ave_cm'])
    
    # Interaction features (using log-transformed height)
    wide_df['NDVI_Height_MUL'] = wide_df['Pre_GSHH_NDVI'] * wide_df['Height_Ave_cm']
    wide_df['NDVI_Height_ADD'] = wide_df['Pre_GSHH_NDVI'] + wide_df['Height_Ave_cm']
    wide_df['NDVI_Height_Ratio'] = wide_df['Pre_GSHH_NDVI'] / (wide_df['Height_Ave_cm'] + 1e-5)
    
    # Count and frequency features
    species_counts = wide_df['Species'].value_counts()
    wide_df['species_count_global'] = wide_df['Species'].map(np.log1p(species_counts))
    wide_df['species_freq_global'] = wide_df['Species'].map(species_counts / len(wide_df))
    
    seasonal_counts = wide_df.groupby(['season', 'Species']).size()
    wide_df['species_count_seasonal'] = wide_df.apply(
        lambda row: np.log1p(seasonal_counts.get((row['season'], row['Species']), 0)), axis=1
    )
    
    return wide_df

# --- Main EDA Script ---
if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE EDA - STAGE-SPECIFIC FEATURE ANALYSIS")
    print("="*80)
    
    # Load data
    try:
        df_long = pd.read_csv('train.csv')
        print(f"Loaded train.csv with {len(df_long)} rows.")
    except FileNotFoundError:
        print("Error: train.csv not found.")
        exit()

    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # ========================================================================
    # COMMON VISUALIZATIONS (Base Folder)
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING COMMON VISUALIZATIONS")
    print("="*80)
    
    df_common = prepare_stage1_features(df_long)
    pbar = tqdm(total=4, desc="Common plots")
    
    # Plot 1: Target Distributions (Log and Real Scale)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Target Variable Distributions', fontsize=20)
    for i, target in enumerate(target_cols):
        sns.histplot(df_common[target], ax=axes[0, i], kde=True, bins=30)
        axes[0, i].set_title(f'{target} (Real Scale)')
        sns.histplot(np.log1p(df_common[target]), ax=axes[1, i], kde=True, bins=30, color='green')
        axes[1, i].set_title(f'{target} (Log1p Scale)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(BASE_VIZ_PATH, '1_target_distributions.png'))
    plt.close()
    pbar.update(1)

    # Plot 2: Categorical Features vs. Main Target
    train_df_common, _ = train_test_split(df_common, test_size=0.2, random_state=42, stratify=df_common['season'])
    cat_features = ['State', 'Species', 'season']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Categorical Features vs. Dry_Total_g (log1p scale)', fontsize=20)
    for i, feature in enumerate(cat_features):
        order = train_df_common[feature].value_counts().index
        sns.violinplot(x=feature, y=np.log1p(train_df_common['Dry_Total_g']), data=train_df_common, 
                      ax=axes[i], order=order, palette='viridis', cut=0)
        axes[i].set_title(f'Dry_Total_g by {feature}')
        if feature == 'Species':
            axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(BASE_VIZ_PATH, '2_categorical_vs_target.png'))
    plt.close()
    pbar.update(1)

    # Plot 3: Samples per State
    plt.figure(figsize=(10, 6))
    sns.countplot(y='State', data=df_common, order=df_common['State'].value_counts().index, palette='crest')
    plt.title('Number of Samples per State', fontsize=16)
    plt.xlabel('Count')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '3_samples_per_state.png'))
    plt.close()
    pbar.update(1)

    # Plot 4: Mean Total Biomass by Month - FIXED ALIGNMENT
    monthly_biomass = df_common.groupby('month')['Dry_Total_g'].mean().reset_index()
    monthly_biomass['month_idx'] = monthly_biomass['month'] - 1  # 0-indexed for plotting
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='month_idx', y='Dry_Total_g', data=monthly_biomass, marker='o', lw=2)
    sns.barplot(x='month_idx', y='Dry_Total_g', data=monthly_biomass, alpha=0.5, palette='coolwarm')
    plt.title('Mean Dry_Total_g by Month', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Mean Dry_Total_g')
    plt.xticks(ticks=range(12), labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_VIZ_PATH, '4_mean_biomass_by_month.png'))
    plt.close()
    pbar.update(1)
    pbar.close()
    
    # ========================================================================
    # STAGE 1 VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING STAGE 1 VISUALIZATIONS")
    print("="*80)
    
    df_stage1 = prepare_stage1_features(df_long)
    train_s1, val_s1 = train_test_split(df_stage1, test_size=0.2, random_state=42, stratify=df_stage1['season'])
    
    pbar = tqdm(total=3, desc="Stage 1 plots")
    
    # Stage 1 features (raw features)
    stage1_features = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month', 'month_sin', 'month_cos']
    
    # Plot 1: Correlation Matrix
    plt.figure(figsize=(14, 10))
    numerical_cols = stage1_features + target_cols
    corr = train_s1[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, annot_kws={"size": 9})
    plt.title('Stage 1: Correlation Matrix (Raw Features)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(STAGE1_VIZ_PATH, '1_correlation_matrix.png'))
    plt.close()
    pbar.update(1)
    
    # Plot 2: Feature vs Target Regression Plots
    g = sns.pairplot(
        train_s1,
        x_vars=['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month_sin', 'month_cos'],
        y_vars=['Dry_Total_g', 'GDM_g', 'Dry_Green_g'],
        kind='reg',
        plot_kws={'scatter_kws': {'alpha': 0.3, 's': 15}, 'line_kws': {'color': 'red', 'lw': 2}},
        height=3
    )
    g.fig.suptitle('Stage 1: Raw Features vs Key Targets', y=1.01, fontsize=16)
    plt.savefig(os.path.join(STAGE1_VIZ_PATH, '2_features_vs_targets.png'))
    plt.close()
    pbar.update(1)
    
    # Plot 3: Train vs Validation Distribution Comparison
    features_to_compare = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month_sin', 'Dry_Total_g']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage 1: Train vs. Validation Distribution Comparison', fontsize=18)
    for i, feature in enumerate(features_to_compare):
        ax = axes.flatten()[i]
        sns.kdeplot(train_s1[feature], ax=ax, label='Train', color='blue', fill=True)
        sns.kdeplot(val_s1[feature], ax=ax, label='Validation', color='orange', fill=True, alpha=0.6)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(STAGE1_VIZ_PATH, '3_train_val_distributions.png'))
    plt.close()
    pbar.update(1)
    pbar.close()
    
    # ========================================================================
    # STAGE 2 VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING STAGE 2 VISUALIZATIONS")
    print("="*80)
    
    df_stage2 = prepare_stage2_features(df_long)
    train_s2, val_s2 = train_test_split(df_stage2, test_size=0.2, random_state=42, stratify=df_stage2['season'])
    
    pbar = tqdm(total=4, desc="Stage 2 plots")
    
    # Stage 2 engineered features
    stage2_features = ['NDVI_Height_MUL', 'NDVI_Height_ADD', 'NDVI_Height_Ratio', 
                       'species_count_global', 'species_freq_global', 'species_count_seasonal']
    
    # Plot 1: Correlation Matrix
    plt.figure(figsize=(16, 12))
    numerical_cols = stage2_features + ['Pre_GSHH_NDVI', 'Height_Ave_cm'] + target_cols
    corr = train_s2[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, annot_kws={"size": 8})
    plt.title('Stage 2: Correlation Matrix (Engineered Features)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(STAGE2_VIZ_PATH, '1_correlation_matrix.png'))
    plt.close()
    pbar.update(1)
    
    # Plot 2: Interaction Features vs Targets
    g = sns.pairplot(
        train_s2,
        x_vars=['NDVI_Height_MUL', 'NDVI_Height_ADD', 'NDVI_Height_Ratio'],
        y_vars=['Dry_Total_g', 'GDM_g', 'Dry_Green_g'],
        kind='reg',
        plot_kws={'scatter_kws': {'alpha': 0.3, 's': 15}, 'line_kws': {'color': 'red', 'lw': 2}},
        height=3
    )
    g.fig.suptitle('Stage 2: Interaction Features vs Key Targets', y=1.01, fontsize=16)
    plt.savefig(os.path.join(STAGE2_VIZ_PATH, '2_interaction_features_vs_targets.png'))
    plt.close()
    pbar.update(1)
    
    # Plot 3: Count/Frequency Features vs Targets
    g = sns.pairplot(
        train_s2,
        x_vars=['species_count_global', 'species_freq_global', 'species_count_seasonal'],
        y_vars=['Dry_Total_g', 'GDM_g'],
        kind='reg',
        plot_kws={'scatter_kws': {'alpha': 0.3, 's': 15}, 'line_kws': {'color': 'red', 'lw': 2}},
        height=3
    )
    g.fig.suptitle('Stage 2: Count/Frequency Features vs Key Targets', y=1.01, fontsize=16)
    plt.savefig(os.path.join(STAGE2_VIZ_PATH, '3_count_features_vs_targets.png'))
    plt.close()
    pbar.update(1)
    
    # Plot 4: Train vs Validation Distribution Comparison
    features_to_compare = ['NDVI_Height_MUL', 'NDVI_Height_Ratio', 'species_count_seasonal', 'Dry_Total_g']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage 2: Train vs. Validation Distribution Comparison', fontsize=18)
    for i, feature in enumerate(features_to_compare):
        ax = axes.flatten()[i]
        sns.kdeplot(train_s2[feature], ax=ax, label='Train', color='blue', fill=True)
        sns.kdeplot(val_s2[feature], ax=ax, label='Validation', color='orange', fill=True, alpha=0.6)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(STAGE2_VIZ_PATH, '4_train_val_distributions.png'))
    plt.close()
    pbar.update(1)
    pbar.close()
    
    print("\n" + "="*80)
    print("EDA COMPLETE")
    print("="*80)
    print(f"Common visualizations: {BASE_VIZ_PATH}/")
    print(f"Stage 1 visualizations: {STAGE1_VIZ_PATH}/")
    print(f"Stage 2 visualizations: {STAGE2_VIZ_PATH}/")