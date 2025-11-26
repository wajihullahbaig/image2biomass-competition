import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
VIZ_PATH = 'visualizations'
if not os.path.exists(VIZ_PATH):
    os.makedirs(VIZ_PATH)
    print(f"Created directory: {VIZ_PATH}")

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100 # Lower DPI for faster generation, increase for publication quality

# --- Helper Functions (Mirrors training scripts) ---

def get_season(month):
    """Maps month to Australian seasons."""
    if month in [12, 1, 2]: return 'Summer'
    elif month in [3, 4, 5]: return 'Autumn'
    elif month in [6, 7, 8]: return 'Winter'
    else: return 'Spring'

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same feature engineering as used in the training pipelines.
    Pivots data to wide format and creates date, interaction, and count features.
    """
    print("Pivoting data to wide format...")
    # --- FIX IS HERE: Added 'Pre_GSHH_NDVI' and 'Height_Ave_cm' to the index ---
    wide_df = df.pivot_table(
        index=['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    print("Engineering date-based features...")
    wide_df['Sampling_Date'] = pd.to_datetime(wide_df['Sampling_Date'])
    wide_df['month'] = wide_df['Sampling_Date'].dt.month
    wide_df['season'] = wide_df['month'].apply(get_season)
    period = 12
    wide_df['month_sin'] = np.sin(2 * np.pi * wide_df['month'] / period)
    wide_df['month_cos'] = np.cos(2 * np.pi * wide_df['month'] / period)
    
    # Impute and Log-transform key predictors
    print("Imputing and transforming predictor variables...")
    for col in ['Pre_GSHH_NDVI', 'Height_Ave_cm']:
        if wide_df[col].isnull().any():
            median_val = wide_df[col].median()
            wide_df[col] = wide_df[col].fillna(median_val)

    wide_df['log_Height_Ave_cm'] = np.log1p(wide_df['Height_Ave_cm'])
    
    print("Engineering interaction features...")
    wide_df['NDVI_x_logHeight'] = wide_df['Pre_GSHH_NDVI'] * wide_df['log_Height_Ave_cm']
    wide_df['NDVI_+_logHeight'] = wide_df['Pre_GSHH_NDVI'] + wide_df['log_Height_Ave_cm']
    ratio = wide_df['Pre_GSHH_NDVI'] / (wide_df['Height_Ave_cm'] + 1e-5) 
    wide_df['NDVI_log_Height_Ratio'] = ratio

    print("Engineering count and frequency features...")
    # Global Species Features
    species_counts = wide_df['Species'].value_counts()
    wide_df['species_count_global'] = wide_df['Species'].map(np.log1p(species_counts))
    wide_df['species_freq_global'] = wide_df['Species'].map(species_counts / len(wide_df))
    
    # Seasonal Species Features
    seasonal_counts = wide_df.groupby(['season', 'Species']).size()
    wide_df['species_count_seasonal'] = wide_df.apply(
        lambda row: np.log1p(seasonal_counts.get((row['season'], row['Species']), 0)), axis=1
    )
    
    return wide_df.drop(columns=['Sampling_Date'])


# --- Main EDA Script ---
if __name__ == '__main__':
    print("Starting Comprehensive EDA...")
    
    # 1. Load and Preprocess Data
    try:
        df_long = pd.read_csv('train.csv')
        print(f"Loaded train.csv with {len(df_long)} rows.")
    except FileNotFoundError:
        print("Error: train.csv not found. Please ensure the file is in the correct directory.")
        exit()

    df = engineer_features(df_long)
    
    # 2. Perform Stratified Split to Analyze Distributions
    print("Performing stratified split to analyze train/val distributions...")
    strat_col = 'season'
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # For EDA, we don't need y, just the split indices
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[strat_col]
    )
    print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # --- 3. Generate Visualizations ---
    print("\nGenerating visualizations...")
    pbar = tqdm(total=8, desc="Creating plots")

    # Plot 1: Correlation Matrix
    plt.figure(figsize=(16, 12))
    numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    corr = train_df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, annot_kws={"size": 8})
    plt.title('Correlation Matrix of Numerical Features and Targets (Train Set)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_PATH, '1_correlation_matrix.png'))
    plt.close()
    pbar.update(1)

    # Plot 2: Train vs. Validation Distribution Comparison
    features_to_compare = [
        'Pre_GSHH_NDVI', 'log_Height_Ave_cm', 'species_count_global', 
        'species_freq_global', 'species_count_seasonal', 'Dry_Total_g'
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Train vs. Validation Set Distribution Comparison', fontsize=20)
    for i, feature in enumerate(features_to_compare):
        ax = axes.flatten()[i]
        sns.kdeplot(train_df[feature], ax=ax, label='Train', color='blue', fill=True)
        sns.kdeplot(val_df[feature], ax=ax, label='Validation', color='orange', fill=True, alpha=0.6)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(VIZ_PATH, '2_train_val_distribution_comparison.png'))
    plt.close()
    pbar.update(1)

    # Plot 3: Target Distributions (Log and Real Scale)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Target Variable Distributions', fontsize=20)
    for i, target in enumerate(target_cols):
        # Real Scale
        sns.histplot(df[target], ax=axes[0, i], kde=True, bins=30)
        axes[0, i].set_title(f'{target} (Real Scale)')
        # Log Scale
        sns.histplot(np.log1p(df[target]), ax=axes[1, i], kde=True, bins=30, color='green')
        axes[1, i].set_title(f'{target} (Log1p Scale)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(VIZ_PATH, '3_target_distributions.png'))
    plt.close()
    pbar.update(1)

    # Plot 4: Categorical Features vs. Main Target (Dry_Total_g)
    cat_features = ['State', 'Species', 'season']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Categorical Features vs. Dry_Total_g (log1p scale)', fontsize=20)
    for i, feature in enumerate(cat_features):
        order = train_df[feature].value_counts().index
        sns.violinplot(x=feature, y=np.log1p(train_df['Dry_Total_g']), data=train_df, ax=axes[i], order=order, palette='viridis', cut=0)
        axes[i].set_title(f'Dry_Total_g by {feature}')
        if feature == 'Species':
            axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(VIZ_PATH, '4_categorical_vs_target.png'))
    plt.close()
    pbar.update(1)
    
    # Plot 5: Numerical Predictors vs. Main Target (Dry_Total_g)
    num_predictors = ['Pre_GSHH_NDVI', 'log_Height_Ave_cm', 'NDVI_x_logHeight']
    g = sns.pairplot(
        train_df,
        x_vars=num_predictors,
        y_vars=['Dry_Total_g', 'GDM_g'],
        kind='reg',
        plot_kws={'scatter_kws': {'alpha': 0.3, 's': 20}, 'line_kws': {'color': 'red'}}
    )
    g.fig.suptitle('Numerical Predictors vs. Key Targets', y=1.02, fontsize=16)
    plt.savefig(os.path.join(VIZ_PATH, '5_numerical_vs_target_pairplot.png'))
    plt.close()
    pbar.update(1)

    # Plot 6: Engineered Count/Frequency Feature Distributions
    count_freq_features = ['species_count_global', 'species_freq_global', 'species_count_seasonal']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribution of Engineered Count/Frequency Features (Train Set)', fontsize=16)
    for i, feature in enumerate(count_freq_features):
        sns.histplot(train_df[feature], ax=axes[i], kde=True, bins=25)
        axes[i].set_title(feature)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(VIZ_PATH, '6_engineered_feature_distributions.png'))
    plt.close()
    pbar.update(1)

    # Plot 7: Samples per State
    plt.figure(figsize=(10, 6))
    sns.countplot(y='State', data=df, order=df['State'].value_counts().index, palette='crest')
    plt.title('Number of Samples per State', fontsize=16)
    plt.xlabel('Count')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_PATH, '7_samples_per_state.png'))
    plt.close()
    pbar.update(1)

    # Plot 8: Mean Total Biomass by Month
    monthly_biomass = df.groupby('month')['Dry_Total_g'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='month', y='Dry_Total_g', data=monthly_biomass, marker='o', lw=2)
    sns.barplot(x='month', y='Dry_Total_g', data=monthly_biomass, alpha=0.5, palette='coolwarm')
    plt.title('Mean Dry_Total_g by Month', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Mean Dry_Total_g')
    plt.xticks(ticks=range(12), labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_PATH, '8_mean_biomass_by_month.png'))
    plt.close()
    pbar.update(1)
    pbar.close()

    print(f"\nEDA complete. All visualizations have been saved to the '{VIZ_PATH}' directory.")