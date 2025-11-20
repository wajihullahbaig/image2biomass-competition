import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

viz_path = 'visualizations'
if not os.path.exists(viz_path):
    os.makedirs(viz_path)


# Load the training data
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure the file is in the correct directory.")
    exit()

df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
df['month'] = df['Sampling_Date'].dt.month
df['Height_Ave_cm'] = np.log(df['Height_Ave_cm']).replace(-np.inf, 0) # handle log(0)
def get_season(month):
    if month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    elif month in [9, 10, 11]:
        return 'Spring'
    else:
        return 'Summer'

df['season'] = df['month'].apply(get_season)
df['State'] = df['State'].astype('category')
df['Species'] = df['Species'].astype('category')
df['month'] = df['month'].astype('category')
df['season'] = df['season'].astype('category')


# Plotting distributions for categorical features
categorical_features = ['State', 'Species', 'season', 'month']
for feature in categorical_features:
    plt.figure(figsize=(15, 10)) # Increased figure height for better spacing
    sns.boxplot(x=feature, y='target', hue='target_name', data=df, palette='colorblind')
    plt.title(f'Distribution of Target vs {feature}')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right') 
    
    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    plt.savefig(f'{viz_path}/target_vs_{feature}.png')
    plt.close()

# Plotting distributions for numerical features
numerical_features = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
for feature in numerical_features:
    g = sns.FacetGrid(df, col="target_name", col_wrap=3, sharey=False, height=4)
    g.map(sns.scatterplot, feature, "target")
    g.fig.suptitle(f'Distribution of Target vs {feature}', y=1.03)
    plt.tight_layout()
    plt.savefig(f'{viz_path}/target_vs_{feature}_scatter.png')
    plt.close()

print(f"All {viz_path} have been regenerated with rotated x-axis labels and saved to the 'plots' directory.")