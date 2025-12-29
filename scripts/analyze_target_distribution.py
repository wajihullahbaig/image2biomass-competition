
import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add src to path to allow importing from common
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

from common import load_data

def setup_logger():
    logger = logging.getLogger("Analysis")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def analyze_distributions(output_dir='time_analysis'):
    logger = setup_logger()
    
    # Load Data (Note: load_data returns targets in KG Scale)
    try:
        # We need to change cwd to root to allow load_data to find train.csv if it expects it in current dir
        # Assuming script is run from project root, or we handle it in common.
        # But common.py says: if not os.path.exists('train.csv'): raise ...
        # So we must run this from project root, or ensure train.csv is found.
        # The user's workspace root seems to be C:\Users\Precision\Onus\GitHub\image2biomass-competition
        # We will assume the script is run from there.
        df = load_data(logger)
    except FileNotFoundError:
        logger.error("Could not find train.csv. Make sure you run this script from the project root.")
        return

    # User asked for "raw targets". Usually that means grams.
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    df_plot = df.copy()
    df_plot[target_cols] = df_plot[target_cols] 

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique States
    states = df_plot['State'].unique()
    
    for state in states:
        state_dir = os.path.join(output_dir, state)
        os.makedirs(state_dir, exist_ok=True)
        
        state_df = df_plot[df_plot['State'] == state]
        species_list = state_df['Species'].unique()
        
        logger.info(f"Processing State: {state} ({len(species_list)} species)")
        
        for species in species_list:
            sp_df = state_df[state_df['Species'] == species].sort_values('Sampling_Date')
            
            if len(sp_df) == 0:
                continue

            # Create Plot: 5 Subplots
            fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
            fig.suptitle(f"Target Distribution Over Time\nState: {state} | Species: {species}", fontsize=16)
            
            colors = ['tab:green', 'tab:brown', 'tab:olive', 'tab:blue', 'tab:purple']
            
            for i, target in enumerate(target_cols):
                ax = axes[i]
                
                # Plot Scatter
                sns.scatterplot(data=sp_df, x='Sampling_Date', y=target, ax=ax, color=colors[i], s=100, alpha=0.7)
                
                # Optional: Connect dots with line if it makes sense, usually scatter is safer for sparse data
                sns.lineplot(data=sp_df, x='Sampling_Date', y=target, ax=ax, color=colors[i], alpha=0.3)
                
                ax.set_ylabel(f"{target} (g)", fontsize=12)
                ax.set_title(target, fontsize=10, loc='left')
                ax.grid(True, alpha=0.3)
                
                # Annotate max value
                if len(sp_df) > 0:
                    max_val = sp_df[target].max()
                    max_date = sp_df.loc[sp_df[target].idxmax(), 'Sampling_Date']
                    ax.annotate(f'Max: {max_val:.1f}g', 
                                xy=(max_date, max_val), 
                                xytext=(10, 10), textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", color='black'))

            plt.xlabel("Date")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(state_dir, f"{species}.png")
            plt.savefig(save_path)
            plt.close()
            
    logger.info(f"Analysis complete. Plots saved to '{output_dir}'")

if __name__ == "__main__":
    analyze_distributions()
