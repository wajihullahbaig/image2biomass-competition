import sys
import os
import pandas as pd
import logging

# Add src to path if running from root
if 'src' not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

import common
from common import load_data, engineer_features


def verify():
    # Setup basic logging to catch load_data info
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Verify")
    stratification_col = 'StratifyKey'
    print("Loading data...")
    try:
        df = load_data(logger)
        df = engineer_features(df, logger)
    except FileNotFoundError:

        print("Error: train.csv not found. Make sure to run from project root.")
        return

    # Simulate Main Logic Pre-processing
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    
    print(f"Total Rows: {len(df)}")
    
    # --- SPLIT LOGIC START (Copied from train_holdout.py) ---
    dev_dfs = []
    holdout_dfs = []
    
    unique_species = df[stratification_col].unique()
    
    for sp in unique_species:
        # Get all samples for this species, ensure sorted by date
        sp_df = df[df[stratification_col] == sp].sort_values('Sampling_Date')
        
        n_samples = len(sp_df)
        if n_samples == 0: continue
            
        # 15% Holdout
        holdout_cnt = int(n_samples * 0.20)
        
        if n_samples < 2:
            dev_dfs.append(sp_df)
            continue
            
        split_idx = n_samples - holdout_cnt
        
        sp_dev = sp_df.iloc[:split_idx]
        sp_hol = sp_df.iloc[split_idx:]
        
        dev_dfs.append(sp_dev)
        holdout_dfs.append(sp_hol)
        
    dev_df = pd.concat(dev_dfs).sort_values('Sampling_Date').reset_index(drop=True)
    global_holdout_df = pd.concat(holdout_dfs).sort_values('Sampling_Date').reset_index(drop=True)
    # --- SPLIT LOGIC END ---

    print("\n" + "="*40)
    print("VERIFICATION REPORT")
    print("="*40)
    print(f"Development Set: {len(dev_df)} ({(len(dev_df)/len(df))*100:.1f}%)")
    print(f"Holdout Set:     {len(global_holdout_df)} ({(len(global_holdout_df)/len(df))*100:.1f}%)")
    
    print("\nPer-Species Temporal Check:")
    print(f"{'FunctionalGroup':<25} | {'Dev Count':<10} | {'Hol Count':<10} | {'Dev Max Date':<12} | {'Hol Min Date':<12} | {'Status':<10}")
    print("-" * 90)
    
    violations = 0
    for sp in unique_species:
        d = dev_df[dev_df[stratification_col] == sp]
        h = global_holdout_df[global_holdout_df[stratification_col] == sp]
        
        d_cnt = len(d)
        h_cnt = len(h)
        
        if d_cnt > 0 and h_cnt > 0:
            d_max = d['Sampling_Date'].max()
            h_min = h['Sampling_Date'].min()
            
            # Allow equal if timestamp is identical (same day, different sample)
            # But effectively we want to see that h_min >= d_max
            valid = h_min >= d_max
            status = "OK" if valid else "FAIL"
            if not valid: violations += 1
            
            print(f"{sp:<25} | {d_cnt:<10} | {h_cnt:<10} | {str(d_max.date()):<12} | {str(h_min.date()):<12} | {status}")
        else:
            print(f"{sp:<25} | {d_cnt:<10} | {h_cnt:<10} | {'N/A':<12} | {'N/A':<12} | SKIPPED")

    print("-" * 90)
    if violations == 0:
        print("VERIFICATION SUCCESS: All measurable species respecting temporal split.")
    else:
        print(f"VERIFICATION FAILED: {violations} species temporal violations.")

if __name__ == "__main__":
    verify()
