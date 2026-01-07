
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from common import load_data, engineer_features, triple_moving_time_series_split

import logging

def verify_regime():
    logger = logging.getLogger("Verify")
    logging.basicConfig(level=logging.INFO)
    
    print("1. Loading data...")
    # Mock logger to avoid too much output
    df = load_data(logger)
    df = engineer_features(df, logger)

    
    if 'SessionID' not in df.columns:
        print("FAILED: SessionID not found in dataframe")
        return
    
    print(f"Total sessions: {df['SessionID'].nunique()}")
    
    print("2. Verifying Triple splits...")
    n_folds = 5
    for fold, (train_idx, val_idx, hold_idx) in enumerate(triple_moving_time_series_split(df, n_splits=n_folds)):
        train_sessions = set(df.iloc[train_idx]['SessionID'])
        val_sessions = set(df.iloc[val_idx]['SessionID'])
        hold_sessions = set(df.iloc[hold_idx]['SessionID'])
        
        # Intersection checks
        tv_leak = train_sessions.intersection(val_sessions)
        vh_leak = val_sessions.intersection(hold_sessions)
        th_leak = train_sessions.intersection(hold_sessions)
        
        if tv_leak or vh_leak or th_leak:
            print(f"FAILED: Leakage in Fold {fold+1}")
            if tv_leak: print(f"  T-V Leak: {tv_leak}")
            if vh_leak: print(f"  V-H Leak: {vh_leak}")
            if th_leak: print(f"  T-H Leak: {th_leak}")
        else:
            print(f"Fold {fold+1}: OK")
            print(f"  Train: {len(train_idx)} samples, {len(train_sessions)} sessions")
            print(f"  Val:   {len(val_idx)} samples, {len(val_sessions)} sessions")
            print(f"  Hold:  {len(hold_idx)} samples, {len(hold_sessions)} sessions")
            
        # Verify temporal order
        train_max_date = df.iloc[train_idx]['Sampling_Date'].max()
        val_min_date = df.iloc[val_idx]['Sampling_Date'].min()
        val_max_date = df.iloc[val_idx]['Sampling_Date'].max()
        hold_min_date = df.iloc[hold_idx]['Sampling_Date'].min()
        
        if train_max_date > val_min_date:
            print(f"  WARNING: T-V Temporal Inversion: {train_max_date} > {val_min_date}")
        if val_max_date > hold_min_date:
            print(f"  WARNING: V-H Temporal Inversion: {val_max_date} > {hold_min_date}")

if __name__ == "__main__":
    verify_regime()
