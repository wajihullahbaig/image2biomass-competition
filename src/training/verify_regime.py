
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from common import load_data, engineer_features, smart_temporal_split

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
    
    print("2. Verifying temporal split (dev vs holdout)...")
    dev_df, hold_df = smart_temporal_split(df, stratify_col='State_Species')
    train_sessions = set(dev_df['SessionID'])
    hold_sessions = set(hold_df['SessionID'])
    leak = train_sessions.intersection(hold_sessions)
    if leak:
        print("FAILED: Leakage between dev and holdout sessions:", leak)
    else:
        print("OK: No session leakage between dev and holdout.")
    # Temporal order checks
    dev_max_date = dev_df['Sampling_Date'].max()
    hold_min_date = hold_df['Sampling_Date'].min()
    if dev_max_date > hold_min_date:
        print(f"WARNING: Temporal inversion: dev max {dev_max_date} > holdout min {hold_min_date}")
    else:
        print("OK: Temporal order respected (dev before holdout).")

if __name__ == "__main__":
    verify_regime()
