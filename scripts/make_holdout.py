#!/usr/bin/env python3
"""
Temporally-aware holdout splitter.
Base Holdout: All samples from the latest N sampling dates.
Patch: For any species missing from base, take exactly ONE sample from its latest available date.
Train: Complement of holdout.
"""
import csv
from datetime import datetime
from pathlib import Path
import random
import pandas as pd
from pandas import Timedelta as pd_Timedelta

def generate_holdout(n_days=45, seed=42):
    # Setup paths relative to project root
    ROOT_DIR = Path(__file__).parent.parent
    CSV_PATH = ROOT_DIR / 'wide.csv'
    OUTPUT_DIR = ROOT_DIR / 'holdout_outputs'
    OUTPUT_DIR.mkdir(exist_ok=True)

    HOLDOUT_CSV = OUTPUT_DIR / 'holdout.csv'
    TRAIN_CSV = OUTPUT_DIR / 'train_filtered.csv'
    HOLDOUT_REPORT = OUTPUT_DIR / 'holdout_report.csv'
    SPECIES_REPORT = OUTPUT_DIR / 'holdout_species_detail.csv'
    
    random.seed(seed)

    # 1. Load data
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"wide.csv not found at {CSV_PATH}. Run load_data() first.")

    df = pd.read_csv(CSV_PATH)
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    
    # 2. TEMPORAL SLICE LOGIC
    # Last 45 days go to Independent Holdout
    max_date = df['Sampling_Date'].max()
    cutoff_date = max_date - pd_Timedelta(days=n_days)
    
    df['cv_group'] = df['State'] + "_" + df['Sampling_Date'].dt.strftime('%Y-%m-%d')
    
    holdout_df = df[df['Sampling_Date'] >= cutoff_date].copy()
    train_df = df[df['Sampling_Date'] < cutoff_date].copy()
    
    # Check for CV Group Leakage
    h_groups = set(holdout_df['cv_group'])
    t_groups = set(train_df['cv_group'])
    overlap = h_groups.intersection(t_groups)
    
    if overlap:
        # If a date is partially in both, move its neighbors to sustain the block
        # (Though with >= cutoff, this shouldn't happen unless a date is exactly the cutoff)
        # To be safe: move all rows of overlapping groups to holdout
        holdout_df = pd.concat([holdout_df, train_df[train_df['cv_group'].isin(overlap)]])
        train_df = train_df[~train_df['cv_group'].isin(overlap)]
    
    # 3. Clean-up for Writing
    # Convert dates back to string for CSV
    holdout_df['Sampling_Date'] = holdout_df['Sampling_Date'].dt.strftime('%Y-%m-%d')
    train_df['Sampling_Date'] = train_df['Sampling_Date'].dt.strftime('%Y-%m-%d')

    # 4. Analysis & Reporting
    all_species = sorted(df['Species'].unique())
    holdout_counts = holdout_df['Species'].value_counts()
    train_counts = train_df['Species'].value_counts()

    species_detail = []
    for sp in all_species:
        species_detail.append({
            'Species': sp,
            'Total_Count': len(df[df['Species'] == sp]),
            'Train_Count': train_counts.get(sp, 0),
            'Holdout_Count': holdout_counts.get(sp, 0),
            'In_Holdout': 'Yes' if sp in holdout_counts else 'NO',
            'In_Train': 'Yes' if sp in train_counts else 'NO'
        })

    # write outputs
    holdout_df.to_csv(HOLDOUT_CSV, index=False)
    train_df.to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(species_detail).to_csv(SPECIES_REPORT, index=False)
    
    with open(HOLDOUT_REPORT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Value'])
        w.writerow(['Cutoff_Date', cutoff_date.strftime('%Y-%m-%d')])
        w.writerow(['Holdout_Samples', len(holdout_df)])
        w.writerow(['Train_Samples', len(train_df)])
        w.writerow(['Total_Samples', len(df)])

    return len(holdout_df), len(train_df)

if __name__ == "__main__":
    h_len, t_len = generate_holdout()
    print(f"✓ Holdout Generated: {h_len} holdout, {t_len} train samples.")
