import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def convert_and_split():
    csv_path = 'train.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Pivot 5 targets into columns
    p = df.pivot_table(
        index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    # Map sample_id
    id_map = df.drop_duplicates('image_path').set_index('image_path')['sample_id'].to_dict()
    p['sample_id'] = p['image_path'].map(id_map)

    # 5-Fold Stratified Split on State
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(p, y=p['State'])):
        p.loc[val_idx, 'fold'] = fold

    # Ensure required target columns exist
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    for col in target_cols:
        if col not in p.columns:
            raise ValueError(f"Missing target column: {col}")

    out_path = 'train_converted.csv'
    p.to_csv(out_path, index=False)
    print(f"Successfully generated {out_path} with {len(p)} clean samples.")
    for f in range(5):
        sub = p[p['fold'] == f]
        state_counts = sub['State'].value_counts().to_dict()
        print(f"  Fold {f+1}: {len(sub)} samples | States: {state_counts}")

if __name__ == '__main__':
    convert_and_split()
