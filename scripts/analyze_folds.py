import pandas as pd
import numpy as np

def analyze_folds():
    df = pd.read_csv('train.csv')
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    
    # We have 4 folds using TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)
    
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Pivot to wide for easier analysis
    wide = df.pivot_table(index=['sample_id', 'Sampling_Date'], columns='target_name', values='target').reset_index()
    wide = wide.sort_values('Sampling_Date')

    print(f"{'Fold':<10} | {'Period':<25} | {'Mean Total (g)':<15} | {'Std Total':<10}")
    print("-" * 70)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(wide)):
        train_df = wide.iloc[train_idx]
        val_df = wide.iloc[val_idx]
        
        # Max train date for temporal filter
        max_train_date = train_df['Sampling_Date'].max()
        val_df = val_df[val_df['Sampling_Date'] > max_train_date]
        
        if len(val_df) == 0: continue
        
        start = val_df['Sampling_Date'].min().strftime('%Y-%m')
        end = val_df['Sampling_Date'].max().strftime('%Y-%m')
        mean_t = val_df['Dry_Total_g'].mean()
        std_t = val_df['Dry_Total_g'].std()
        
        print(f"Fold {fold+1:<5} | {start} to {end:<15} | {mean_t:<14.2f} | {std_t:<10.2f}")

if __name__ == '__main__':
    analyze_folds()
