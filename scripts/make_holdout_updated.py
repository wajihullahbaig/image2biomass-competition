#!/usr/bin/env python3
"""
Updated temporally-aware holdout splitter for engineered species.
Ensures all base species components are represented in both train and holdout sets.

Strategy:
1. Base Holdout: Last N days of samples (temporal split)
2. Component Coverage Check: Verify all base species are in both sets
3. Patch if needed: If a base species is missing from holdout, add one representative sample
"""
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
INPUT_CSV = Path('./wide.csv')
OUTPUT_DIR = Path('./analysis_results/holdout_updated/')
N_DAYS = 45
SEED = 42

BASE_SPECIES = [
    'Clover', 'WhiteClover', 'SubcloverDalkeith', 'SubcloverLosa',
    'Ryegrass', 'Phalaris', 'Fescue', 'Lucerne',
    'Barleygrass', 'Silvergrass', 'Speargrass', 'Bromegrass',
    'Capeweed', 'Crumbweed'
]

def parse_species_components(species_name):
    """Parse species name into its component base species."""
    if species_name == 'Mixed':
        return BASE_SPECIES.copy()
    
    components = []
    parts = species_name.split('_')
    
    for part in parts:
        if part == 'Clover':
            components.extend(['Clover', 'WhiteClover', 'SubcloverDalkeith', 'SubcloverLosa'])
        else:
            components.append(part)
    
    return components

def get_base_species_in_set(df):
    """Get all base species present in a dataframe."""
    base_species_set = set()
    for species in df['Species'].unique():
        components = parse_species_components(species)
        base_species_set.update(components)
    return base_species_set

def generate_holdout(n_days=N_DAYS, seed=SEED):
    """Generate holdout with base species coverage guarantee."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Output paths
    HOLDOUT_CSV = OUTPUT_DIR / 'holdout.csv'
    TRAIN_CSV = OUTPUT_DIR / 'train_filtered.csv'
    HOLDOUT_REPORT = OUTPUT_DIR / 'holdout_report.csv'
    SPECIES_REPORT = OUTPUT_DIR / 'holdout_species_detail.csv'
    COMPONENT_REPORT = OUTPUT_DIR / 'base_species_coverage.csv'
    
    print('='*80)
    print('GENERATING TEMPORALLY-AWARE HOLDOUT WITH BASE SPECIES COVERAGE')
    print('='*80)
    
    # Load data
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df['components'] = df['Species'].apply(parse_species_components)
    
    print(f'\n📊 Dataset Info:')
    print(f'   Total samples: {len(df)}')
    print(f'   Date range: {df["Sampling_Date"].min().date()} to {df["Sampling_Date"].max().date()}')
    print(f'   Unique species combinations: {df["Species"].nunique()}')
    
    # 1. TEMPORAL SPLIT: Last N days -> Holdout
    max_date = df['Sampling_Date'].max()
    cutoff_date = max_date - pd.Timedelta(days=n_days)
    
    print(f'\n📅 Temporal Split:')
    print(f'   Cutoff date: {cutoff_date.date()}')
    print(f'   Holdout window: Last {n_days} days')
    
    # Create CV groups to prevent leakage
    df['cv_group'] = df['State'] + "_" + df['Sampling_Date'].dt.strftime('%Y-%m-%d')
    
    # Initial split
    holdout_mask = df['Sampling_Date'] >= cutoff_date
    holdout_df = df[holdout_mask].copy()
    train_df = df[~holdout_mask].copy()
    
    # Check for CV group leakage
    h_groups = set(holdout_df['cv_group'])
    t_groups = set(train_df['cv_group'])
    overlap = h_groups.intersection(t_groups)
    
    if overlap:
        print(f'   ⚠️  Found {len(overlap)} overlapping CV groups, moving to holdout...')
        holdout_df = pd.concat([holdout_df, train_df[train_df['cv_group'].isin(overlap)]])
        train_df = train_df[~train_df['cv_group'].isin(overlap)]
    
    print(f'   ✓ Initial split: {len(train_df)} train, {len(holdout_df)} holdout')
    
    # 2. BASE SPECIES COVERAGE CHECK
    print(f'\n🔍 Checking Base Species Coverage:')
    
    train_base_species = get_base_species_in_set(train_df)
    holdout_base_species = get_base_species_in_set(holdout_df)
    all_base_species = get_base_species_in_set(df)
    
    missing_from_train = all_base_species - train_base_species
    missing_from_holdout = all_base_species - holdout_base_species
    
    print(f'   Total base species in dataset: {len(all_base_species)}')
    print(f'   Base species in train: {len(train_base_species)}')
    print(f'   Base species in holdout: {len(holdout_base_species)}')
    
    # 3. PATCH MISSING BASE SPECIES
    patches_applied = []
    
    if missing_from_holdout:
        print(f'\n   ⚠️  {len(missing_from_holdout)} base species missing from holdout:')
        print(f'      {sorted(missing_from_holdout)}')
        print(f'\n   🔧 Applying patches...')
        
        for base_species in sorted(missing_from_holdout):
            # Find the most recent sample containing this base species
            candidate_mask = train_df['components'].apply(lambda x: base_species in x)
            candidates = train_df[candidate_mask].sort_values('Sampling_Date', ascending=False)
            
            if len(candidates) > 0:
                # Move the most recent sample to holdout
                patch_sample = candidates.iloc[0:1]
                patch_id = patch_sample['sample_id'].iloc[0]
                patch_species = patch_sample['Species'].iloc[0]
                patch_date = patch_sample['Sampling_Date'].iloc[0]
                
                holdout_df = pd.concat([holdout_df, patch_sample])
                train_df = train_df[train_df['sample_id'] != patch_id]
                
                patches_applied.append({
                    'base_species': base_species,
                    'sample_id': patch_id,
                    'species': patch_species,
                    'date': patch_date.strftime('%Y-%m-%d'),
                    'state': patch_sample['State'].iloc[0]
                })
                
                print(f'      ✓ Patched {base_species}: moved sample {patch_id} '
                      f'({patch_species}, {patch_date.date()})')
            else:
                print(f'      ⚠️  Could not find sample with {base_species} in train set!')
    else:
        print(f'   ✓ All base species already present in holdout!')
    
    if missing_from_train:
        print(f'\n   ⚠️  Warning: {len(missing_from_train)} base species missing from train:')
        print(f'      {sorted(missing_from_train)}')
        print(f'      This may affect model training!')
    
    # 4. FINAL STATISTICS
    train_base_species = get_base_species_in_set(train_df)
    holdout_base_species = get_base_species_in_set(holdout_df)
    
    print(f'\n📈 Final Split Statistics:')
    print(f'   Train samples: {len(train_df)}')
    print(f'   Holdout samples: {len(holdout_df)} (includes {len(patches_applied)} patches)')
    print(f'   Train base species coverage: {len(train_base_species)}/{len(all_base_species)}')
    print(f'   Holdout base species coverage: {len(holdout_base_species)}/{len(all_base_species)}')
    
    # 5. WRITE OUTPUTS
    print(f'\n💾 Writing outputs...')
    
    # Keep components for coverage report, clean for CSV output
    holdout_df_for_csv = holdout_df.drop(columns=['components', 'cv_group']).copy()
    train_df_for_csv = train_df.drop(columns=['components', 'cv_group']).copy()
    holdout_df_for_csv['Sampling_Date'] = holdout_df_for_csv['Sampling_Date'].dt.strftime('%Y-%m-%d')
    train_df_for_csv['Sampling_Date'] = train_df_for_csv['Sampling_Date'].dt.strftime('%Y-%m-%d')
    
    # Write CSV files
    holdout_df_for_csv.to_csv(HOLDOUT_CSV, index=False)
    train_df_for_csv.to_csv(TRAIN_CSV, index=False)
    
    print(f'   ✓ {HOLDOUT_CSV.name}')
    print(f'   ✓ {TRAIN_CSV.name}')
    
    # Species detail report
    all_species = sorted(df['Species'].unique())
    holdout_counts = holdout_df_for_csv['Species'].value_counts()
    train_counts = train_df_for_csv['Species'].value_counts()
    
    species_detail = []
    for sp in all_species:
        components = parse_species_components(sp)
        species_detail.append({
            'Species': sp,
            'N_Components': len(components),
            'Components': '|'.join(components),
            'Total_Count': len(df[df['Species'] == sp]),
            'Train_Count': train_counts.get(sp, 0),
            'Holdout_Count': holdout_counts.get(sp, 0),
            'In_Train': 'Yes' if sp in train_counts else 'No',
            'In_Holdout': 'Yes' if sp in holdout_counts else 'No'
        })
    
    pd.DataFrame(species_detail).to_csv(SPECIES_REPORT, index=False)
    print(f'   ✓ {SPECIES_REPORT.name}')
    
    # Base species coverage report
    coverage_data = []
    for base_sp in sorted(all_base_species):
        # Count samples containing this base species
        train_samples = sum(1 for comps in train_df['components'] if base_sp in comps)
        holdout_samples = sum(1 for comps in holdout_df['components'] if base_sp in comps)
        total_samples = train_samples + holdout_samples
        
        # Find which species combinations contain this base species
        containing_species = [sp for sp in all_species 
                             if base_sp in parse_species_components(sp)]
        
        coverage_data.append({
            'Base_Species': base_sp,
            'Total_Samples': total_samples,
            'Train_Samples': train_samples,
            'Holdout_Samples': holdout_samples,
            'In_Train': 'Yes' if train_samples > 0 else 'No',
            'In_Holdout': 'Yes' if holdout_samples > 0 else 'No',
            'N_Species_Combinations': len(containing_species),
            'Species_Combinations': '|'.join(containing_species)
        })
    
    pd.DataFrame(coverage_data).to_csv(COMPONENT_REPORT, index=False)
    print(f'   ✓ {COMPONENT_REPORT.name}')
    
    # Main report
    with open(HOLDOUT_REPORT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Value'])
        w.writerow(['Cutoff_Date', cutoff_date.strftime('%Y-%m-%d')])
        w.writerow(['Holdout_Days', n_days])
        w.writerow(['Total_Samples', len(df)])
        w.writerow(['Train_Samples', len(train_df)])
        w.writerow(['Holdout_Samples', len(holdout_df)])
        w.writerow(['Patches_Applied', len(patches_applied)])
        w.writerow(['Base_Species_Total', len(all_base_species)])
        w.writerow(['Base_Species_In_Train', len(train_base_species)])
        w.writerow(['Base_Species_In_Holdout', len(holdout_base_species)])
        w.writerow(['Species_Combinations_Total', df['Species'].nunique()])
        w.writerow(['Species_Combinations_In_Train', train_df['Species'].nunique()])
        w.writerow(['Species_Combinations_In_Holdout', holdout_df['Species'].nunique()])
    
    print(f'   ✓ {HOLDOUT_REPORT.name}')
    
    print(f'\n{"="*80}')
    print('✅ HOLDOUT GENERATION COMPLETE')
    print('='*80)
    
    return len(holdout_df), len(train_df), len(patches_applied)

if __name__ == "__main__":
    h_len, t_len, n_patches = generate_holdout()
    print(f"\n📊 Summary: {h_len} holdout samples ({n_patches} patches), "
          f"{t_len} train samples")
