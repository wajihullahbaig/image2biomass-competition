#!/usr/bin/env python3
"""
Temporally-aware holdout splitter.
Base Holdout: All samples from the latest N sampling dates.
Patch: For any species missing from base, take exactly ONE sample from its latest available date.
Train: Complement of holdout.
"""
import csv
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import random
import pandas as pd

def generate_holdout(n_base_dates=2, seed=42):
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

    # ============================================================================
    # Load data
    # ============================================================================
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"wide.csv not found at {CSV_PATH}. Run load_data() first.")

    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Parse dates and group by species/date
    date_rows = []
    all_species_set = set()
    species_data = defaultdict(list) # species -> list of (date, row)

    for row in rows:
        try:
            d = datetime.strptime(row['Sampling_Date'], '%Y-%m-%d').date()
            date_rows.append((d, row))
            species_data[row['Species']].append((d, row))
            all_species_set.add(row['Species'])
        except (ValueError, KeyError):
            continue

    # Sort unique dates descending
    unique_dates = sorted(set(d for d, _ in date_rows), reverse=True)
    
    # ============================================================================
    # Create Holdout
    # ============================================================================
    base_holdout_dates = unique_dates[:n_base_dates]
    
    holdout_rows = []
    holdout_sample_ids = set()

    # 1. Add all rows from base dates
    for d, row in date_rows:
        if d in base_holdout_dates:
            holdout_rows.append(row)
            holdout_sample_ids.add(row['sample_id'])

    species_in_base = set(row['Species'] for row in holdout_rows)
    missing_species = all_species_set - species_in_base

    # 2. Patch missing species
    for species in sorted(missing_species):
        candidates = sorted(species_data[species], key=lambda x: x[0], reverse=True)
        latest_date, row = candidates[0]
        holdout_rows.append(row)
        holdout_sample_ids.add(row['sample_id'])

    # 3. Create Train Set (Complement)
    train_rows = [row for row in rows if row['sample_id'] not in holdout_sample_ids]

    # ============================================================================
    # Analysis & Reporting
    # ============================================================================
    holdout_counts = Counter(row['Species'] for row in holdout_rows)
    train_counts = Counter(row['Species'] for row in train_rows)
    species_date_counts = {s: len(set(d for d, r in data)) for s, data in species_data.items()}

    species_detail = []
    conflicts = 0
    for species in sorted(all_species_set):
        h_count = holdout_counts[species]
        t_count = train_counts[species]
        total = h_count + t_count
        d_count = species_date_counts[species]
        
        if t_count == 0: conflicts += 1
        
        h_states = set(row['State'] for row in holdout_rows if row['Species'] == species)
        h_seasons = set(row['season'] for row in holdout_rows if row['Species'] == species)
        
        species_detail.append({
            'Species': species,
            'Total_Count': total,
            'Train_Count': t_count,
            'Holdout_Count': h_count,
            'Unique_Dates': d_count,
            'Holdout_States': '|'.join(sorted(h_states)),
            'Holdout_Seasons': '|'.join(sorted(h_seasons)),
            'In_Train': 'Yes' if t_count > 0 else 'NO',
        })

    # ============================================================================
    # Write Outputs
    # ============================================================================
    with open(HOLDOUT_CSV, 'w', newline='', encoding='utf-8') as f:
        if holdout_rows:
            writer = csv.DictWriter(f, fieldnames=holdout_rows[0].keys())
            writer.writeheader()
            writer.writerows(holdout_rows)

    with open(TRAIN_CSV, 'w', newline='', encoding='utf-8') as f:
        if train_rows:
            writer = csv.DictWriter(f, fieldnames=train_rows[0].keys())
            writer.writeheader()
            writer.writerows(train_rows)

    with open(SPECIES_REPORT, 'w', newline='', encoding='utf-8') as f:
        if species_detail:
            writer = csv.DictWriter(f, fieldnames=species_detail[0].keys())
            writer.writeheader()
            writer.writerows(species_detail)

    with open(HOLDOUT_REPORT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total_Samples', len(rows)])
        writer.writerow(['Holdout_Samples', len(holdout_rows)])
        writer.writerow(['Train_Samples', len(train_rows)])
        writer.writerow(['Holdout_Species_Covered', len(species_in_base.union(missing_species))])
        writer.writerow(['Species_Missing_From_Train', conflicts])
        writer.writerow(['Base_Holdout_Dates', '|'.join(map(str, base_holdout_dates))])

    return len(holdout_rows), len(train_rows)

if __name__ == "__main__":
    h_len, t_len = generate_holdout()
    print(f"✓ Holdout Generated: {h_len} holdout, {t_len} train samples.")
