#!/usr/bin/env python3
"""
Temporally-aware holdout splitter.
Base Holdout: All samples from the latest N sampling dates.
Patch: For any species missing from base, take exactly ONE sample from its latest available date.
Train: Complement of holdout.

This approach minimizes temporal leakage while ensuring all species are represented in the validation set.
"""
import csv
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import random

# Set seed for reproducible patching if multiple samples exist on the same latest date
random.seed(42)

# Config
CSV_PATH = Path(__file__).parent.parent / 'wide.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'holdout_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

HOLDOUT_CSV = OUTPUT_DIR / 'holdout.csv'
TRAIN_CSV = OUTPUT_DIR / 'train_filtered.csv'
HOLDOUT_REPORT = OUTPUT_DIR / 'holdout_report.csv'
SPECIES_REPORT = OUTPUT_DIR / 'holdout_species_detail.csv'

# Parameters
N_BASE_DATES = 2  # Use the last 2 sampling dates as the primary temporal holdout

# ============================================================================
# Load data
# ============================================================================
print(f"Loading {CSV_PATH.name}...")
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
print(f"Found {len(unique_dates)} unique sampling dates.")
print(f"Total rows: {len(rows)}, valid date rows: {len(date_rows)}")
print(f"Total unique species: {len(all_species_set)}")

# ============================================================================
# Create Holdout
# ============================================================================
base_holdout_dates = unique_dates[:N_BASE_DATES]
print(f"\nBase holdout dates (last {N_BASE_DATES}):")
for d in base_holdout_dates:
    print(f"  - {d}")

holdout_rows = []
holdout_sample_ids = set()

# 1. Add all rows from base dates
for d, row in date_rows:
    if d in base_holdout_dates:
        holdout_rows.append(row)
        holdout_sample_ids.add(row['sample_id'])

species_in_base = set(row['Species'] for row in holdout_rows)
missing_species = all_species_set - species_in_base

print(f"Base holdout: {len(holdout_rows)} samples, covers {len(species_in_base)}/{len(all_species_set)} species.")

# 2. Patch missing species
print(f"\nPatching {len(missing_species)} missing species...")
for species in sorted(missing_species):
    # Find latest date available for this species
    candidates = sorted(species_data[species], key=lambda x: x[0], reverse=True)
    # Take the VERY latest (even if it's far back, it's the 'future-most' for this species)
    latest_date, row = candidates[0]
    holdout_rows.append(row)
    holdout_sample_ids.add(row['sample_id'])

# 3. Create Train Set (Complement)
train_rows = [row for row in rows if row['sample_id'] not in holdout_sample_ids]

# ============================================================================
# Analysis & Reporting
# ============================================================================
print("\n" + "="*70)
print(f"Holdout size: {len(holdout_rows)} samples")
print(f"Train size:   {len(train_rows)} samples")
print("="*70)

# Species coverage analysis
holdout_counts = Counter(row['Species'] for row in holdout_rows)
train_counts = Counter(row['Species'] for row in train_rows)
species_date_counts = {s: len(set(d for d, r in data)) for s, data in species_data.items()}

# Prepare species detail report
species_detail = []
conflicts = []

for species in sorted(all_species_set):
    h_count = holdout_counts[species]
    t_count = train_counts[species]
    total = h_count + t_count
    d_count = species_date_counts[species]
    
    # Conflict: Species only in holdout (unseen in training)
    if t_count == 0:
        conflicts.append(f"{species} (Total={total}, Dates={d_count})")
    
    # Coverage of States/Seasons in Holdout
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
        'In_Train': 'Yes' if t_count > 0 else 'NO (Conflict)',
    })

if conflicts:
    print(f"\n⚠ Warning: {len(conflicts)} species NOT present in training set:")
    for c in conflicts[:10]:
        print(f"  - {c}")
    if len(conflicts) > 10:
        print(f"  ... and {len(conflicts)-10} more.")

# ============================================================================
# Write Outputs
# ============================================================================
print(f"\nWriting outputs to {OUTPUT_DIR}...")

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
    writer.writerow(['Species_Missing_From_Train', len(conflicts)])
    writer.writerow(['Base_Holdout_Dates', '|'.join(map(str, base_holdout_dates))])

print("✓ Done!")
