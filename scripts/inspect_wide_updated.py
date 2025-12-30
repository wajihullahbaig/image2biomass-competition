#!/usr/bin/env python3
"""
Updated inspect_wide.py for engineered species analysis.
Handles species that are combinations of base species (e.g., Ryegrass_Clover).
"""
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_CSV = Path('./wide.csv')
OUTPUT_DIR = Path('analysis_results/wide_summary')
OUTPUT_DIR.mkdir(exist_ok=True)

# Base species that compose the combinations
BASE_SPECIES = [
    'Clover', 'WhiteClover', 'SubcloverDalkeith', 'SubcloverLosa',
    'Ryegrass', 'Phalaris', 'Fescue', 'Lucerne',
    'Barleygrass', 'Silvergrass', 'Speargrass', 'Bromegrass',
    'Capeweed', 'Crumbweed'
]

def parse_species_components(species_name):
    """
    Parse species name into its component base species.
    Examples:
        'Clover' -> ['WhiteClover', 'SubcloverDalkeith', 'SubcloverLosa', 'Clover']
        'Ryegrass_Clover' -> ['Ryegrass', 'WhiteClover', 'SubcloverDalkeith', 'SubcloverLosa', 'Clover']
        'Mixed' -> all 14 base species
    """
    if species_name == 'Mixed':
        return BASE_SPECIES.copy()
    
    # Split by underscore and expand 'Clover' into its sub-types
    components = []
    parts = species_name.split('_')
    
    for part in parts:
        if part == 'Clover':
            # Clover expands to 4 sub-types
            components.extend(['Clover', 'WhiteClover', 'SubcloverDalkeith', 'SubcloverLosa'])
        else:
            components.append(part)
    
    return components

def analyze_wide_csv():
    """Main analysis function for engineered species."""
    
    # Counters
    cnt = Counter()
    state_per_species = defaultdict(set)
    season_per_species = defaultdict(set)
    dates_per_species = defaultdict(list)
    components_per_species = {}
    base_species_counts = Counter()  # Count occurrences of base species across all samples
    
    with open(INPUT_CSV, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            s = row['Species']
            st = row['State']
            se = row.get('season', 'Unknown')
            
            try:
                d = datetime.strptime(row['Sampling_Date'], '%Y-%m-%d').date()
            except Exception:
                continue
            
            # Regular counters
            cnt[s] += 1
            state_per_species[s].add(st)
            season_per_species[s].add(se)
            dates_per_species[s].append(d)
            
            # Parse components
            if s not in components_per_species:
                components_per_species[s] = parse_species_components(s)
            
            # Count base species occurrences
            for component in components_per_species[s]:
                base_species_counts[component] += 1
    
    # Print basic statistics
    print('='*80)
    print('ENGINEERED SPECIES ANALYSIS')
    print('='*80)
    print(f'\nTotal Samples: {sum(cnt.values())}')
    print(f'Unique Species (including combinations): {len(cnt)}')
    print(f'\nTop 20 species by sample count:')
    for s, c in cnt.most_common(20):
        n_components = len(components_per_species[s])
        print(f'  {s:<50} {c:>3} samples  ({n_components} components)')
    
    # Single sample species
    singles = [s for s, c in cnt.items() if c == 1]
    print(f'\nSpecies with only 1 sample: {len(singles)}')
    if singles:
        print('  Examples (up to 10):', ', '.join(singles[:10]))
    
    # Geographic and temporal constraints
    only_one_state = [s for s in cnt if len(state_per_species[s]) == 1]
    only_one_season = [s for s in cnt if len(season_per_species[s]) == 1]
    print(f'\nSpecies in only one state: {len(only_one_state)}')
    print(f'Species in only one season: {len(only_one_season)}')
    
    same_date_species = [s for s, ds in dates_per_species.items() if min(ds) == max(ds)]
    print(f'Species with all samples on same date: {len(same_date_species)}')
    
    # Date range
    all_dates = [d for ds in dates_per_species.values() for d in ds]
    print(f'\nDate range: {min(all_dates)} to {max(all_dates)}')
    
    # Base species analysis
    print(f'\n{"="*80}')
    print('BASE SPECIES OCCURRENCE COUNTS')
    print('='*80)
    print('(How many samples contain each base species as a component)\n')
    for base, count in base_species_counts.most_common():
        print(f'  {base:<25} {count:>3} samples')
    
    # Species with potential data issues
    conflicts = [s for s in cnt if cnt[s] <= 2 and 
                 len(state_per_species[s]) == 1 and 
                 len(season_per_species[s]) == 1]
    print(f'\n{"="*80}')
    print(f'Potential problematic species (≤2 samples, 1 state, 1 season): {len(conflicts)}')
    print('='*80)
    if conflicts:
        for s in sorted(conflicts)[:20]:
            print(f'  {s}: {cnt[s]} sample(s)')
    
    # Component analysis
    print(f'\n{"="*80}')
    print('SPECIES BY NUMBER OF COMPONENTS')
    print('='*80)
    by_n_components = defaultdict(list)
    for species, components in components_per_species.items():
        by_n_components[len(components)].append(species)
    
    for n_comp in sorted(by_n_components.keys()):
        species_list = by_n_components[n_comp]
        print(f'\n{n_comp} components ({len(species_list)} species):')
        for s in sorted(species_list):
            count = cnt[s]
            components = ', '.join(components_per_species[s])
            print(f'  {s:<50} {count:>3} samples')
            print(f'    └─ [{components}]')
    
    # Write detailed CSV summary
    output_csv = OUTPUT_DIR / 'wide_summary.csv'
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Species', 'Count', 'N_Components', 'Components', 'States', 
                    'Seasons', 'MinDate', 'MaxDate', 'DateSpan_Days'])
        
        for s in sorted(cnt.keys()):
            components = '|'.join(components_per_species[s])
            states = '|'.join(sorted(state_per_species[s]))
            seasons = '|'.join(sorted(season_per_species[s]))
            min_date = min(dates_per_species[s])
            max_date = max(dates_per_species[s])
            date_span = (max_date - min_date).days
            
            w.writerow([
                s, cnt[s], len(components_per_species[s]), components,
                states, seasons, min_date.isoformat(), max_date.isoformat(),
                date_span
            ])
    
    print(f'\n{"="*80}')
    print(f'✓ Detailed summary written to: {output_csv}')
    print('='*80)

if __name__ == '__main__':
    analyze_wide_csv()
