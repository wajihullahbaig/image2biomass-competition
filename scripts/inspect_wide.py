#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
p = Path(__file__).resolve().parents[0] / '..' / 'wide.csv'
cnt = Counter()
state_per_species = defaultdict(set)
season_per_species = defaultdict(set)
dates_per_species = defaultdict(list)
with open(p, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        s = row['Species']
        st = row['State']
        se = row['season']
        try:
            d = datetime.strptime(row['Sampling_Date'], '%Y-%m-%d').date()
        except Exception:
            continue
        cnt[s] += 1
        state_per_species[s].add(st)
        season_per_species[s].add(se)
        dates_per_species[s].append(d)

print('n_samples', sum(cnt.values()))
print('n_species', len(cnt))
print('\nTop 20 species by sample count:')
for s, c in cnt.most_common(20):
    print(f'{s}: {c}')

singles = [s for s, c in cnt.items() if c == 1]
print('\nsingle_species_count', len(singles))
if singles:
    print('examples (up to 30):', singles[:30])

only_one_state = [s for s in cnt if len(state_per_species[s]) == 1]
only_one_season = [s for s in cnt if len(season_per_species[s]) == 1]
print('\nonly_one_state_count', len(only_one_state))
print('only_one_season_count', len(only_one_season))

same_date_species = [s for s, ds in dates_per_species.items() if min(ds) == max(ds)]
print('\nsame_date_species_count', len(same_date_species))

all_dates = [d for ds in dates_per_species.values() for d in ds]
print('\nmin_date', min(all_dates).isoformat(), 'max_date', max(all_dates).isoformat())

by_min = sorted(dates_per_species.items(), key=lambda x: min(x[1]))[:10]
print('\n10 species with earliest sample:')
for s, ds in by_min:
    print(s, min(ds).isoformat())

by_max = sorted(dates_per_species.items(), key=lambda x: max(x[1]), reverse=True)[:10]
print('\n10 species with latest sample:')
for s, ds in by_max:
    print(s, max(ds).isoformat())

# list species only present in one state + one season and single sample
conflicts = [s for s in cnt if cnt[s] < 3 and len(state_per_species[s])==1 and len(season_per_species[s])==1]
print('\nconflict_candidates (<=2 samples & only 1 state & 1 season):', len(conflicts))
print(conflicts[:50])

# write a small csv summary for inspection
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / 'wide_summary'
OUTPUT_DIR.mkdir(exist_ok=True)
out = OUTPUT_DIR / 'wide_summary.csv'
# create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Species','Count','States','Seasons','MinDate','MaxDate'])
    for s in sorted(cnt.keys()):
        w.writerow([s, cnt[s], '|'.join(sorted(state_per_species[s])), '|'.join(sorted(season_per_species[s])), min(dates_per_species[s]).isoformat(), max(dates_per_species[s]).isoformat()])
print('\nSummary written to', out)
