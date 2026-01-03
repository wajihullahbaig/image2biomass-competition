#!/usr/bin/env python3
"""
Updated analyze_target_distribution.py for engineered species.
Creates plots organized by State -> Base Species (showing all combinations that contain it).
"""
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
INPUT_CSV = Path('./wide.csv')
OUTPUT_DIR = Path('analysis_results/target_distribution_updated/')

TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

BASE_SPECIES = [
    'clover', 'whiteClover', 'subcloverdalkeith', 'subcloverLosa',
    'ryegrass', 'phalaris', 'fescue', 'lucerne',
    'barleygrass', 'silvergrass', 'speargrass', 'bromegrass',
    'capeweed', 'crumbweed'
]

def parse_species_components(species_name):
    """Parse species name into its component base species."""
    if species_name == 'Mixed':
        return BASE_SPECIES.copy()
    
    components = []
    parts = species_name.split('_')
    
    for part in parts:
        if part == 'clover':
            components.extend(['clover', 'whiteclover', 'subcloverdalkeith', 'subcloverlosa'])
        else:
            components.append(part)
    
    return components

def create_species_plots():
    """Create plots organized by State and Base Species."""
    
    # Load data
    df = pd.read_csv(INPUT_CSV)
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    
    # Parse species components
    df['components'] = df['Species'].apply(parse_species_components)
    
    # Create output structure
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    states = sorted(df['State'].unique())
    
    print('='*80)
    print('GENERATING SPECIES DISTRIBUTION PLOTS')
    print('='*80)
    
    for state in states:
        state_df = df[df['State'] == state].copy()
        state_dir = OUTPUT_DIR / state
        state_dir.mkdir(exist_ok=True)
        
        print(f'\n📍 Processing State: {state}')
        print(f'   Total samples: {len(state_df)}')
        
        # Get all base species present in this state
        base_species_in_state = set()
        for components_list in state_df['components']:
            base_species_in_state.update(components_list)
        
        base_species_in_state = sorted(base_species_in_state)
        
        for base_species in base_species_in_state:
            # Find all species combinations that contain this base species
            matching_species = []
            for species in state_df['Species'].unique():
                components = parse_species_components(species)
                if base_species in components:
                    matching_species.append(species)
            
            if not matching_species:
                continue
            
            # Filter data for these species
            species_df = state_df[state_df['Species'].isin(matching_species)].copy()
            species_df = species_df.sort_values('Sampling_Date')
            
            if len(species_df) == 0:
                continue
            
            print(f'   ├─ {base_species}: {len(matching_species)} species combo(s), '
                  f'{len(species_df)} samples')
            
            # Create plot
            fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
            fig.suptitle(f'Target Distribution Over Time\n'
                        f'State: {state} | Base Species: {base_species}', 
                        fontsize=16, fontweight='bold')
            
            colors = ['#2E7D32', '#8B4513', '#808000', '#1565C0', '#6A1B9A']
            
            for i, target in enumerate(TARGET_COLS):
                ax = axes[i]
                
                # Plot each species combination separately
                for species in matching_species:
                    sp_data = species_df[species_df['Species'] == species]
                    
                    # Scatter plot
                    ax.scatter(sp_data['Sampling_Date'], sp_data[target], 
                              label=species, s=80, alpha=0.7)
                    
                    # Connect with lines if multiple points
                    if len(sp_data) > 1:
                        ax.plot(sp_data['Sampling_Date'], sp_data[target], 
                               alpha=0.3, linewidth=1.5)
                
                ax.set_ylabel(f'{target} (g)', fontsize=11, fontweight='bold')
                ax.set_title(target, fontsize=10, loc='left', pad=5)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add legend if multiple species
                if len(matching_species) > 1:
                    ax.legend(fontsize=8, loc='upper left', 
                             framealpha=0.9, ncol=min(3, len(matching_species)))
                
                # Annotate max value
                if len(species_df) > 0 and species_df[target].max() > 0:
                    max_val = species_df[target].max()
                    max_idx = species_df[target].idxmax()
                    max_date = species_df.loc[max_idx, 'Sampling_Date']
                    max_species = species_df.loc[max_idx, 'Species']
                    
                    ax.annotate(f'Max: {max_val:.1f}g\n({max_species})', 
                               xy=(max_date, max_val), 
                               xytext=(10, 10), textcoords='offset points',
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                               bbox=dict(boxstyle='round,pad=0.5', 
                                       facecolor='yellow', alpha=0.7),
                               fontsize=8)
            
            plt.xlabel('Sampling Date', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save
            safe_name = base_species.replace('/', '_')
            save_path = state_dir / f'{safe_name}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f'\n{"="*80}')
    print(f'✓ Analysis complete. Plots saved to: {OUTPUT_DIR}/')
    print('='*80)
    
    # Generate summary report
    summary_path = OUTPUT_DIR / 'analysis_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['State', 'Base_Species', 'N_Combinations', 'N_Samples', 
                        'Date_Range_Start', 'Date_Range_End'])
        
        for state in states:
            state_df = df[df['State'] == state]
            base_species_in_state = set()
            for components_list in state_df['components']:
                base_species_in_state.update(components_list)
            
            for base_species in sorted(base_species_in_state):
                matching_species = [s for s in state_df['Species'].unique() 
                                   if base_species in parse_species_components(s)]
                species_df = state_df[state_df['Species'].isin(matching_species)]
                
                if len(species_df) > 0:
                    writer.writerow([
                        state,
                        base_species,
                        len(matching_species),
                        len(species_df),
                        species_df['Sampling_Date'].min().strftime('%Y-%m-%d'),
                        species_df['Sampling_Date'].max().strftime('%Y-%m-%d')
                    ])
    
    print(f'✓ Summary report saved to: {summary_path}')

if __name__ == '__main__':
    create_species_plots()
