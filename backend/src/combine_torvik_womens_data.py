"""
Combine Bart Torvik CSV files and tournament team info
Saves to data/women/ subfolder
"""
import pandas as pd
import numpy as np
import os
import csv

print("="*80)
print("COMBINING BART TORVIK WOMEN'S DATA FILES")
print("="*80 + "\n")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')

# Load files
print("Loading files...")

# Read fffinal manually because it's malformed
fffinal_file = os.path.join(data_dir, '2026_fffinal.csv')
fffinal_data = []

with open(fffinal_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    
    for row in reader:
        # First column is team name
        team_name = row[0]
        # Get the stats we need (skip rank columns)
        fffinal_data.append({
            'team': team_name,
            'eFG%': float(row[1]),
            'eFG% Def': float(row[3]),
            'FTR': float(row[5]),
            'FTR Def': float(row[7]),
            'OR%': float(row[9]),
            'DR%': float(row[11]),
            'TO%': float(row[13]),
            'TO% Def.': float(row[15]),
            '3P%': float(row[17]),
            '3pD%': float(row[19]),
            '2p%': float(row[21]),
            '2p%D': float(row[23]),
            '3P rate': float(row[29]),
            '3P rate D': float(row[31])
        })

fffinal = pd.DataFrame(fffinal_data)

team_results = pd.read_csv(os.path.join(data_dir, '2026_team_results.csv'))
tournament_teams = pd.read_csv(os.path.join(data_dir, 'womens_teams_current.csv'))

print(f"✓ fffinal: {fffinal.shape}")
print(f"✓ team_results: {team_results.shape}")
print(f"✓ tournament_teams: {tournament_teams.shape}\n")

# Filter to only tournament teams first
tournament_team_list = tournament_teams['team'].tolist()
team_results_filtered = team_results[team_results['team'].isin(tournament_team_list)]

combined = pd.merge(
    team_results_filtered, 
    fffinal, 
    on='team',
    how='left'
)

print(f"✓ Combined Torvik data: {combined.shape}\n")

# Check for missing teams
tournament_team_names = set(tournament_teams['team'])
merged_team_names = set(combined['team'])
missing_teams = tournament_team_names - merged_team_names

if missing_teams:
    print(f"\n⚠ WARNING: {len(missing_teams)} teams from tournament list not found in Torvik data:")
    for team in missing_teams:
        print(f"  - {team}")
    print()

# Map to model column names
print("Mapping to model column names...")

output = pd.DataFrame()

# Basic info
output['team'] = combined['team']
output['year'] = 2026
output['conf'] = combined['conf']

# From team_results
output['adj_oe'] = combined['adjoe']
output['adj_de'] = combined['adjde']
output['barthag'] = combined['barthag']
output['wab'] = combined['WAB']
output['adj_tempo'] = combined['adjt']

# From fffinal
output['efg_pct'] = combined['eFG%']
output['efgd_pct'] = combined['eFG% Def']
output['tor'] = combined['TO%']
output['tord'] = combined['TO% Def.']
output['orb_pct'] = combined['OR%']
output['drb_pct'] = combined['DR%']
output['ftr'] = combined['FTR']
output['ftrd'] = combined['FTR Def']
output['2p_pct'] = combined['2p%']
output['2pd_pct'] = combined['2p%D']
output['3p_pct'] = combined['3P%']
output['3pd_pct'] = combined['3pD%']
output['3pr'] = combined['3P rate']
output['3prd'] = combined['3P rate D']

# Merge with tournament teams to add seed and region
print("Adding tournament seeds and regions...")
output = pd.merge(
    output,
    tournament_teams[['team', 'seed', 'region']],
    on='team',
    how='left'
)

print(f"✓ Added seeds and regions\n")

# Save
output_file = os.path.join(data_dir, '2026_team_stats.csv')
output.to_csv(output_file, index=False)

print(f"✓ Saved to: {output_file}")
print(f"✓ Total teams: {len(output)}")
print(f"✓ Teams with seeds: {output['seed'].notna().sum()}")
print(f"✓ Teams without seeds: {output['seed'].isna().sum()}")
print(f"✓ Total columns: {len(output.columns)}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)