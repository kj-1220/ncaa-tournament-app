"""
Create 2026 women's matchup dataset with all possible team combinations
"""
import pandas as pd
import os
from itertools import combinations

print("="*80)
print("CREATING 2026 WOMEN'S MATCHUP DATASET")
print("="*80 + "\n")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')

# Load 2026 team stats
print("Loading 2026 team stats...")
teams = pd.read_csv(os.path.join(data_dir, '2026_team_stats.csv'))
print(f"✓ Loaded {len(teams)} teams\n")

# Only use tournament teams (teams with seeds)
tournament_teams = teams[teams['seed'].notna()].copy()
print(f"✓ {len(tournament_teams)} tournament teams\n")

# Invert defensive stats (lower is better → higher is better)
print("Inverting defensive stats...")
tournament_teams['adj_de'] = 200 - tournament_teams['adj_de']
tournament_teams['efgd_pct'] = 100 - tournament_teams['efgd_pct']
tournament_teams['tord'] = 100 - tournament_teams['tord']
tournament_teams['drb_pct'] = 100 - tournament_teams['drb_pct']
tournament_teams['ftrd'] = 100 - tournament_teams['ftrd']
tournament_teams['2pd_pct'] = 100 - tournament_teams['2pd_pct']
tournament_teams['3pd_pct'] = 100 - tournament_teams['3pd_pct']
tournament_teams['3prd'] = 100 - tournament_teams['3prd']
print("✓ Defensive stats inverted\n")

# Create all possible matchups
print("Creating all possible matchups...")
matchups = []

for team1_idx, team1 in tournament_teams.iterrows():
    for team2_idx, team2 in tournament_teams.iterrows():
        if team1['team'] != team2['team']:  # Don't match team with itself
            
            # High seed is lower number (1 is better than 16)
            if team1['seed'] < team2['seed']:
                high_team = team1
                low_team = team2
            else:
                high_team = team2
                low_team = team1
            
            matchup = {
                'year': 2026,
                'high_team': high_team['team'],
                'high_seed': high_team['seed'],
                'high_region': high_team['region'],
                'low_team': low_team['team'],
                'low_seed': low_team['seed'],
                'low_region': low_team['region'],
                
                # Calculate differentials (high - low)
                'wab': high_team['wab'] - low_team['wab'],
                'barthag': high_team['barthag'] - low_team['barthag'],
                'adj_oe': high_team['adj_oe'] - low_team['adj_de'],
                'adj_de': high_team['adj_de'] - low_team['adj_oe'],
                'efg_pct': high_team['efg_pct'] - low_team['efgd_pct'],
                'efgd_pct': high_team['efgd_pct'] - low_team['efg_pct'],
                'tor': high_team['tor'] - low_team['tord'],
                'tord': high_team['tord'] - low_team['tor'],
                'orb_pct': high_team['orb_pct'] - low_team['drb_pct'],
                'drb_pct': high_team['drb_pct'] - low_team['orb_pct'],
                'ftr': high_team['ftr'] - low_team['ftrd'],
                'ftrd': high_team['ftrd'] - low_team['ftr'],
                '2p_pct': high_team['2p_pct'] - low_team['2pd_pct'],
                '2pd_pct': high_team['2pd_pct'] - low_team['2p_pct'],
                '3p_pct': high_team['3p_pct'] - low_team['3pd_pct'],
                '3pd_pct': high_team['3pd_pct'] - low_team['3p_pct'],
                '3pr': high_team['3pr'] - low_team['3prd'],
                '3prd': high_team['3prd'] - low_team['3pr'],
                'adj_tempo': high_team['adj_tempo'] - low_team['adj_tempo']
            }
            
            matchups.append(matchup)

matchups_df = pd.DataFrame(matchups)

print(f"✓ Created {len(matchups_df)} total matchups\n")

# Save
output_file = os.path.join(data_dir, '2026_womens_all_matchups.csv')
matchups_df.to_csv(output_file, index=False)

print(f"✓ Saved to: {output_file}")
print(f"✓ Total matchups: {len(matchups_df)}")
print(f"✓ Total columns: {len(matchups_df.columns)}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nWith {len(tournament_teams)} teams, you have {len(matchups_df)} possible matchups")
print("(Each team plays every other team once as high seed and once as low seed)")