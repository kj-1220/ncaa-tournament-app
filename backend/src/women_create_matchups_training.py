"""
Women's NCAA Tournament - Build Matchup Training Data
  Loads historical team stats + game results, inverts defensive stats,
  builds matchup differentials with corrected inversion handling.

Inputs (in backend/data/women/):
  - women_torvik_historical.csv (team stats by year)
  - women_teams_historical.csv (tournament teams with torvik_id)
  - women_games_historical.csv (game results)

Outputs (in backend/data/women/):
  - women_matchups_training.csv

Usage:
  python3 backend/src/women_create_matchups_training.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')

print("=" * 80)
print("WOMEN'S NCAA TOURNAMENT - BUILD MATCHUP TRAINING DATA")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: Loading data...")

torvik = pd.read_csv(os.path.join(data_dir, 'women_torvik_historical.csv'))
teams = pd.read_csv(os.path.join(data_dir, 'women_teams_historical.csv'))
games = pd.read_csv(os.path.join(data_dir, 'women_games_historical.csv'))

print(f"  Torvik: {torvik.shape}")
print(f"  Teams: {teams.shape}")
print(f"  Games: {games.shape}")

# ============================================================================
# STEP 2: MERGE TEAM STATS
# ============================================================================
print("\nSTEP 2: Merging team stats...")

df = teams.merge(torvik, on='torvik_id', how='left', suffixes=('', '_torvik'))

# Drop duplicate year column if exists
if 'year_torvik' in df.columns:
    df = df.drop(columns=['year_torvik'])
if 'year_x' in df.columns:
    df = df.rename(columns={'year_x': 'year'})

print(f"  Merged: {df.shape}")

# ============================================================================
# STEP 3: INVERT DEFENSIVE STATS (higher = better)
# ============================================================================
print("\nSTEP 3: Inverting defensive stats...")

df['adj_de'] = 200 - df['adj_de']
df['efgd_pct'] = 100 - df['efgd_pct']
df['tord'] = 100 - df['tord']
df['drb_pct'] = 100 - df['drb_pct']
df['ftrd'] = 100 - df['ftrd']
df['2pd_pct'] = 100 - df['2pd_pct']
df['3pd_pct'] = 100 - df['3pd_pct']
df['3prd'] = 100 - df['3prd']

print("  ✓ Defensive stats inverted")

# ============================================================================
# STEP 4: BUILD MATCHUP DIFFERENTIALS
# ============================================================================
print("\nSTEP 4: Building matchup differentials...")

# Stat columns to use from team data (exclude metadata)
team_meta = ['team_id', 'torvik_id', 'year', 'team', 'seed', 'finish', 'region',
             'conference', 'team_torvik']
stat_cols = [c for c in df.columns if c not in team_meta]

# Merge high-bracket team stats
high_stats = df[['team_id'] + stat_cols].copy()
high_stats.columns = ['team_id'] + [f'high_{c}' for c in stat_cols]
matchups = games.merge(high_stats, left_on='high_team_id', right_on='team_id', how='inner')
matchups = matchups.drop(columns=['team_id'])

# Merge low-bracket team stats
low_stats = df[['team_id'] + stat_cols].copy()
low_stats.columns = ['team_id'] + [f'low_{c}' for c in stat_cols]
matchups = matchups.merge(low_stats, left_on='low_team_id', right_on='team_id', how='inner')
matchups = matchups.drop(columns=['team_id'])

print(f"  Matched {len(matchups)} games with team stats")

# Build differential features
# NOTE: Inverted stats use (a + b - constant) instead of (a - b)
matchups_df = pd.DataFrame()

# Game info (these come from the games table, not prefixed)
matchups_df['game_id'] = matchups['game_id'].values
matchups_df['year'] = matchups['year'].values
matchups_df['region'] = matchups['region'].values
matchups_df['round'] = matchups['round'].values
matchups_df['high_bracket_team'] = matchups['high_team_id'].values
matchups_df['low_bracket_team'] = matchups['low_team_id'].values
matchups_df['high_bracket_seed'] = matchups['high_bracket_seed'].values
matchups_df['low_bracket_seed'] = matchups['low_bracket_seed'].values
matchups_df['win'] = matchups['win'].values

# Not inverted — simple subtraction
matchups_df['wab'] = matchups['high_wab'] - matchups['low_wab']
matchups_df['barthag'] = matchups['high_barthag'] - matchups['low_barthag']
matchups_df['adj_tempo'] = matchups['high_adj_tempo'] - matchups['low_adj_tempo']

# Offense vs Defense — corrected for inversions
# inverted-200
matchups_df['adj_oe'] = matchups['high_adj_oe'] + matchups['low_adj_de'] - 200
matchups_df['adj_de'] = matchups['high_adj_de'] + matchups['low_adj_oe'] - 200

# inverted-100
matchups_df['efg_pct'] = matchups['high_efg_pct'] + matchups['low_efgd_pct'] - 100
matchups_df['efgd_pct'] = matchups['high_efgd_pct'] + matchups['low_efg_pct'] - 100
matchups_df['tor'] = matchups['high_tor'] + matchups['low_tord'] - 100
matchups_df['tord'] = matchups['high_tord'] + matchups['low_tor'] - 100
matchups_df['orb_pct'] = matchups['high_orb_pct'] + matchups['low_drb_pct'] - 100
matchups_df['drb_pct'] = matchups['high_drb_pct'] + matchups['low_orb_pct'] - 100
matchups_df['ftr'] = matchups['high_ftr'] + matchups['low_ftrd'] - 100
matchups_df['ftrd'] = matchups['high_ftrd'] + matchups['low_ftr'] - 100
matchups_df['2p_pct'] = matchups['high_2p_pct'] + matchups['low_2pd_pct'] - 100
matchups_df['2pd_pct'] = matchups['high_2pd_pct'] + matchups['low_2p_pct'] - 100
matchups_df['3p_pct'] = matchups['high_3p_pct'] + matchups['low_3pd_pct'] - 100
matchups_df['3pd_pct'] = matchups['high_3pd_pct'] + matchups['low_3p_pct'] - 100
matchups_df['3pr'] = matchups['high_3pr'] + matchups['low_3prd'] - 100
matchups_df['3prd'] = matchups['high_3prd'] + matchups['low_3pr'] - 100

print(f"  ✓ {len(matchups_df)} matchups x {len(matchups_df.columns) - 9} differential features")

# ============================================================================
# STEP 5: SAVE
# ============================================================================
print("\nSTEP 5: Saving...")

output_path = os.path.join(data_dir, 'women_matchups_training.csv')
matchups_df.to_csv(output_path, index=False)
print(f"  ✓ Saved: {output_path}")

# Summary
print(f"\n  Years: {matchups_df['year'].min()}-{matchups_df['year'].max()}")
print(f"  Rounds: {matchups_df['round'].value_counts().to_string()}")
print(f"  Win rate: {matchups_df['win'].mean():.4f}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nNext: Run feature selection, then update women_generate_outputs.py")