"""
Men's NCAA Tournament - Data Preparation
  Loads raw data files (historical + current 2026), merges, engineers features,
  builds matchup differentials.

Inputs (in backend/data/men/):
  Historical: men_teams_historical.csv, men_lineups_5man_historical.csv,
              men_lineups_3man_historical.csv, men_kenpom_ratings_historical.csv,
              men_kenpom_roster_historical.csv, men_torvik_ratings_historical.csv,
              men_torvik_splits_historical.csv, men_games_historical.csv
  Current:    men_teams_current.csv, men_lineups_5man_current.csv,
              men_lineups_3man_current.csv, men_kenpom_ratings_current.csv,
              men_kenpom_roster_current.csv, men_torvik_ratings_current.csv,
              men_torvik_splits_current.csv

Outputs (in backend/data/men/):
  - men_2026_teams_training.csv
  - men_2026_matchups_training.csv

Usage:
  python3 backend/src/men_create_data.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'men')

print("=" * 80)
print("MEN'S NCAA TOURNAMENT - DATA PREPARATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD & MERGE RAW DATA
# ============================================================================
print("\nSTEP 1: Loading and merging data...")

# --- HISTORICAL ---
teams_hist = pd.read_csv(os.path.join(data_dir, 'men_teams_historical.csv'))
teams_hist = teams_hist[teams_hist['year'] >= 2015]
print(f"  Teams historical: {teams_hist.shape}")

five_man_hist = pd.read_csv(os.path.join(data_dir, 'men_lineups_5man_historical.csv'))
three_man_hist = pd.read_csv(os.path.join(data_dir, 'men_lineups_3man_historical.csv'))

kp_ratings_hist = pd.read_csv(os.path.join(data_dir, 'men_kenpom_ratings_historical.csv'))
kp_ratings_hist = kp_ratings_hist[['kenpom_id', 'kenpom_off', 'kenpom_def', 'kenpom_rtg']]

kp_roster_hist = pd.read_csv(os.path.join(data_dir, 'men_kenpom_roster_historical.csv'))
kp_roster_hist = kp_roster_hist.drop(columns=['year', 'team'])

bt_ratings_hist = pd.read_csv(os.path.join(data_dir, 'men_torvik_ratings_historical.csv'))
bt_ratings_hist = bt_ratings_hist.drop(columns=['year', 'team'])

bt_splits_hist = pd.read_csv(os.path.join(data_dir, 'men_torvik_splits_historical.csv'))
bt_splits_hist = bt_splits_hist.drop(columns=['year', 'team'])

# Merge historical
df_hist = teams_hist.merge(five_man_hist, on='team_id', how='inner')
df_hist = df_hist.merge(three_man_hist, on='team_id', how='inner')
df_hist = df_hist.merge(kp_ratings_hist, on='kenpom_id', how='inner')
df_hist = df_hist.merge(kp_roster_hist, on='kenpom_id', how='inner')
df_hist = df_hist.merge(bt_ratings_hist, on='torvik_id', how='inner')
df_hist = df_hist.merge(bt_splits_hist, on='torvik_id', how='inner')
print(f"  Historical merged: {df_hist.shape}")

# --- CURRENT 2026 ---
teams_cur = pd.read_csv(os.path.join(data_dir, 'men_teams_current.csv'))
print(f"  Teams current: {teams_cur.shape}")

five_man_cur = pd.read_csv(os.path.join(data_dir, 'men_lineups_5man_current.csv'))
three_man_cur = pd.read_csv(os.path.join(data_dir, 'men_lineups_3man_current.csv'))

kp_ratings_cur = pd.read_csv(os.path.join(data_dir, 'men_kenpom_ratings_current.csv'))
# kenpom_id already exists in file
kp_ratings_cur = kp_ratings_cur[['kenpom_id', 'kenpom_off', 'kenpom_def', 'kenpom_rtg']]

kp_roster_cur = pd.read_csv(os.path.join(data_dir, 'men_kenpom_roster_current.csv'))
# kenpom_id already exists in file
kp_roster_cur = kp_roster_cur.drop(columns=['year', 'team'])

bt_ratings_cur = pd.read_csv(os.path.join(data_dir, 'men_torvik_ratings_current.csv'))
# torvik_id does NOT exist — create it from year + team
bt_ratings_cur['torvik_id'] = bt_ratings_cur['year'].astype(str) + ' ' + bt_ratings_cur['team']
bt_ratings_cur = bt_ratings_cur.drop(columns=['year', 'team'])

bt_splits_cur = pd.read_csv(os.path.join(data_dir, 'men_torvik_splits_current.csv'))
# torvik_id already exists in file
bt_splits_cur = bt_splits_cur.drop(columns=['year', 'team'])

# Current teams need kenpom_id and torvik_id for merging
teams_cur['kenpom_id'] = teams_cur['team_id']
teams_cur['torvik_id'] = teams_cur['team_id']
# Add missing columns that historical has
teams_cur['finish'] = 'TBD'
teams_cur['weekend'] = 0
teams_cur['conference'] = ''

# Merge current
df_cur = teams_cur.merge(five_man_cur, on='team_id', how='inner')
df_cur = df_cur.merge(three_man_cur, on='team_id', how='inner')
df_cur = df_cur.merge(kp_ratings_cur, on='kenpom_id', how='inner')
df_cur = df_cur.merge(kp_roster_cur, on='kenpom_id', how='inner')
df_cur = df_cur.merge(bt_ratings_cur, on='torvik_id', how='inner')
df_cur = df_cur.merge(bt_splits_cur, on='torvik_id', how='inner')
print(f"  Current merged: {df_cur.shape}")

# --- COMBINE ---
df = pd.concat([df_hist, df_cur], ignore_index=True)
print(f"\n  Combined: {df.shape[0]} teams x {df.shape[1]} columns")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\nSTEP 2: Feature engineering...")

# --------------------------------------------------------------------------
# INVERSIONS
# --------------------------------------------------------------------------
for col in ['efgd%', '2p%d', '3p%d', 'ft%d', 'def_dunk_fg%', 'def_close2_fg%',
            'def_far2_fg%', 'def_3pt_fg%', 'ast%d']:
    df[col] = 100 - df[col]

df['tor'] = 100 - df['tor']
df['blked%'] = 100 - df['blked%']
df['kenpom_def'] = 200 - df['kenpom_def']
df['torvik_def'] = 200 - df['torvik_def']
df['ftrd'] = 100 - df['ftrd']
df['5man_dprpg'] = 50 - df['5man_dprpg']
df['3man_dprpg'] = 50 - df['3man_dprpg']

# --------------------------------------------------------------------------
# 1. EFFICIENCY MARGINS
# --------------------------------------------------------------------------
df['net_efg_margin'] = df['efg%'] - df['efgd%']
df['net_turnover_margin'] = df['tord'] - df['tor']
df['net_rebounding_margin'] = (df['orb%'] + df['drb%']) - 100
df['net_ftr_margin'] = df['ftr'] - df['ftrd']

df['def_net_efg_margin'] = df['efgd%'] - df['efg%']
df['def_net_turnover_margin'] = df['tor'] - df['tord']
df['def_net_rebounding_margin'] = 100 - (df['orb%'] + df['drb%'])
df['def_net_ftr_margin'] = df['ftrd'] - df['ftr']

# --------------------------------------------------------------------------
# 2. SHOT QUALITY
# --------------------------------------------------------------------------
df['rim_efficiency'] = (df['off_dunk_fg%'] * df['off_dunk_share'] / 100) + \
                       (df['off_close2_fg%'] * df['off_close2_share'] / 100)
df['perimeter_efficiency'] = (df['off_far2_fg%'] * df['off_far2_share'] / 100) + \
                              (df['off_3pt_fg%'] * df['off_3pt_share'] / 100)
df['three_point_volume_efficiency'] = df['off_3pt_fg%'] * df['off_3pt_share'] / 100
df['shot_quality_variance'] = df[['off_dunk_fg%', 'off_close2_fg%',
                                   'off_far2_fg%', 'off_3pt_fg%']].std(axis=1)

df['def_rim_efficiency'] = (df['def_dunk_fg%'] * df['def_dunk_share'] / 100) + \
                            (df['def_close2_fg%'] * df['def_close2_share'] / 100)
df['def_perimeter_efficiency'] = (df['def_far2_fg%'] * df['def_far2_share'] / 100) + \
                                  (df['def_3pt_fg%'] * df['def_3pt_share'] / 100)
df['def_three_point_volume_efficiency'] = df['def_3pt_fg%'] * df['def_3pt_share'] / 100
df['def_shot_quality_variance'] = df[['def_dunk_fg%', 'def_close2_fg%',
                                       'def_far2_fg%', 'def_3pt_fg%']].std(axis=1)

# --------------------------------------------------------------------------
# 3. SHOT SELECTION
# --------------------------------------------------------------------------
df['rim_to_three_ratio'] = (df['off_dunk_share'] + df['off_close2_share']) / \
                            (df['off_3pt_share'] + 0.001)
df['mid_range_reliance'] = 50 - df['off_far2_share']
df['paint_touch_rate'] = df['off_dunk_share'] + df['off_close2_share']

df['def_rim_to_three_ratio'] = (df['def_dunk_share'] + df['def_close2_share']) / \
                                (df['def_3pt_share'] + 0.001)
df['def_mid_range_reliance'] = df['def_far2_share']
df['def_paint_touch_rate'] = 100 - (df['def_dunk_share'] + df['def_close2_share'])

# --------------------------------------------------------------------------
# 4. DEPTH & ROTATION
# --------------------------------------------------------------------------
pts_cols = ['pts_from_5', 'pts_from_4', 'pts_from_3', 'pts_from_2', 'pts_from_1']
df['top5_scoring_concentration'] = 100 - df[pts_cols].sum(axis=1)
df['top5_rebounding_concentration'] = 100 - (df['or_from_5'] + df['or_from_4'] +
                                              df['or_from_3'] + df['or_from_2'] +
                                              df['or_from_1'])
df['top5_def_rebounding_concentration'] = 100 - (df['dr_from_5'] + df['dr_from_4'] +
                                                   df['dr_from_3'] + df['dr_from_2'] +
                                                   df['dr_from_1'])
df['bench_scoring_ratio'] = df['bench']
df['rotation_balance'] = 100 - df[pts_cols].std(axis=1)

# --------------------------------------------------------------------------
# 5. PACE & TEMPO
# --------------------------------------------------------------------------
df['tempo_advantage'] = df['adj_tempo'] - 68
df['effective_possession_rate'] = (df['tor'] / 100) * df['efg%']
df['def_effective_possession_rate'] = (df['tord'] / 100) * df['efgd%']

# --------------------------------------------------------------------------
# 6. VERSATILITY
# --------------------------------------------------------------------------
off_metrics = df[['2p%', '3p%', 'ast%', 'orb%']].values
df['offensive_versatility_score'] = 100 / (np.std(off_metrics, axis=1) /
                                            (np.mean(off_metrics, axis=1) + 0.001) + 1)
df['assist_to_usage_ratio'] = df['ast%'] / (100 - df['tord'] + 0.001)

def_metrics = df[['2p%d', '3p%d', 'ast%d', 'drb%']].values
df['defensive_versatility_score'] = 100 / (np.std(def_metrics, axis=1) /
                                            (np.mean(def_metrics, axis=1) + 0.001) + 1)
df['def_assist_suppression'] = df['ast%d'] / (100 - df['tor'] + 0.001)

df['size_speed_index'] = (df['height'] * df['adj_tempo']) / 100
df['def_size_speed_index'] = df['size_speed_index']

# --------------------------------------------------------------------------
# 7. CLUTCH & PRESSURE
# --------------------------------------------------------------------------
df['free_throw_advantage'] = (df['ft%'] * df['ftr']) - (df['ft%d'] * df['ftrd'])
df['block_efficiency'] = df['blk%'] / (df['blked%'] + 0.001)
df['experience_weighted_production'] = df['5man_obpm'] * df['experience']

df['def_free_throw_advantage'] = (df['ft%d'] * df['ftrd']) - (df['ft%'] * df['ftr'])
df['def_block_efficiency'] = df['blked%'] / (df['blk%'] + 0.001)
df['def_experience_impact'] = df['5man_dbpm'] * df['experience']

# --------------------------------------------------------------------------
# 8. ADVANCED COMPOSITES
# --------------------------------------------------------------------------
df['four_factors_composite'] = (0.40 * df['efg%']) + (0.25 * df['tor']) + \
                                (0.20 * df['orb%']) + (0.15 * df['ftr'])
df['def_four_factors_composite'] = (0.40 * df['efgd%']) + (0.25 * df['tord']) + \
                                    (0.20 * df['drb%']) + (0.15 * df['ftrd'])

df['elite_outcome_probability'] = 0
df['elite_outcome_probability'] += (df['efg%'] >= df['efg%'].quantile(0.75)).astype(int)
df['elite_outcome_probability'] += (df['tor'] >= df['tor'].quantile(0.75)).astype(int)
df['elite_outcome_probability'] += (df['orb%'] >= df['orb%'].quantile(0.75)).astype(int)
df['elite_outcome_probability'] += (df['3p%'] >= df['3p%'].quantile(0.75)).astype(int)
df['elite_outcome_probability'] += (df['drb%'] >= df['drb%'].quantile(0.75)).astype(int)

# --------------------------------------------------------------------------
# 9. SYNERGY
# --------------------------------------------------------------------------
shooting_cols = ['off_dunk_fg%', 'off_close2_fg%', 'off_far2_fg%', 'off_3pt_fg%']
df['shooting_variance_resilience'] = 100 / (df[shooting_cols].std(axis=1) /
                                             (df[shooting_cols].mean(axis=1) + 0.001) + 1)
df['lineup_depth_quality'] = -(df['5man_obpm'] - df['3man_obpm'])
df['def_lineup_depth_quality'] = -(df['5man_dbpm'] - df['3man_dbpm'])

balance_raw = df['kenpom_off'] / (df['kenpom_def'] + 0.001)
df['offense_defense_balance'] = 10 - abs(balance_raw - 1.25) * 10

print(f"  Complete: {df.shape[0]} teams x {df.shape[1]} columns")

# ============================================================================
# STEP 3: SAVE TEAMS TRAINING DATA
# ============================================================================
print("\nSTEP 3: Saving teams training data...")

teams_path = os.path.join(data_dir, 'men_2026_teams_training.csv')
df.to_csv(teams_path, index=True)
print(f"  ✓ Saved: {teams_path}")
print(f"    {df.shape[0]} teams x {df.shape[1]} columns")

# ============================================================================
# STEP 4: BUILD MATCHUP DIFFERENTIALS
# ============================================================================
print("\nSTEP 4: Building matchup differentials...")

games = pd.read_csv(os.path.join(data_dir, 'men_games_historical.csv'))
print(f"  Games: {games.shape}")

# Merge high-bracket team stats
matchups = games.merge(df, left_on='high_bracket_team', right_on='team_id',
                       how='inner', suffixes=('', '_DROP'))
# Drop duplicate columns from first merge
matchups = matchups[[c for c in matchups.columns if not c.endswith('_DROP')]]
for col in df.columns:
    if col in matchups.columns and col not in games.columns:
        matchups.rename(columns={col: f'high_{col}'}, inplace=True)

# Merge low-bracket team stats
matchups = matchups.merge(df, left_on='low_bracket_team', right_on='team_id',
                          how='inner', suffixes=('', '_DROP'))
# Drop duplicate columns from second merge
matchups = matchups[[c for c in matchups.columns if not c.endswith('_DROP')]]
for col in df.columns:
    if col in matchups.columns and col not in games.columns and not col.startswith('high_'):
        matchups.rename(columns={col: f'low_{col}'}, inplace=True)

# Build differentials
matchups_df = pd.DataFrame()

# Game info
matchups_df['game_id'] = matchups['game_id']
matchups_df['year'] = matchups['year']
matchups_df['region'] = matchups['region']
matchups_df['round'] = matchups['round']
matchups_df['high_bracket_team'] = matchups['high_bracket_team']
matchups_df['low_bracket_team'] = matchups['low_bracket_team']
matchups_df['high_bracket_seed'] = matchups['high_bracket_seed']
matchups_df['low_bracket_seed'] = matchups['low_bracket_seed']
matchups_df['win'] = matchups['win']

# COMPOSITES (team vs team)
matchups_df['5man_bpm'] = matchups['high_5man_bpm'] - matchups['low_5man_bpm']
matchups_df['3man_bpm'] = matchups['high_3man_bpm'] - matchups['low_3man_bpm']
matchups_df['wab'] = matchups['high_wab'] - matchups['low_wab']
matchups_df['kenpom_rtg'] = matchups['high_kenpom_rtg'] - matchups['low_kenpom_rtg']
matchups_df['torvik_rtg'] = matchups['high_torvik_rtg'] - matchups['low_torvik_rtg']

# LINEUP PRODUCTION RATES - HIGH OFFENSE vs LOW DEFENSE
matchups_df['5man_prpg'] = matchups['high_5man_prpg!'] - matchups['low_5man_dprpg']
matchups_df['3man_prpg'] = matchups['high_3man_prpg!'] - matchups['low_3man_dprpg']

# LINEUP PRODUCTION RATES - HIGH DEFENSE vs LOW OFFENSE
matchups_df['5man_dprpg'] = matchups['high_5man_dprpg'] - matchups['low_5man_prpg!']
matchups_df['3man_dprpg'] = matchups['high_3man_dprpg'] - matchups['low_3man_prpg!']

# TEAM vs TEAM
matchups_df['size'] = matchups['high_size'] - matchups['low_size']
matchups_df['height'] = matchups['high_height'] - matchups['low_height']
matchups_df['experience'] = matchups['high_experience'] - matchups['low_experience']
matchups_df['bench'] = matchups['high_bench'] - matchups['low_bench']
matchups_df['raw_tempo'] = matchups['high_raw_tempo'] - matchups['low_raw_tempo']
matchups_df['adj_tempo'] = matchups['high_adj_tempo'] - matchups['low_adj_tempo']
matchups_df['3pr'] = matchups['high_3pr'] - matchups['low_3prd']
matchups_df['3prd'] = matchups['high_3prd'] - matchups['low_3pr']

# SHOT SHARES - HIGH OFFENSE vs LOW DEFENSE
matchups_df['off_dunk_share'] = matchups['high_off_dunk_share'] - matchups['low_def_dunk_share']
matchups_df['off_close2_share'] = matchups['high_off_close2_share'] - matchups['low_def_close2_share']
matchups_df['off_far2_share'] = matchups['high_off_far2_share'] - matchups['low_def_far2_share']
matchups_df['off_3pt_share'] = matchups['high_off_3pt_share'] - matchups['low_def_3pt_share']

# SHOT SHARES - HIGH DEFENSE vs LOW OFFENSE
matchups_df['def_dunk_share'] = matchups['high_def_dunk_share'] - matchups['low_off_dunk_share']
matchups_df['def_close2_share'] = matchups['high_def_close2_share'] - matchups['low_off_close2_share']
matchups_df['def_far2_share'] = matchups['high_def_far2_share'] - matchups['low_off_far2_share']
matchups_df['def_3pt_share'] = matchups['high_def_3pt_share'] - matchups['low_off_3pt_share']

# HIGH OFFENSE vs LOW DEFENSE
matchups_df['5man_obpm'] = matchups['high_5man_obpm'] - matchups['low_5man_dbpm']
matchups_df['3man_obpm'] = matchups['high_3man_obpm'] - matchups['low_3man_dbpm']
matchups_df['kenpom_off'] = matchups['high_kenpom_off'] - matchups['low_kenpom_def']
matchups_df['torvik_off'] = matchups['high_torvik_off'] - matchups['low_torvik_def']
matchups_df['efg_pct'] = matchups['high_efg%'] - matchups['low_efgd%']
matchups_df['2p_pct'] = matchups['high_2p%'] - matchups['low_2p%d']
matchups_df['3p_pct'] = matchups['high_3p%'] - matchups['low_3p%d']
matchups_df['ft_pct'] = matchups['high_ft%'] - matchups['low_ft%d']
matchups_df['ftr'] = matchups['high_ftr'] - matchups['low_ftrd']
matchups_df['tor'] = matchups['high_tor'] - matchups['low_tord']
matchups_df['orb_pct'] = matchups['high_orb%'] - matchups['low_drb%']
matchups_df['ast_pct'] = matchups['high_ast%'] - matchups['low_ast%d']
matchups_df['blk_pct'] = matchups['high_blk%'] - matchups['low_blked%']
matchups_df['off_dunk_fg_pct'] = matchups['high_off_dunk_fg%'] - matchups['low_def_dunk_fg%']
matchups_df['off_close2_fg_pct'] = matchups['high_off_close2_fg%'] - matchups['low_def_close2_fg%']
matchups_df['off_far2_fg_pct'] = matchups['high_off_far2_fg%'] - matchups['low_def_far2_fg%']
matchups_df['off_3pt_fg_pct'] = matchups['high_off_3pt_fg%'] - matchups['low_def_3pt_fg%']

# HIGH DEFENSE vs LOW OFFENSE
matchups_df['5man_dbpm'] = matchups['high_5man_dbpm'] - matchups['low_5man_obpm']
matchups_df['3man_dbpm'] = matchups['high_3man_dbpm'] - matchups['low_3man_obpm']
matchups_df['kenpom_def'] = matchups['high_kenpom_def'] - matchups['low_kenpom_off']
matchups_df['torvik_def'] = matchups['high_torvik_def'] - matchups['low_torvik_off']
matchups_df['efgd_pct'] = matchups['high_efgd%'] - matchups['low_efg%']
matchups_df['2pd_pct'] = matchups['high_2p%d'] - matchups['low_2p%']
matchups_df['3pd_pct'] = matchups['high_3p%d'] - matchups['low_3p%']
matchups_df['ftd_pct'] = matchups['high_ft%d'] - matchups['low_ft%']
matchups_df['ftrd'] = matchups['high_ftrd'] - matchups['low_ftr']
matchups_df['tord'] = matchups['high_tord'] - matchups['low_tor']
matchups_df['drb_pct'] = matchups['high_drb%'] - matchups['low_orb%']
matchups_df['astd_pct'] = matchups['high_ast%d'] - matchups['low_ast%']
matchups_df['blked_pct'] = matchups['high_blked%'] - matchups['low_blk%']
matchups_df['def_dunk_fg_pct'] = matchups['high_def_dunk_fg%'] - matchups['low_off_dunk_fg%']
matchups_df['def_close2_fg_pct'] = matchups['high_def_close2_fg%'] - matchups['low_off_close2_fg%']
matchups_df['def_far2_fg_pct'] = matchups['high_def_far2_fg%'] - matchups['low_off_far2_fg%']
matchups_df['def_3pt_fg_pct'] = matchups['high_def_3pt_fg%'] - matchups['low_off_3pt_fg%']

# NEW FEATURES - HIGH OFFENSE vs LOW DEFENSE
matchups_df['net_efg_margin'] = matchups['high_net_efg_margin'] - matchups['low_def_net_efg_margin']
matchups_df['net_turnover_margin'] = matchups['high_net_turnover_margin'] - matchups['low_def_net_turnover_margin']
matchups_df['net_rebounding_margin'] = matchups['high_net_rebounding_margin'] - matchups['low_def_net_rebounding_margin']
matchups_df['net_ftr_margin'] = matchups['high_net_ftr_margin'] - matchups['low_def_net_ftr_margin']
matchups_df['rim_efficiency'] = matchups['high_rim_efficiency'] - matchups['low_def_rim_efficiency']
matchups_df['perimeter_efficiency'] = matchups['high_perimeter_efficiency'] - matchups['low_def_perimeter_efficiency']
matchups_df['three_point_volume_efficiency'] = matchups['high_three_point_volume_efficiency'] - matchups['low_def_three_point_volume_efficiency']
matchups_df['shot_quality_variance'] = matchups['high_shot_quality_variance'] - matchups['low_def_shot_quality_variance']
matchups_df['rim_to_three_ratio'] = matchups['high_rim_to_three_ratio'] - matchups['low_def_rim_to_three_ratio']
matchups_df['mid_range_reliance'] = matchups['high_mid_range_reliance'] - matchups['low_def_mid_range_reliance']
matchups_df['paint_touch_rate'] = matchups['high_paint_touch_rate'] - matchups['low_def_paint_touch_rate']
matchups_df['top5_scoring_concentration'] = matchups['high_top5_scoring_concentration'] - matchups['low_top5_scoring_concentration']
matchups_df['top5_rebounding_concentration'] = matchups['high_top5_rebounding_concentration'] - matchups['low_top5_rebounding_concentration']
matchups_df['top5_def_rebounding_concentration'] = matchups['high_top5_def_rebounding_concentration'] - matchups['low_top5_def_rebounding_concentration']
matchups_df['bench_scoring_ratio'] = matchups['high_bench_scoring_ratio'] - matchups['low_bench_scoring_ratio']
matchups_df['rotation_balance'] = matchups['high_rotation_balance'] - matchups['low_rotation_balance']
matchups_df['tempo_advantage'] = matchups['high_tempo_advantage'] - matchups['low_tempo_advantage']
matchups_df['effective_possession_rate'] = matchups['high_effective_possession_rate'] - matchups['low_def_effective_possession_rate']
matchups_df['offensive_versatility_score'] = matchups['high_offensive_versatility_score'] - matchups['low_defensive_versatility_score']
matchups_df['size_speed_index'] = matchups['high_size_speed_index'] - matchups['low_size_speed_index']
matchups_df['assist_to_usage_ratio'] = matchups['high_assist_to_usage_ratio'] - matchups['low_def_assist_suppression']
matchups_df['free_throw_advantage'] = matchups['high_free_throw_advantage'] - matchups['low_def_free_throw_advantage']
matchups_df['block_efficiency'] = matchups['high_block_efficiency'] - matchups['low_def_block_efficiency']
matchups_df['experience_weighted_production'] = matchups['high_experience_weighted_production'] - matchups['low_def_experience_impact']
matchups_df['four_factors_composite'] = matchups['high_four_factors_composite'] - matchups['low_def_four_factors_composite']
matchups_df['elite_outcome_probability'] = matchups['high_elite_outcome_probability'] - matchups['low_elite_outcome_probability']
matchups_df['offense_defense_balance'] = matchups['high_offense_defense_balance'] - matchups['low_offense_defense_balance']
matchups_df['shooting_variance_resilience'] = matchups['high_shooting_variance_resilience'] - matchups['low_shooting_variance_resilience']
matchups_df['lineup_depth_quality'] = matchups['high_lineup_depth_quality'] - matchups['low_def_lineup_depth_quality']

# NEW FEATURES - HIGH DEFENSE vs LOW OFFENSE
matchups_df['def_net_efg_margin'] = matchups['high_def_net_efg_margin'] - matchups['low_net_efg_margin']
matchups_df['def_net_turnover_margin'] = matchups['high_def_net_turnover_margin'] - matchups['low_net_turnover_margin']
matchups_df['def_net_rebounding_margin'] = matchups['high_def_net_rebounding_margin'] - matchups['low_net_rebounding_margin']
matchups_df['def_net_ftr_margin'] = matchups['high_def_net_ftr_margin'] - matchups['low_net_ftr_margin']
matchups_df['def_rim_efficiency'] = matchups['high_def_rim_efficiency'] - matchups['low_rim_efficiency']
matchups_df['def_perimeter_efficiency'] = matchups['high_def_perimeter_efficiency'] - matchups['low_perimeter_efficiency']
matchups_df['def_three_point_volume_efficiency'] = matchups['high_def_three_point_volume_efficiency'] - matchups['low_three_point_volume_efficiency']
matchups_df['def_shot_quality_variance'] = matchups['high_def_shot_quality_variance'] - matchups['low_shot_quality_variance']
matchups_df['def_rim_to_three_ratio'] = matchups['high_def_rim_to_three_ratio'] - matchups['low_rim_to_three_ratio']
matchups_df['def_mid_range_reliance'] = matchups['high_def_mid_range_reliance'] - matchups['low_mid_range_reliance']
matchups_df['def_paint_touch_rate'] = matchups['high_def_paint_touch_rate'] - matchups['low_paint_touch_rate']
matchups_df['def_effective_possession_rate'] = matchups['high_def_effective_possession_rate'] - matchups['low_effective_possession_rate']
matchups_df['defensive_versatility_score'] = matchups['high_defensive_versatility_score'] - matchups['low_offensive_versatility_score']
matchups_df['def_size_speed_index'] = matchups['high_def_size_speed_index'] - matchups['low_size_speed_index']
matchups_df['def_assist_suppression'] = matchups['high_def_assist_suppression'] - matchups['low_assist_to_usage_ratio']
matchups_df['def_free_throw_advantage'] = matchups['high_def_free_throw_advantage'] - matchups['low_free_throw_advantage']
matchups_df['def_block_efficiency'] = matchups['high_def_block_efficiency'] - matchups['low_block_efficiency']
matchups_df['def_experience_impact'] = matchups['high_def_experience_impact'] - matchups['low_experience_weighted_production']
matchups_df['def_four_factors_composite'] = matchups['high_def_four_factors_composite'] - matchups['low_four_factors_composite']
matchups_df['def_lineup_depth_quality'] = matchups['high_def_lineup_depth_quality'] - matchups['low_lineup_depth_quality']

print(f"  Matchups: {matchups_df.shape[0]} games x {len(matchups_df.columns) - 9} differential features")

# ============================================================================
# STEP 5: SAVE MATCHUPS
# ============================================================================
print("\nSTEP 5: Saving matchups...")

matchups_path = os.path.join(data_dir, 'men_2026_matchups_training.csv')
matchups_df.to_csv(matchups_path, index=False)
print(f"  ✓ Saved: {matchups_path}")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)
print(f"\nOutputs:")
print(f"  1. {teams_path}")
print(f"  2. {matchups_path}")
print(f"\nNext: python3 backend/src/men_generate_outputs.py")