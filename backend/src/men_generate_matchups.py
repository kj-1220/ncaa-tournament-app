"""
Men's NCAA Tournament - Matchup Models & Probabilities (Stage 2)
  Trains 4 round-specific models, applies Platt Scaling,
  calculates advancement probabilities and bracket value.

  Round 1:   XGBoost (23 features)
  Round 2:   Random Forest (30 features)
  Weekend 2: Logistic Regression (9 features) - Sweet 16 + Elite Eight
  Weekend 3: Logistic Regression (7 features) - Final Four + Championship

Inputs (in backend/data/men/):
  - men_2026_matchups_training.csv (from men_create_data.py)
  - men_teams_output.csv (from men_generate_outputs.py)
  - bracket_template.csv

Outputs (in backend/data/men/):
  - men_teams_output.csv (updated with probabilities + bracket_value)
  - men_matchups_output.csv (all matchups with win probabilities)

Usage:
  python3 backend/src/men_generate_matchups.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ============================================================================
# PATHS
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'men')

print("=" * 80)
print("MEN'S NCAA TOURNAMENT - MATCHUP MODELS & PROBABILITIES")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: Loading data...")

matchups = pd.read_csv(os.path.join(data_dir, 'men_2026_matchups_training.csv'))
teams_output = pd.read_csv(os.path.join(data_dir, 'men_teams_output.csv'))

print(f"  Matchups: {len(matchups)} games")
print(f"  Teams: {len(teams_output)} teams")

# ============================================================================
# STEP 2: TRAIN ROUND 1 MODEL (XGBoost)
# ============================================================================
print("\nSTEP 2: Training Round 1 model (XGBoost)...")

r1_features = [
    '5man_bpm', 'torvik_rtg', 'elite_outcome_probability', '5man_dbpm',
    'paint_touch_rate', 'bench', 'def_close2_fg_pct', 'bench_scoring_ratio',
    'adj_tempo', 'experience', 'lineup_depth_quality', '5man_obpm',
    'height', 'off_close2_fg_pct', 'def_paint_touch_rate', 'tempo_advantage',
    'def_mid_range_reliance', 'torvik_off', 'top5_rebounding_concentration',
    'def_3pt_share', 'mid_range_reliance', '3man_prpg', 'offense_defense_balance'
]

df_r1 = matchups[matchups['round'] == 'First Round'].copy()
X_r1 = df_r1[r1_features].fillna(df_r1[r1_features].median())
y_r1 = df_r1['win']

X_temp, X_test_r1, y_temp, y_test_r1 = train_test_split(
    X_r1, y_r1, test_size=0.15, random_state=42, stratify=y_r1
)
X_train_r1, X_val_r1, y_train_r1, y_val_r1 = train_test_split(
    X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
)

scaler_r1 = StandardScaler()
X_train_r1_s = scaler_r1.fit_transform(X_train_r1)
X_val_r1_s = scaler_r1.transform(X_val_r1)
X_test_r1_s = scaler_r1.transform(X_test_r1)

xgb_model = XGBClassifier(
    subsample=0.3, scale_pos_weight=1.0, reg_lambda=5.0, reg_alpha=0.1,
    n_estimators=250, min_child_weight=3, max_depth=4, max_delta_step=0,
    learning_rate=0.02, gamma=0.1, colsample_bytree=0.5,
    colsample_bynode=0.4, colsample_bylevel=0.5,
    tree_method='hist', random_state=42, eval_metric='logloss',
    enable_categorical=False
)
xgb_model.fit(X_train_r1_s, y_train_r1,
              eval_set=[(X_train_r1_s, y_train_r1), (X_val_r1_s, y_val_r1)],
              verbose=False)

# Platt Scaling
r1_calibrated = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
r1_calibrated.fit(X_val_r1_s, y_val_r1)

from sklearn.metrics import accuracy_score
r1_acc = accuracy_score(y_test_r1, xgb_model.predict(X_test_r1_s))
print(f"  ✓ Round 1: {len(df_r1)} games, test accuracy: {r1_acc:.4f}")

# ============================================================================
# STEP 3: TRAIN ROUND 2 MODEL (Random Forest)
# ============================================================================
print("\nSTEP 3: Training Round 2 model (Random Forest)...")

r2_features = [
    '5man_bpm', 'kenpom_rtg', 'def_lineup_depth_quality', 'torvik_rtg',
    'kenpom_off', '3man_bpm', '5man_dbpm', 'lineup_depth_quality',
    '5man_obpm', 'kenpom_def', 'experience_weighted_production', 'wab',
    'defensive_versatility_score', 'four_factors_composite', 'torvik_def',
    '3man_obpm', 'off_3pt_fg_pct', 'def_four_factors_composite',
    'torvik_off', 'size_speed_index', 'def_size_speed_index',
    'def_rim_efficiency', 'def_experience_impact', 'efg_pct',
    'offensive_versatility_score', 'efgd_pct', '3pd_pct',
    'def_3pt_fg_pct', 'blk_pct', 'blked_pct'
]

df_r2 = matchups[matchups['round'] == 'Second Round'].copy()
X_r2 = df_r2[r2_features].fillna(df_r2[r2_features].median())
y_r2 = df_r2['win']

X_train_r2, X_test_r2, y_train_r2, y_test_r2 = train_test_split(
    X_r2, y_r2, test_size=0.2, random_state=42, stratify=y_r2
)

scaler_r2 = StandardScaler()
X_train_r2_s = scaler_r2.fit_transform(X_train_r2)
X_test_r2_s = scaler_r2.transform(X_test_r2)

rf_model = RandomForestClassifier(
    n_estimators=300, min_samples_split=5, min_samples_leaf=1,
    max_samples=0.65, max_features=0.4, max_depth=12, bootstrap=True,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_r2_s, y_train_r2)

# Platt Scaling
r2_calibrated = CalibratedClassifierCV(rf_model, method='sigmoid', cv='prefit')
r2_calibrated.fit(X_test_r2_s, y_test_r2)

r2_acc = accuracy_score(y_test_r2, rf_model.predict(X_test_r2_s))
print(f"  ✓ Round 2: {len(df_r2)} games, test accuracy: {r2_acc:.4f}")

# ============================================================================
# STEP 4: TRAIN WEEKEND 2 MODEL (Logistic Regression)
# ============================================================================
print("\nSTEP 4: Training Weekend 2 model (Logistic Regression)...")

w2_features = [
    '5man_bpm', 'wab', 'torvik_rtg', 'elite_outcome_probability',
    '5man_dbpm', '3man_prpg', 'four_factors_composite',
    'lineup_depth_quality', '3man_bpm'
]

df_w2 = matchups[matchups['round'].isin(['Sweet 16', 'Elite Eight'])].copy()
X_w2 = df_w2[w2_features].fillna(df_w2[w2_features].median())
y_w2 = df_w2['win']

X_train_w2, X_test_w2, y_train_w2, y_test_w2 = train_test_split(
    X_w2, y_w2, test_size=0.30, random_state=42, stratify=y_w2
)

scaler_w2 = StandardScaler()
X_train_w2_s = scaler_w2.fit_transform(X_train_w2)
X_test_w2_s = scaler_w2.transform(X_test_w2)

lr_w2 = LogisticRegression(max_iter=1000, random_state=42)
lr_w2.fit(X_train_w2_s, y_train_w2)

# Platt Scaling
w2_calibrated = CalibratedClassifierCV(lr_w2, method='sigmoid', cv='prefit')
w2_calibrated.fit(X_test_w2_s, y_test_w2)

w2_acc = accuracy_score(y_test_w2, lr_w2.predict(X_test_w2_s))
print(f"  ✓ Weekend 2: {len(df_w2)} games, test accuracy: {w2_acc:.4f}")

# ============================================================================
# STEP 5: TRAIN WEEKEND 3 MODEL (Logistic Regression)
# ============================================================================
print("\nSTEP 5: Training Weekend 3 model (Logistic Regression)...")

w3_features = [
    '5man_bpm', 'assist_to_usage_ratio', 'experience_weighted_production',
    'effective_possession_rate', 'off_3pt_share', 'kenpom_off', 'blk_pct'
]

df_w3 = matchups[matchups['round'].isin(['Final Four', 'Championship'])].copy()
X_w3 = df_w3[w3_features].fillna(df_w3[w3_features].median())
y_w3 = df_w3['win']

X_train_w3, X_test_w3, y_train_w3, y_test_w3 = train_test_split(
    X_w3, y_w3, test_size=20, random_state=42, stratify=y_w3
)

scaler_w3 = StandardScaler()
X_train_w3_s = scaler_w3.fit_transform(X_train_w3)
X_test_w3_s = scaler_w3.transform(X_test_w3)

lr_w3 = LogisticRegression(max_iter=1000, random_state=42)
lr_w3.fit(X_train_w3_s, y_train_w3)

# Platt Scaling
w3_calibrated = CalibratedClassifierCV(lr_w3, method='sigmoid', cv='prefit')
w3_calibrated.fit(X_test_w3_s, y_test_w3)

w3_acc = accuracy_score(y_test_w3, lr_w3.predict(X_test_w3_s))
print(f"  ✓ Weekend 3: {len(df_w3)} games, test accuracy: {w3_acc:.4f}")

# ============================================================================
# STEP 6: BUILD 2026 MATCHUPS & PREDICT
# ============================================================================
print("\nSTEP 6: Building 2026 matchups and predicting...")

# Load full team data for 2026
teams_full = pd.read_csv(os.path.join(data_dir, 'men_2026_teams_training.csv'), index_col=0)
teams_2026 = teams_full[teams_full['year'] == 2026].copy()

# Load bracket template
bracket = pd.read_csv(os.path.join(data_dir, 'bracket_template.csv'))
print(f"  Bracket template: {len(bracket)} matchups")

# Build all possible matchups for 2026
# Create every team vs every other team
team_ids = teams_2026['team_id'].tolist()
all_matchups = []

for i, t1_id in enumerate(team_ids):
    for j, t2_id in enumerate(team_ids):
        if i != j:
            t1 = teams_2026[teams_2026['team_id'] == t1_id].iloc[0]
            t2 = teams_2026[teams_2026['team_id'] == t2_id].iloc[0]

            row = {'team_id': t1_id, 'opponent_id': t2_id,
                   'team': t1['team'], 'opponent': t2['team'],
                   'team_seed': t1['seed'], 'opp_seed': t2['seed'],
                   'team_region': t1.get('region', ''),
                   'opp_region': t2.get('region', '')}

            # Build differentials matching training format
            # COMPOSITES (team vs team)
            row['5man_bpm'] = t1['5man_bpm'] - t2['5man_bpm']
            row['3man_bpm'] = t1['3man_bpm'] - t2['3man_bpm']
            row['wab'] = t1['wab'] - t2['wab']
            row['kenpom_rtg'] = t1['kenpom_rtg'] - t2['kenpom_rtg']
            row['torvik_rtg'] = t1['torvik_rtg'] - t2['torvik_rtg']

            # LINEUP PRODUCTION - team offense vs opponent defense
            row['5man_prpg'] = t1['5man_prpg!'] - t2['5man_dprpg']
            row['3man_prpg'] = t1['3man_prpg!'] - t2['3man_dprpg']
            row['5man_dprpg'] = t1['5man_dprpg'] - t2['5man_prpg!']
            row['3man_dprpg'] = t1['3man_dprpg'] - t2['3man_prpg!']

            # TEAM vs TEAM
            row['size'] = t1['size'] - t2['size']
            row['height'] = t1['height'] - t2['height']
            row['experience'] = t1['experience'] - t2['experience']
            row['bench'] = t1['bench'] - t2['bench']
            row['raw_tempo'] = t1['raw_tempo'] - t2['raw_tempo']
            row['adj_tempo'] = t1['adj_tempo'] - t2['adj_tempo']
            row['3pr'] = t1['3pr'] - t2['3prd']
            row['3prd'] = t1['3prd'] - t2['3pr']

            # SHOT SHARES - offense vs defense
            row['off_dunk_share'] = t1['off_dunk_share'] - t2['def_dunk_share']
            row['off_close2_share'] = t1['off_close2_share'] - t2['def_close2_share']
            row['off_far2_share'] = t1['off_far2_share'] - t2['def_far2_share']
            row['off_3pt_share'] = t1['off_3pt_share'] - t2['def_3pt_share']
            row['def_dunk_share'] = t1['def_dunk_share'] - t2['off_dunk_share']
            row['def_close2_share'] = t1['def_close2_share'] - t2['off_close2_share']
            row['def_far2_share'] = t1['def_far2_share'] - t2['off_far2_share']
            row['def_3pt_share'] = t1['def_3pt_share'] - t2['off_3pt_share']

            # OFFENSE vs DEFENSE
            row['5man_obpm'] = t1['5man_obpm'] - t2['5man_dbpm']
            row['3man_obpm'] = t1['3man_obpm'] - t2['3man_dbpm']
            row['kenpom_off'] = t1['kenpom_off'] - t2['kenpom_def']
            row['torvik_off'] = t1['torvik_off'] - t2['torvik_def']
            row['efg_pct'] = t1['efg%'] - t2['efgd%']
            row['2p_pct'] = t1['2p%'] - t2['2p%d']
            row['3p_pct'] = t1['3p%'] - t2['3p%d']
            row['ft_pct'] = t1['ft%'] - t2['ft%d']
            row['ftr'] = t1['ftr'] - t2['ftrd']
            row['tor'] = t1['tor'] - t2['tord']
            row['orb_pct'] = t1['orb%'] - t2['drb%']
            row['ast_pct'] = t1['ast%'] - t2['ast%d']
            row['blk_pct'] = t1['blk%'] - t2['blked%']
            row['off_dunk_fg_pct'] = t1['off_dunk_fg%'] - t2['def_dunk_fg%']
            row['off_close2_fg_pct'] = t1['off_close2_fg%'] - t2['def_close2_fg%']
            row['off_far2_fg_pct'] = t1['off_far2_fg%'] - t2['def_far2_fg%']
            row['off_3pt_fg_pct'] = t1['off_3pt_fg%'] - t2['def_3pt_fg%']

            # DEFENSE vs OFFENSE
            row['5man_dbpm'] = t1['5man_dbpm'] - t2['5man_obpm']
            row['3man_dbpm'] = t1['3man_dbpm'] - t2['3man_obpm']
            row['kenpom_def'] = t1['kenpom_def'] - t2['kenpom_off']
            row['torvik_def'] = t1['torvik_def'] - t2['torvik_off']
            row['efgd_pct'] = t1['efgd%'] - t2['efg%']
            row['2pd_pct'] = t1['2p%d'] - t2['2p%']
            row['3pd_pct'] = t1['3p%d'] - t2['3p%']
            row['ftd_pct'] = t1['ft%d'] - t2['ft%']
            row['ftrd'] = t1['ftrd'] - t2['ftr']
            row['tord'] = t1['tord'] - t2['tor']
            row['drb_pct'] = t1['drb%'] - t2['orb%']
            row['astd_pct'] = t1['ast%d'] - t2['ast%']
            row['blked_pct'] = t1['blked%'] - t2['blk%']
            row['def_dunk_fg_pct'] = t1['def_dunk_fg%'] - t2['off_dunk_fg%']
            row['def_close2_fg_pct'] = t1['def_close2_fg%'] - t2['off_close2_fg%']
            row['def_far2_fg_pct'] = t1['def_far2_fg%'] - t2['off_far2_fg%']
            row['def_3pt_fg_pct'] = t1['def_3pt_fg%'] - t2['off_3pt_fg%']

            # ENGINEERED - offense vs defense
            row['net_efg_margin'] = t1['net_efg_margin'] - t2['def_net_efg_margin']
            row['net_turnover_margin'] = t1['net_turnover_margin'] - t2['def_net_turnover_margin']
            row['net_rebounding_margin'] = t1['net_rebounding_margin'] - t2['def_net_rebounding_margin']
            row['net_ftr_margin'] = t1['net_ftr_margin'] - t2['def_net_ftr_margin']
            row['rim_efficiency'] = t1['rim_efficiency'] - t2['def_rim_efficiency']
            row['perimeter_efficiency'] = t1['perimeter_efficiency'] - t2['def_perimeter_efficiency']
            row['three_point_volume_efficiency'] = t1['three_point_volume_efficiency'] - t2['def_three_point_volume_efficiency']
            row['shot_quality_variance'] = t1['shot_quality_variance'] - t2['def_shot_quality_variance']
            row['rim_to_three_ratio'] = t1['rim_to_three_ratio'] - t2['def_rim_to_three_ratio']
            row['mid_range_reliance'] = t1['mid_range_reliance'] - t2['def_mid_range_reliance']
            row['paint_touch_rate'] = t1['paint_touch_rate'] - t2['def_paint_touch_rate']
            row['top5_scoring_concentration'] = t1['top5_scoring_concentration'] - t2['top5_scoring_concentration']
            row['top5_rebounding_concentration'] = t1['top5_rebounding_concentration'] - t2['top5_rebounding_concentration']
            row['top5_def_rebounding_concentration'] = t1['top5_def_rebounding_concentration'] - t2['top5_def_rebounding_concentration']
            row['bench_scoring_ratio'] = t1['bench_scoring_ratio'] - t2['bench_scoring_ratio']
            row['rotation_balance'] = t1['rotation_balance'] - t2['rotation_balance']
            row['tempo_advantage'] = t1['tempo_advantage'] - t2['tempo_advantage']
            row['effective_possession_rate'] = t1['effective_possession_rate'] - t2['def_effective_possession_rate']
            row['offensive_versatility_score'] = t1['offensive_versatility_score'] - t2['defensive_versatility_score']
            row['size_speed_index'] = t1['size_speed_index'] - t2['size_speed_index']
            row['assist_to_usage_ratio'] = t1['assist_to_usage_ratio'] - t2['def_assist_suppression']
            row['free_throw_advantage'] = t1['free_throw_advantage'] - t2['def_free_throw_advantage']
            row['block_efficiency'] = t1['block_efficiency'] - t2['def_block_efficiency']
            row['experience_weighted_production'] = t1['experience_weighted_production'] - t2['def_experience_impact']
            row['four_factors_composite'] = t1['four_factors_composite'] - t2['def_four_factors_composite']
            row['elite_outcome_probability'] = t1['elite_outcome_probability'] - t2['elite_outcome_probability']
            row['offense_defense_balance'] = t1['offense_defense_balance'] - t2['offense_defense_balance']
            row['shooting_variance_resilience'] = t1['shooting_variance_resilience'] - t2['shooting_variance_resilience']
            row['lineup_depth_quality'] = t1['lineup_depth_quality'] - t2['def_lineup_depth_quality']

            # ENGINEERED - defense vs offense
            row['def_net_efg_margin'] = t1['def_net_efg_margin'] - t2['net_efg_margin']
            row['def_net_turnover_margin'] = t1['def_net_turnover_margin'] - t2['net_turnover_margin']
            row['def_net_rebounding_margin'] = t1['def_net_rebounding_margin'] - t2['net_rebounding_margin']
            row['def_net_ftr_margin'] = t1['def_net_ftr_margin'] - t2['net_ftr_margin']
            row['def_rim_efficiency'] = t1['def_rim_efficiency'] - t2['rim_efficiency']
            row['def_perimeter_efficiency'] = t1['def_perimeter_efficiency'] - t2['perimeter_efficiency']
            row['def_three_point_volume_efficiency'] = t1['def_three_point_volume_efficiency'] - t2['three_point_volume_efficiency']
            row['def_shot_quality_variance'] = t1['def_shot_quality_variance'] - t2['shot_quality_variance']
            row['def_rim_to_three_ratio'] = t1['def_rim_to_three_ratio'] - t2['rim_to_three_ratio']
            row['def_mid_range_reliance'] = t1['def_mid_range_reliance'] - t2['mid_range_reliance']
            row['def_paint_touch_rate'] = t1['def_paint_touch_rate'] - t2['paint_touch_rate']
            row['def_effective_possession_rate'] = t1['def_effective_possession_rate'] - t2['effective_possession_rate']
            row['defensive_versatility_score'] = t1['defensive_versatility_score'] - t2['offensive_versatility_score']
            row['def_size_speed_index'] = t1['def_size_speed_index'] - t2['size_speed_index']
            row['def_assist_suppression'] = t1['def_assist_suppression'] - t2['assist_to_usage_ratio']
            row['def_free_throw_advantage'] = t1['def_free_throw_advantage'] - t2['free_throw_advantage']
            row['def_block_efficiency'] = t1['def_block_efficiency'] - t2['block_efficiency']
            row['def_experience_impact'] = t1['def_experience_impact'] - t2['experience_weighted_production']
            row['def_four_factors_composite'] = t1['def_four_factors_composite'] - t2['four_factors_composite']
            row['def_lineup_depth_quality'] = t1['def_lineup_depth_quality'] - t2['lineup_depth_quality']

            all_matchups.append(row)

matchups_2026 = pd.DataFrame(all_matchups)
print(f"  Built {len(matchups_2026)} matchups for 2026")

# ============================================================================
# STEP 7: PREDICT WIN PROBABILITIES
# ============================================================================
print("\nSTEP 7: Predicting win probabilities...")

# Round 1 predictions
X_r1_2026 = matchups_2026[r1_features].fillna(0)
X_r1_2026_s = scaler_r1.transform(X_r1_2026)
matchups_2026['r1_win_prob'] = r1_calibrated.predict_proba(X_r1_2026_s)[:, 1]

# Round 2 predictions
X_r2_2026 = matchups_2026[r2_features].fillna(0)
X_r2_2026_s = scaler_r2.transform(X_r2_2026)
matchups_2026['r2_win_prob'] = r2_calibrated.predict_proba(X_r2_2026_s)[:, 1]

# Weekend 2 predictions
X_w2_2026 = matchups_2026[w2_features].fillna(0)
X_w2_2026_s = scaler_w2.transform(X_w2_2026)
matchups_2026['w2_win_prob'] = w2_calibrated.predict_proba(X_w2_2026_s)[:, 1]

# Weekend 3 predictions
X_w3_2026 = matchups_2026[w3_features].fillna(0)
X_w3_2026_s = scaler_w3.transform(X_w3_2026)
matchups_2026['w3_win_prob'] = w3_calibrated.predict_proba(X_w3_2026_s)[:, 1]

# Pairwise normalize: P(A beats B) + P(B beats A) = 1
for idx, row in matchups_2026.iterrows():
    opp_idx = matchups_2026[
        (matchups_2026['team_id'] == row['opponent_id']) &
        (matchups_2026['opponent_id'] == row['team_id'])
    ].index
    if len(opp_idx) > 0:
        opp_idx = opp_idx[0]
        for prob_col in ['r1_win_prob', 'r2_win_prob', 'w2_win_prob', 'w3_win_prob']:
            p1 = matchups_2026.loc[idx, prob_col]
            p2 = matchups_2026.loc[opp_idx, prob_col]
            total = p1 + p2
            if total > 0:
                matchups_2026.loc[idx, prob_col] = p1 / total

print("  ✓ Win probabilities calculated and normalized")

# ============================================================================
# STEP 8: CALCULATE ADVANCEMENT PROBABILITIES
# ============================================================================
print("\nSTEP 8: Calculating advancement probabilities...")

# Map round to model
round_model_map = {
    'Round 1': 'r1_win_prob',
    'Round 2': 'r2_win_prob',
    'Sweet 16': 'w2_win_prob',
    'Elite Eight': 'w2_win_prob',
    'Final Four': 'w3_win_prob',
    'Championship': 'w3_win_prob'
}

rounds_order = ['Round 1', 'Round 2', 'Sweet 16', 'Elite Eight', 'Final Four', 'Championship']
prob_cols = ['pct_round_2', 'pct_sweet_16', 'pct_elite_eight', 'pct_final_four', 'pct_championship', 'pct_champion']

# Initialize advancement probs
team_probs = {}
for tid in team_ids:
    team_probs[tid] = {pc: 0.0 for pc in prob_cols}

# Round 1: P(advance) = P(win matchup) for actual bracket matchup
# For simplicity with all-vs-all, use region + seed to determine bracket path
# Round 1 matchups: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
r1_seed_matchups = [(1,16), (2,15), (3,14), (4,13), (5,12), (6,11), (7,10), (8,9)]

regions = teams_2026['region'].unique()

for region in regions:
    region_teams = teams_2026[teams_2026['region'] == region]

    for high_seed, low_seed in r1_seed_matchups:
        high_team = region_teams[region_teams['seed'] == high_seed]
        low_team = region_teams[region_teams['seed'] == low_seed]

        if len(high_team) == 0 or len(low_team) == 0:
            continue

        h_id = high_team.iloc[0]['team_id']
        l_id = low_team.iloc[0]['team_id']

        matchup_row = matchups_2026[
            (matchups_2026['team_id'] == h_id) & (matchups_2026['opponent_id'] == l_id)
        ]
        if len(matchup_row) > 0:
            p_high = matchup_row.iloc[0]['r1_win_prob']
            team_probs[h_id]['pct_round_2'] = p_high
            team_probs[l_id]['pct_round_2'] = 1 - p_high

# Subsequent rounds: P(advance) = sum over possible opponents of P(team in round) * P(opp in round) * P(win)
def calc_round_prob(teams_in_pod, current_round_col, next_round_col, prob_key):
    """Calculate advancement probability for a round within a pod of teams."""
    for team_id in teams_in_pod:
        total_prob = 0.0
        p_team = team_probs[team_id][current_round_col]
        if p_team == 0:
            continue
        for opp_id in teams_in_pod:
            if opp_id == team_id:
                continue
            p_opp = team_probs[opp_id][current_round_col]
            if p_opp == 0:
                continue
            matchup_row = matchups_2026[
                (matchups_2026['team_id'] == team_id) & (matchups_2026['opponent_id'] == opp_id)
            ]
            if len(matchup_row) > 0:
                win_prob = matchup_row.iloc[0][prob_key]
                total_prob += p_opp * win_prob
        team_probs[team_id][next_round_col] = p_team * total_prob

# Round 2 pods (seeds that meet in R2): {1,16,8,9}, {2,15,7,10}, {3,14,6,11}, {4,13,5,12}
r2_pods = [(1,16,8,9), (2,15,7,10), (3,14,6,11), (4,13,5,12)]

for region in regions:
    region_teams = teams_2026[teams_2026['region'] == region]

    for pod_seeds in r2_pods:
        pod_ids = []
        for s in pod_seeds:
            t = region_teams[region_teams['seed'] == s]
            if len(t) > 0:
                pod_ids.append(t.iloc[0]['team_id'])
        if pod_ids:
            calc_round_prob(pod_ids, 'pct_round_2', 'pct_sweet_16', 'r2_win_prob')

# Sweet 16 pods (top half: seeds 1,16,8,9,4,13,5,12 and bottom half: 2,15,7,10,3,14,6,11)
s16_top = (1,16,8,9,4,13,5,12)
s16_bot = (2,15,7,10,3,14,6,11)

for region in regions:
    region_teams = teams_2026[teams_2026['region'] == region]

    for pod_seeds in [s16_top, s16_bot]:
        pod_ids = []
        for s in pod_seeds:
            t = region_teams[region_teams['seed'] == s]
            if len(t) > 0:
                pod_ids.append(t.iloc[0]['team_id'])
        if pod_ids:
            calc_round_prob(pod_ids, 'pct_sweet_16', 'pct_elite_eight', 'w2_win_prob')

# Elite Eight: full region
for region in regions:
    region_teams = teams_2026[teams_2026['region'] == region]
    region_ids = region_teams['team_id'].tolist()
    calc_round_prob(region_ids, 'pct_elite_eight', 'pct_final_four', 'w2_win_prob')

# Final Four: all teams
all_ids = teams_2026['team_id'].tolist()
calc_round_prob(all_ids, 'pct_final_four', 'pct_championship', 'w3_win_prob')

# Championship
calc_round_prob(all_ids, 'pct_championship', 'pct_champion', 'w3_win_prob')

# Normalize champion probabilities to sum to 1
total_champ = sum(team_probs[tid]['pct_champion'] for tid in team_ids)
if total_champ > 0:
    for tid in team_ids:
        team_probs[tid]['pct_champion'] /= total_champ

print(f"  ✓ Total champion probability: {sum(team_probs[tid]['pct_champion'] for tid in team_ids):.4f}")

# ============================================================================
# STEP 9: CALCULATE BRACKET VALUE
# ============================================================================
print("\nSTEP 9: Calculating bracket value...")

for tid in team_ids:
    p = team_probs[tid]
    p['bracket_value'] = (
        p['pct_round_2'] * 1 +
        p['pct_sweet_16'] * 2 +
        p['pct_elite_eight'] * 4 +
        p['pct_final_four'] * 8 +
        p['pct_championship'] * 16 +
        p['pct_champion'] * 32
    )

# ============================================================================
# STEP 10: UPDATE TEAMS OUTPUT
# ============================================================================
print("\nSTEP 10: Updating teams output...")

for tid in team_ids:
    team_name = teams_2026[teams_2026['team_id'] == tid].iloc[0]['team']
    mask = (teams_output['team'] == team_name) & (teams_output['year'] == 2026)
    for col in prob_cols + ['bracket_value']:
        teams_output.loc[mask, col] = round(team_probs[tid][col], 4)

# Save updated teams output
teams_path = os.path.join(data_dir, 'men_teams_output.csv')
teams_output.to_csv(teams_path, index=False)
print(f"  ✓ Saved: {teams_path}")

# Show top 10 by bracket value
top10 = teams_output[teams_output['year'] == 2026].nlargest(10, 'bracket_value')
print(f"\n  2026 Top 10 by Bracket Value:")
print(top10[['team', 'seed', 'tier', 'bracket_value', 'pct_champion', 'pct_championship']].to_string(index=False))

total_champ_check = teams_output[teams_output['year'] == 2026]['pct_champion'].sum()
print(f"\n  Total champion probability (2026): {total_champ_check:.4f}")

# ============================================================================
# STEP 11: SAVE MATCHUPS OUTPUT
# ============================================================================
print("\nSTEP 11: Saving matchups output...")

matchups_out = matchups_2026[['team_id', 'opponent_id', 'team', 'opponent',
                               'team_seed', 'opp_seed', 'team_region', 'opp_region',
                               'r1_win_prob', 'r2_win_prob', 'w2_win_prob', 'w3_win_prob']].copy()

matchups_path = os.path.join(data_dir, 'men_matchups_output.csv')
matchups_out.to_csv(matchups_path, index=False)
print(f"  ✓ Saved: {matchups_path}")
print(f"    {len(matchups_out)} matchups")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nMen's pipeline:")
print(f"  1. python3 backend/src/men_create_data.py")
print(f"  2. python3 backend/src/men_generate_outputs.py")
print(f"  3. python3 backend/src/men_generate_matchups.py")