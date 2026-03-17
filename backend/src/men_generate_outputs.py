"""
Men's NCAA Tournament - Composite Scores & Tiers
  Loads prepared training data, builds composite scores, clusters, assigns tiers.

Inputs (in backend/data/men/):
  - men_2026_teams_training.csv (from men_create_data.py)

Outputs (in backend/data/men/):
  - men_teams_output.csv
  - men_model_weights.csv

Usage:
  python3 backend/src/men_generate_outputs.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# ============================================================================
# PATHS
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'men')

print("=" * 80)
print("MEN'S NCAA TOURNAMENT - COMPOSITES & TIERS")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: Loading data...")

df = pd.read_csv(os.path.join(data_dir, 'men_2026_teams_training.csv'), index_col=0)
print(f"  {len(df)} teams, {df['year'].nunique()} years ({df['year'].min()}-{df['year'].max()})")

df['performance'] = df['weekend']

# ============================================================================
# STEP 2: BUILD COMPOSITE SCORES
# ============================================================================
print("\nSTEP 2: Building composite scores...")

offensive_vars = [
    '3man_dbpm', '5man_obpm', 'assist_to_usage_ratio', 'ast%',
    'effective_possession_rate', 'efg%', 'four_factors_composite',
    'free_throw_advantage', 'lineup_depth_quality', 'mid_range_reliance',
    'net_rebounding_margin', 'net_turnover_margin', 'off_3pt_fg%',
    'off_3pt_share', 'off_close2_fg%', 'off_close2_share', 'off_dunk_fg%',
    'off_dunk_share', 'off_far2_fg%', 'off_far2_share',
    'offensive_versatility_score', 'orb%', 'paint_touch_rate',
    'perimeter_efficiency', 'rim_efficiency', 'rim_to_three_ratio',
    'shot_quality_variance', 'size_speed_index', 'three_point_volume_efficiency',
    'tor', 'kenpom_off', 'torvik_off'
]

defensive_vars = [
    '3man_obpm', '5man_dbpm', 'ast%d', 'def_3pt_fg%', 'def_3pt_share',
    'def_assist_suppression', 'def_close2_fg%', 'def_close2_share',
    'def_dunk_fg%', 'def_dunk_share', 'def_effective_possession_rate',
    'def_experience_impact', 'def_far2_fg%', 'def_far2_share',
    'def_four_factors_composite', 'def_free_throw_advantage',
    'def_lineup_depth_quality', 'def_mid_range_reliance',
    'def_net_rebounding_margin', 'def_net_turnover_margin',
    'def_paint_touch_rate', 'def_perimeter_efficiency', 'def_rim_efficiency',
    'def_rim_to_three_ratio', 'def_shot_quality_variance',
    'def_size_speed_index', 'def_three_point_volume_efficiency',
    'defensive_versatility_score', 'drb%', 'efgd%', 'kenpom_def',
    'tord', 'torvik_def'
]

overall_vars = [
    '3man_bpm', '5man_bpm', 'bench_scoring_ratio', 'elite_outcome_probability',
    'kenpom_rtg', 'rotation_balance', 'torvik_rtg', 'wab'
]

offensive_vars = [v for v in offensive_vars if v in df.columns]
defensive_vars = [v for v in defensive_vars if v in df.columns]
overall_vars = [v for v in overall_vars if v in df.columns]

print(f"  Variables: {len(offensive_vars)} offensive, {len(defensive_vars)} defensive, {len(overall_vars)} overall")


def build_composite_score(df, variables, target_col):
    """Build a composite score using 50/50 correlation + RF importance weighting."""
    score_df = df[variables].copy().fillna(df[variables].median())
    X = score_df.values
    y = df[target_col].values

    correlations = {m: abs(df[m].corr(df[target_col])) for m in variables}

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    rf_importance = dict(zip(variables, rf.feature_importances_))

    combined = {}
    for m in variables:
        corr_norm = correlations[m] / sum(correlations.values())
        rf_norm = rf_importance[m] / sum(rf_importance.values())
        combined[m] = (corr_norm + rf_norm) / 2

    total = sum(combined.values())
    weights = {k: v / total for k, v in combined.items()}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(score_df)
    weighted = np.average(scaled, axis=1, weights=[weights[m] for m in variables])

    mm = MinMaxScaler(feature_range=(0, 10))
    scores = np.round(mm.fit_transform(weighted.reshape(-1, 1)).flatten(), 2)

    return scores, weights


# Offensive
df['offensive_score'], weights_off = build_composite_score(df, offensive_vars, 'performance')
print(f"  Offensive: {df['offensive_score'].min():.2f}-{df['offensive_score'].max():.2f}, "
      f"corr={df['offensive_score'].corr(df['performance']):.4f}")

# Defensive
df['defensive_score'], weights_def = build_composite_score(df, defensive_vars, 'performance')
print(f"  Defensive: {df['defensive_score'].min():.2f}-{df['defensive_score'].max():.2f}, "
      f"corr={df['defensive_score'].corr(df['performance']):.4f}")

# Overall (includes offensive_score and defensive_score as inputs)
overall_input = df[overall_vars].copy()
overall_input['offensive_score'] = df['offensive_score']
overall_input['defensive_score'] = df['defensive_score']
overall_input = overall_input.fillna(overall_input.median())

all_overall_vars = overall_vars + ['offensive_score', 'defensive_score']
X_ovr = overall_input.values
y_ovr = df['performance'].values

corr_ovr = {m: abs(overall_input[m].corr(df['performance'])) for m in all_overall_vars}
rf_ovr = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_ovr.fit(X_ovr, y_ovr)
rf_imp_ovr = dict(zip(all_overall_vars, rf_ovr.feature_importances_))

combined_ovr = {}
for m in all_overall_vars:
    c_norm = corr_ovr[m] / sum(corr_ovr.values())
    r_norm = rf_imp_ovr[m] / sum(rf_imp_ovr.values())
    combined_ovr[m] = (c_norm + r_norm) / 2

total_ovr = sum(combined_ovr.values())
weights_ovr = {k: v / total_ovr for k, v in combined_ovr.items()}

scaler_ovr = StandardScaler()
overall_scaled = scaler_ovr.fit_transform(overall_input)
overall_weighted = np.average(overall_scaled, axis=1,
                              weights=[weights_ovr[m] for m in all_overall_vars])

mm_ovr = MinMaxScaler(feature_range=(0, 10))
df['overall_score'] = np.round(mm_ovr.fit_transform(overall_weighted.reshape(-1, 1)).flatten(), 2)

print(f"  Overall:   {df['overall_score'].min():.2f}-{df['overall_score'].max():.2f}, "
      f"corr={df['overall_score'].corr(df['performance']):.4f}")

# ============================================================================
# STEP 3: K-MEANS CLUSTERING & TIER ASSIGNMENT
# ============================================================================
print("\nSTEP 3: Clustering & tier assignment...")

cluster_vars = [
    '5man_prpg!', '5man_dprpg', '5man_bpm', '5man_obpm', '5man_dbpm',
    '3man_prpg!', '3man_dprpg', '3man_bpm', '3man_obpm', '3man_dbpm',
    'kenpom_off', 'kenpom_def', 'kenpom_rtg', 'size', 'height',
    'dr_from_5', 'wab', 'torvik_off', 'torvik_def', 'torvik_rtg',
    'efg%', 'efgd%', 'tor', 'orb%', '2p%', '2p%d', '3p%d', 'blk%',
    'off_dunk_share', 'off_close2_fg%', 'def_close2_share',
    'def_far2_share', 'off_3pt_fg%', 'def_3pt_fg%',
    'net_rebounding_margin', 'def_net_rebounding_margin',
    'rim_efficiency', 'def_perimeter_efficiency', 'def_mid_range_reliance',
    'effective_possession_rate', 'size_speed_index', 'def_size_speed_index',
    'block_efficiency', 'experience_weighted_production',
    'def_block_efficiency', 'def_experience_impact',
    'four_factors_composite', 'def_four_factors_composite',
    'lineup_depth_quality', 'def_lineup_depth_quality'
]

cluster_vars = [v for v in cluster_vars if v in df.columns]

# Fit on historical only (704 teams), matching the notebook exactly
hist_mask = df['year'] < 2026
hist_df = df.loc[hist_mask, cluster_vars].copy()
hist_medians = hist_df.median()
hist_df = hist_df.fillna(hist_medians)

sc = StandardScaler()
scaled_hist = sc.fit_transform(hist_df)

kmeans = KMeans(n_clusters=8, init='k-means++', algorithm='lloyd',
                max_iter=500, random_state=123)
df.loc[hist_mask, 'cluster'] = kmeans.fit_predict(scaled_hist)

# Assign 2026 teams to nearest cluster
cur_df = df.loc[~hist_mask, cluster_vars].copy().fillna(hist_medians)
scaled_cur = sc.transform(cur_df)
df.loc[~hist_mask, 'cluster'] = kmeans.predict(scaled_cur)
df['cluster'] = df['cluster'].astype(int)

print(f"\n  Cluster x Finish:")
print(pd.crosstab(index=df['finish'], columns=df['cluster']).to_string())

# KenPom rank within each year
df['kenpom_rtg_rank'] = df.groupby('year')['kenpom_rtg'].rank(method='min', ascending=False)

# Tier mapping
df['tier'] = ''

# S Tier
df.loc[(df['cluster'] == 1) & (df['kenpom_rtg_rank'] <= 6), 'tier'] = 'S'

# A Tier
df.loc[(df['cluster'] == 7) & (df['kenpom_rtg_rank'] <= 14), 'tier'] = 'A'
df.loc[(df['cluster'] == 1) & (df['kenpom_rtg_rank'] > 6) &
       (df['kenpom_rtg_rank'] <= 12), 'tier'] = 'A'
df.loc[(df['cluster'] == 3) & (df['kenpom_rtg_rank'] <= 11), 'tier'] = 'A'

# B Tier
df.loc[(df['cluster'] == 7) & (df['kenpom_rtg_rank'] > 14) &
       (df['kenpom_rtg_rank'] <= 24), 'tier'] = 'B'
df.loc[(df['cluster'] == 5) & (df['kenpom_rtg_rank'] <= 13), 'tier'] = 'B'
df.loc[(df['cluster'] == 3) & (df['kenpom_rtg_rank'] > 11), 'tier'] = 'B'
df.loc[(df['cluster'] == 0) & (df['kenpom_rtg_rank'] <= 14), 'tier'] = 'B'
df.loc[(df['cluster'] == 2) & (df['kenpom_rtg_rank'] <= 24), 'tier'] = 'B'

# C Tier
df.loc[(df['cluster'] == 7) & (df['kenpom_rtg_rank'] > 24) &
       (df['kenpom_rtg_rank'] <= 38), 'tier'] = 'C'
df.loc[(df['cluster'] == 5) & (df['kenpom_rtg_rank'] > 13) &
       (df['kenpom_rtg_rank'] <= 44), 'tier'] = 'C'
df.loc[(df['cluster'] == 1) & (df['kenpom_rtg_rank'] > 12), 'tier'] = 'C'
df.loc[(df['cluster'] == 0) & (df['kenpom_rtg_rank'] > 14) &
       (df['kenpom_rtg_rank'] <= 39), 'tier'] = 'C'
df.loc[(df['cluster'] == 2) & (df['kenpom_rtg_rank'] > 24) &
       (df['kenpom_rtg_rank'] <= 45), 'tier'] = 'C'

# D Tier
df.loc[(df['cluster'] == 7) & (df['kenpom_rtg_rank'] > 38), 'tier'] = 'D'
df.loc[(df['cluster'] == 5) & (df['kenpom_rtg_rank'] > 44), 'tier'] = 'D'
df.loc[(df['cluster'] == 0) & (df['kenpom_rtg_rank'] > 39), 'tier'] = 'D'
df.loc[(df['cluster'] == 2) & (df['kenpom_rtg_rank'] > 45), 'tier'] = 'D'

# F Tier
df.loc[df['cluster'] == 4, 'tier'] = 'F'
df.loc[df['cluster'] == 6, 'tier'] = 'F'

# Fill remaining
df.loc[df['tier'] == '', 'tier'] = 'C'

print(f"\n  Tier x Finish:")
print(pd.crosstab(index=df['finish'], columns=df['tier']).to_string())

print(f"\n  Tier distribution:")
for tier in ['S', 'A', 'B', 'C', 'D', 'F']:
    count = len(df[df['tier'] == tier])
    if count > 0:
        print(f"    {tier}: {count} teams")

# ============================================================================
# STEP 4: VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print(f"\n  Correlation with tournament performance:")
print(f"    Overall:   {df['overall_score'].corr(df['performance']):.4f}")
print(f"    Offensive: {df['offensive_score'].corr(df['performance']):.4f}")
print(f"    Defensive: {df['defensive_score'].corr(df['performance']):.4f}")

champions = df[df['finish'] == 'Champion']
if len(champions) > 0:
    print(f"\n  Champions average scores ({len(champions)} champions):")
    print(f"    Overall:   {champions['overall_score'].mean():.2f}")
    print(f"    Offensive: {champions['offensive_score'].mean():.2f}")
    print(f"    Defensive: {champions['defensive_score'].mean():.2f}")

print(f"\n  Top 10 teams (overall score):")
top10 = df.nlargest(10, 'overall_score')[
    ['team', 'year', 'seed', 'tier', 'overall_score',
     'offensive_score', 'defensive_score', 'finish']
]
print(top10.to_string(index=False))

# ============================================================================
# STEP 5: SAVE OUTPUTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Saving outputs...")

output_df = pd.DataFrame()
output_df['year'] = df['year']
output_df['team'] = df['team']
output_df['region'] = df['region'] if 'region' in df.columns else ''
output_df['seed'] = df['seed']
output_df['tier'] = df['tier']
output_df['overall'] = df['overall_score']
output_df['offense'] = df['offensive_score']
output_df['defense'] = df['defensive_score']

# Placeholder columns for Stage 2
output_df['pct_round_2'] = 0.0
output_df['pct_sweet_16'] = 0.0
output_df['pct_elite_eight'] = 0.0
output_df['pct_final_four'] = 0.0
output_df['pct_championship'] = 0.0
output_df['pct_champion'] = 0.0
output_df['bracket_value'] = 0.0

# Raw stats for dashboard
output_df['5man_bpm'] = df['5man_bpm'].values
output_df['3man_bpm'] = df['3man_bpm'].values
output_df['elite_outcome_probability'] = df['elite_outcome_probability'].values
output_df['efg%'] = df['efg%'].values
output_df['def_rim_efficiency'] = df['def_rim_efficiency'].values

teams_path = os.path.join(data_dir, 'men_teams_output.csv')
output_df.to_csv(teams_path, index=False)
print(f"  ✓ Saved: {teams_path}")
print(f"    {len(output_df)} rows, years {output_df['year'].min()}-{output_df['year'].max()}")

# Variable weights
weights_all = pd.concat([
    pd.DataFrame({'Variable': list(weights_off.keys()),
                   'Weight': list(weights_off.values()), 'Score': 'Offensive'}),
    pd.DataFrame({'Variable': list(weights_def.keys()),
                   'Weight': list(weights_def.values()), 'Score': 'Defensive'}),
    pd.DataFrame({'Variable': list(weights_ovr.keys()),
                   'Weight': list(weights_ovr.values()), 'Score': 'Overall'})
])
weights_path = os.path.join(data_dir, 'men_model_weights.csv')
weights_all.to_csv(weights_path, index=False)
print(f"  ✓ Saved: {weights_path}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nMen's pipeline:")
print(f"  1. python3 backend/src/men_create_data.py")
print(f"  2. python3 backend/src/men_generate_outputs.py")
print(f"\nNext: Share the 4 model feature sets for Stage 2 (matchup models)")