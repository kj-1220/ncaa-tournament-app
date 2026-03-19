"""
Women's NCAA Tournament - Full Output Generator
Trains matchup models once on all available data, then scores each year 2021-2026.
Outputs:
  1. women_teams_output.csv    - All teams 2021-2026 with composite scores + round probabilities + bracket value
  2. women_matchups_output.csv - All matchups 2021-2026 with win probabilities
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

# ============================================================================
# PATHS
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')

print("="*80)
print("WOMEN'S NCAA TOURNAMENT - FULL OUTPUT GENERATOR")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD ALL DATA
# ============================================================================
print("STEP 1: Loading data...")

torvik_hist  = pd.read_csv(os.path.join(data_dir, 'women_torvik_historical.csv'))
teams_hist   = pd.read_csv(os.path.join(data_dir, 'women_teams_historical.csv'))
matchups_training = pd.read_csv(os.path.join(data_dir, 'women_matchups_training.csv'))
current_composites = pd.read_csv(os.path.join(data_dir, 'women_composites_current.csv'))
current_matchups   = pd.read_csv(os.path.join(data_dir, 'women_matchups_with_probs.csv'))

print(f"  Torvik historical: {len(torvik_hist)} rows, years {torvik_hist['year'].min()}-{torvik_hist['year'].max()}")
print(f"  Teams historical:  {len(teams_hist)} rows, years {teams_hist['year'].min()}-{teams_hist['year'].max()}")
print(f"  Matchup training:  {len(matchups_training)} rows")
print(f"  Current composites: {len(current_composites)} teams (2026)")
print(f"  Current matchups:   {len(current_matchups)} rows (2026)")

# ============================================================================
# STEP 2: TRAIN MATCHUP MODELS (once, on all training data)
# ============================================================================
print("\nSTEP 2: Training matchup models on all available data...")

early_features = ['barthag', 'adj_oe', 'adj_de', 'orb_pct', 'drb_pct', 'ftr', '2p_pct']
elite_features = ['wab', 'barthag', 'adj_oe', 'adj_de', 'efg_pct', 'efgd_pct',
                  'orb_pct', 'drb_pct', '2p_pct', '2pd_pct', '3p_pct', '3pd_pct', '3pr']

early_rounds_labels = ['First Round', 'Second Round']
elite_rounds_labels = ['Sweet 16', 'Elite Eight', 'Final Four', 'Championship']

early_df = matchups_training[matchups_training['round'].isin(early_rounds_labels)].copy()
elite_df = matchups_training[matchups_training['round'].isin(elite_rounds_labels)].copy()

# Early rounds model
X_early = early_df[early_features]
y_early = early_df['win']
X_tr, X_te, y_tr, y_te = train_test_split(X_early, y_early, test_size=0.3, random_state=42)
base_early = LogisticRegression(random_state=42, max_iter=1000)
base_early.fit(X_tr, y_tr)
early_model = CalibratedClassifierCV(FrozenEstimator(base_early), method='sigmoid', cv=2)
early_model.fit(X_te, y_te)
print(f"  ✓ Early rounds model trained ({len(early_df)} games)")

# Elite rounds model
X_elite = elite_df[elite_features]
y_elite = elite_df['win']
X_tr, X_te, y_tr, y_te = train_test_split(X_elite, y_elite, test_size=0.3, random_state=42)
base_elite = XGBClassifier(
    learning_rate=0.2997738363859162, max_depth=9, min_child_weight=8.623522034407337,
    subsample=0.8324211691115178, colsample_bytree=0.9988769480719698,
    gamma=2.017715776385069, reg_alpha=0.9692563913308194,
    reg_lambda=2.5910989850621258, n_estimators=335,
    random_state=42, eval_metric='logloss'
)
base_elite.fit(X_tr, y_tr)
elite_model = CalibratedClassifierCV(FrozenEstimator(base_elite), method='sigmoid', cv=2)
elite_model.fit(X_te, y_te)
print(f"  ✓ Elite rounds model trained ({len(elite_df)} games)")

# ============================================================================
# STEP 3: TRAIN COMPOSITE + TIER MODELS (NCAAPredictor logic inline)
# ============================================================================
print("\nSTEP 3: Training composite and tier models on historical data...")

finish_mapping = {
    'First Round': 1, 'Second Round': 2, 'Sweet 16': 3,
    'Elite Eight': 4, 'Final Four': 5, 'Runner Up': 6, 'Champion': 7
}

offensive_vars = ['adj_oe', 'efg_pct', 'tor', 'orb_pct', 'ftr', '2p_pct', '3p_pct', '3pr']
defensive_vars = ['adj_de_inv', 'efgd_pct_inv', 'tord', 'drb_pct', 'ftrd', '2pd_pct_inv', '3pd_pct_inv', '3prd']
overall_vars   = ['barthag', 'wab', 'adj_tempo']
cluster_features = ['adj_oe', 'adj_de', 'barthag', 'efg_pct', 'efgd_pct',
                    'orb_pct', 'drb_pct', 'tor', 'tord', 'wab']

# Merge historical teams + torvik for training
hist_merged = teams_hist.merge(torvik_hist, on='torvik_id', how='left', suffixes=('', '_torvik'))
hist_merged['performance'] = hist_merged['finish'].map(finish_mapping)
hist_merged['adj_de_inv']   = -hist_merged['adj_de']
hist_merged['efgd_pct_inv'] = -hist_merged['efgd_pct']
hist_merged['2pd_pct_inv']  = -hist_merged['2pd_pct']
hist_merged['3pd_pct_inv']  = -hist_merged['3pd_pct']

train_df = hist_merged.dropna(subset=['performance'])
y_train  = train_df['performance'].values

def calc_weights(df, features, target):
    fdf = df[features].copy().fillna(df[features].median())
    correlations = {f: abs(df[f].corr(pd.Series(target))) for f in features}
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(fdf.values, target)
    rf_imp = dict(zip(features, rf.feature_importances_))
    combined = {}
    for f in features:
        cn = correlations[f] / sum(correlations.values())
        rn = rf_imp[f] / sum(rf_imp.values())
        combined[f] = (cn + rn) / 2
    total = sum(combined.values())
    return {k: v/total for k, v in combined.items()}

def calc_percentiles(df, features):
    return {f: {'p01': df[f].quantile(0.01), 'p99': df[f].quantile(0.99)} for f in features}

def calc_score(df, features, weights, percentiles):
    fdf = df[features].fillna(df[features].median())
    normalized = pd.DataFrame(index=df.index)
    for f in features:
        p01, p99 = percentiles[f]['p01'], percentiles[f]['p99']
        if p99 > p01:
            normalized[f] = ((fdf[f] - p01) / (p99 - p01) * 100).clip(0, 100)
        else:
            normalized[f] = 50.0
    return sum(normalized[f] * weights[f] for f in features)

# Train weights
w_off  = calc_weights(train_df, offensive_vars, y_train)
p_off  = calc_percentiles(train_df, offensive_vars)

w_def  = calc_weights(train_df, defensive_vars, y_train)
p_def  = calc_percentiles(train_df, defensive_vars)

train_df['offensive_score'] = calc_score(train_df, offensive_vars, w_off, p_off)
train_df['defensive_score'] = calc_score(train_df, defensive_vars, w_def, p_def)
all_overall = overall_vars + ['offensive_score', 'defensive_score']
w_ovr  = calc_weights(train_df[all_overall], all_overall, y_train)
p_ovr  = calc_percentiles(train_df[all_overall], all_overall)

# Train tier model
scaler = StandardScaler()
scaled = scaler.fit_transform(train_df[cluster_features].fillna(train_df[cluster_features].median()))
kmeans = KMeans(n_clusters=5, init='k-means++', algorithm='lloyd', max_iter=500, random_state=123)
kmeans.fit(scaled)
print("  ✓ Composite, tier, and clustering models trained")

def assign_tiers(df):
    tier = pd.Series('', index=df.index)

    ### S Tier ###
    tier[(df['cluster']==0) & (df['barthag_rtg_rank']<=4)] = 'S'

    ### A Tier ###
    tier[(df['cluster']==0) & (df['barthag_rtg_rank']>4) & (df['seed']<=3)] = 'A'
    tier[(df['cluster']==2) & (df['barthag_rtg_rank']<=8) & (df['seed']<=3)] = 'A'
    tier[(df['cluster']==3) & (df['barthag_rtg_rank']<=8) & (df['seed']<=3)] = 'A'

    ### B Tier ###
    tier[(df['cluster']==4) & (df['barthag_rtg_rank']<=24) & (df['seed']<=6)] = 'B'
    tier[(df['cluster']==0) & (df['seed']>3) & (df['seed']<=6)] = 'B'
    tier[(df['cluster']==2) & (df['barthag_rtg_rank']<=8) & (df['seed']>3)] = 'B'
    tier[(df['cluster']==3) & (df['barthag_rtg_rank']<=8) & (df['seed']>3)] = 'B'

    ### C Tier ###
    tier[(df['cluster']==4) & (df['barthag_rtg_rank']>24) & (df['seed']<=6)] = 'C'
    tier[(df['cluster']==4) & (df['barthag_rtg_rank']<=24) & (df['seed']>6)] = 'C'
    tier[(df['cluster']==4) & (df['barthag_rtg_rank']>24) & (df['seed']<=9)] = 'C'
    tier[(df['cluster']==2) & (df['barthag_rtg_rank']>8) & (df['seed']<=3)] = 'C'
    tier[(df['cluster']==2) & (df['barthag_rtg_rank']<=8) & (df['seed']>3)] = 'C'
    tier[(df['cluster']==2) & (df['seed']>3) & (df['seed']<=6)] = 'C'
    tier[(df['cluster']==3) & (df['barthag_rtg_rank']>8) & (df['seed']<=3)] = 'C'
    tier[(df['cluster']==3) & (df['barthag_rtg_rank']<=8) & (df['seed']>3)] = 'C'
    tier[(df['cluster']==3) & (df['seed']>3) & (df['seed']<=6)] = 'C'

    ### D Tier ###
    tier[(df['cluster']==4) & (df['seed']>9) & (df['seed']<=12)] = 'D'
    tier[(df['cluster']==0) & (df['seed']>6)] = 'D'
    tier[(df['cluster']==2) & (df['seed']>6) & (df['seed']<=12)] = 'D'
    tier[(df['cluster']==3) & (df['seed']>6) & (df['seed']<=12)] = 'D'

    ### F Tier ###
    tier[(df['cluster']==4) & (df['seed']>12)] = 'F'
    tier[(df['cluster']==1)] = 'F'
    tier[(df['cluster']==2) & (df['seed']>12)] = 'F'
    tier[(df['cluster']==3) & (df['seed']>12)] = 'F'

    tier[tier==''] = 'C'
    return tier

def score_teams(df):
    """Score composite + tiers for a set of teams."""
    d = df.copy()
    d['adj_de_inv']   = -d['adj_de']
    d['efgd_pct_inv'] = -d['efgd_pct']
    d['2pd_pct_inv']  = -d['2pd_pct']
    d['3pd_pct_inv']  = -d['3pd_pct']

    d['offensive_score'] = calc_score(d, offensive_vars, w_off, p_off)
    d['defensive_score'] = calc_score(d, defensive_vars, w_def, p_def)
    ovr_df = d[all_overall].copy()
    d['overall_score'] = calc_score(ovr_df, all_overall, w_ovr, p_ovr)

    n = len(d)
    d['offense'] = (10 - (d['offensive_score'].rank(method='first', ascending=False) - 1) / (n - 1) * 9).round(2)
    d['defense'] = (10 - (d['defensive_score'].rank(method='first', ascending=False) - 1) / (n - 1) * 9).round(2)
    d['overall'] = (10 - (d['overall_score'].rank(method='first', ascending=False) - 1) / (n - 1) * 9).round(2)

    # Tiers
    clust_scaled = scaler.transform(d[cluster_features].fillna(d[cluster_features].median()))
    d['cluster'] = kmeans.predict(clust_scaled)
    d['barthag_rtg_rank'] = d['barthag'].rank(method='min', ascending=False)
    d['tier'] = assign_tiers(d)

    return d

# ============================================================================
# STEP 4: BUILD MATCHUP HELPER (with proper differentials)
# ============================================================================

# Raw stat columns needed from team data
raw_stat_cols = ['wab', 'barthag', 'adj_oe', 'adj_de', 'efg_pct', 'efgd_pct',
                 'tor', 'tord', 'orb_pct', 'drb_pct', 'ftr', 'ftrd',
                 '2p_pct', '2pd_pct', '3p_pct', '3pd_pct', '3pr', '3prd', 'adj_tempo']


def invert_defensive_stats(df):
    """
    Invert defensive stats so higher = better, matching women_create_matchups.py.
    Operates on a copy so original data is not modified.
    """
    d = df.copy()
    d['adj_de']   = 200 - d['adj_de']
    d['efgd_pct'] = 100 - d['efgd_pct']
    d['tord']     = 100 - d['tord']
    d['drb_pct']  = 100 - d['drb_pct']
    d['ftrd']     = 100 - d['ftrd']
    d['2pd_pct']  = 100 - d['2pd_pct']
    d['3pd_pct']  = 100 - d['3pd_pct']
    d['3prd']     = 100 - d['3prd']
    return d


def compute_differentials(team_stats, opp_stats):
    """
    Compute matchup differentials with CORRECTED inversion handling.
    Both team_stats and opp_stats have inverted defensive stats.
    For inverted-100 stats: use (a + b - 100) instead of (a - b).
    For inverted-200 stats: use (a + b - 200) instead of (a - b).
    """
    return {
        # Not inverted — simple subtraction
        'wab':      team_stats['wab']      - opp_stats['wab'],
        'barthag':  team_stats['barthag']  - opp_stats['barthag'],
        'adj_tempo': team_stats['adj_tempo'] - opp_stats['adj_tempo'],
        # Inverted-200: offense vs defense
        'adj_oe':   team_stats['adj_oe']   + opp_stats['adj_de']   - 200,
        'adj_de':   team_stats['adj_de']   + opp_stats['adj_oe']   - 200,
        # Inverted-100: offense vs defense
        'efg_pct':  team_stats['efg_pct']  + opp_stats['efgd_pct'] - 100,
        'efgd_pct': team_stats['efgd_pct'] + opp_stats['efg_pct']  - 100,
        'tor':      team_stats['tor']      + opp_stats['tord']     - 100,
        'tord':     team_stats['tord']     + opp_stats['tor']      - 100,
        'orb_pct':  team_stats['orb_pct']  + opp_stats['drb_pct']  - 100,
        'drb_pct':  team_stats['drb_pct']  + opp_stats['orb_pct']  - 100,
        'ftr':      team_stats['ftr']      + opp_stats['ftrd']     - 100,
        'ftrd':     team_stats['ftrd']     + opp_stats['ftr']      - 100,
        '2p_pct':   team_stats['2p_pct']   + opp_stats['2pd_pct']  - 100,
        '2pd_pct':  team_stats['2pd_pct']  + opp_stats['2p_pct']   - 100,
        '3p_pct':   team_stats['3p_pct']   + opp_stats['3pd_pct']  - 100,
        '3pd_pct':  team_stats['3pd_pct']  + opp_stats['3p_pct']   - 100,
        '3pr':      team_stats['3pr']      + opp_stats['3prd']     - 100,
        '3prd':     team_stats['3prd']     + opp_stats['3pr']      - 100,
    }


# Load bracket template (same structure for every year)
bracket_template = pd.read_csv(os.path.join(data_dir, 'bracket_template.csv'))


def build_matchups_for_year(teams_df, year):
    """
    Build all possible matchups for a bracket using the bracket template.
    Uses the same template structure as women_create_matchups.py.
    teams_df must have: team, region, seed, + all raw stat columns.
    Returns DataFrame with differential features (same format models expect).
    """
    # Invert defensive stats first (matching women_create_matchups.py)
    teams_inv = invert_defensive_stats(teams_df)

    # Build lookup: (region, seed) -> team row with inverted stats
    team_lookup = {}
    for _, row in teams_inv.iterrows():
        team_lookup[(row['region'], int(row['seed']))] = row

    rows = []

    for _, tmpl in bracket_template.iterrows():
        t_key = (tmpl['team_region'], int(tmpl['team_seed']))
        o_key = (tmpl['opp_region'], int(tmpl['opp_seed']))

        if t_key not in team_lookup or o_key not in team_lookup:
            continue

        t_stats = team_lookup[t_key]
        o_stats = team_lookup[o_key]

        diff = compute_differentials(t_stats, o_stats)
        row = {
            'year': year,
            'game_id': tmpl['game_id'],
            'round': tmpl['round'],
            'team_region': tmpl['team_region'],
            'team_seed': int(tmpl['team_seed']),
            'team': t_stats['team'],
            'opp_region': tmpl['opp_region'],
            'opp_seed': int(tmpl['opp_seed']),
            'opponent': o_stats['team'],
        }
        row.update(diff)
        rows.append(row)

    return pd.DataFrame(rows)


def predict_win_probs(matchups_df):
    """Add win_prob_raw and win_prob columns to matchups using trained models."""
    early_rounds_pred = ['Round 1', 'Round 2']
    elite_rounds_pred = ['Sweet 16', 'Elite Eight', 'Final Four', 'Championship']

    df = matchups_df.copy()
    df['win_prob_raw'] = 0.0

    early_mask = df['round'].isin(early_rounds_pred)
    if early_mask.sum() > 0:
        df.loc[early_mask, 'win_prob_raw'] = early_model.predict_proba(
            df.loc[early_mask, early_features])[:, 1]

    elite_mask = df['round'].isin(elite_rounds_pred)
    if elite_mask.sum() > 0:
        df.loc[elite_mask, 'win_prob_raw'] = elite_model.predict_proba(
            df.loc[elite_mask, elite_features])[:, 1]

    # Pairwise normalization: for each game_id, normalize the two sides to sum to 1
    df['win_prob'] = 0.0
    normalized_pairs = set()

    for idx in df.index:
        row = df.loc[idx]
        reverse = df[(df['game_id']==row['game_id']) &
                     (df['team']==row['opponent']) &
                     (df['opponent']==row['team'])]
        if len(reverse) == 0:
            df.at[idx, 'win_prob'] = df.at[idx, 'win_prob_raw']
            continue
        rev_idx = reverse.index[0]
        pair_key = tuple(sorted([idx, rev_idx]))
        if pair_key in normalized_pairs:
            continue
        normalized_pairs.add(pair_key)
        p1, p2 = df.at[idx, 'win_prob_raw'], df.at[rev_idx, 'win_prob_raw']
        total = p1 + p2
        df.at[idx, 'win_prob']     = p1/total if total > 0 else 0.5
        df.at[rev_idx, 'win_prob'] = p2/total if total > 0 else 0.5

    return df


def calc_advancement_probs(teams, matchups_df):
    """
    Calculate per-round advancement probabilities for each team.

    For each round transition:
      P(team in next round) = P(team in current round) *
          SUM over possible opponents: P(opponent in current round) * P(team wins matchup)

    Since matchups contain both sides of every game, we only reference each team's
    own matchup rows — the opponent's probability is used only to weight how likely
    that specific matchup is to occur.
    """
    probs = {t: {'Round 1': 1.0} for t in teams}

    round_progression = [
        ('Round 1',    'Round 2',     'Round 1'),
        ('Round 2',    'Sweet 16',    'Round 2'),
        ('Sweet 16',   'Elite Eight', 'Sweet 16'),
        ('Elite Eight','Final Four',  'Elite Eight'),
        ('Final Four', 'Championship','Final Four'),
    ]

    for cur, nxt, matchup_rnd in round_progression:
        for team in teams:
            p_cur = probs[team][cur]
            if p_cur == 0:
                probs[team][nxt] = 0.0
                continue
            tm = matchups_df[(matchups_df['team']==team) & (matchups_df['round']==matchup_rnd)]
            if len(tm) == 0:
                probs[team][nxt] = 0.0
                continue
            adv = 0.0
            for _, row in tm.iterrows():
                opp = row['opponent']
                p_opp = probs[opp][cur]
                adv += p_opp * row['win_prob']
            probs[team][nxt] = p_cur * adv

    # Champion probability
    for team in teams:
        p_final = probs[team]['Championship']
        if p_final == 0:
            probs[team]['Champion'] = 0.0
            continue
        fm = matchups_df[(matchups_df['team']==team) & (matchups_df['round']=='Championship')]
        win_p = 0.0
        for _, row in fm.iterrows():
            opp = row['opponent']
            win_p += probs[opp]['Championship'] * row['win_prob']
        probs[team]['Champion'] = p_final * win_p

    return probs


def calc_bracket_value(probs_dict):
    """
    Calculate bracket expected value:
      Round 1 win = 1 pt, Round 2 = 2, Sweet 16 = 4, Elite 8 = 8, Final 4 = 16, Championship = 32
    """
    return (
        probs_dict.get('Round 2', 0) * 1 +
        probs_dict.get('Sweet 16', 0) * 2 +
        probs_dict.get('Elite Eight', 0) * 4 +
        probs_dict.get('Final Four', 0) * 8 +
        probs_dict.get('Championship', 0) * 16 +
        probs_dict.get('Champion', 0) * 32
    )


# ============================================================================
# STEP 5: PROCESS EACH YEAR
# ============================================================================
print("\nSTEP 5: Processing each year 2021-2026...")

all_teams_output    = []
all_matchups_output = []

for year in range(2021, 2027):
    print(f"\n  Processing {year}...")

    if year == 2026:
        # Use current files directly (these were already built correctly by the individual scripts)
        composites = current_composites.copy()
        matchups_raw = current_matchups.copy()
        matchups_raw['year'] = 2026

        teams = composites['team'].tolist()
        probs = calc_advancement_probs(teams, matchups_raw)

        for _, row in composites.iterrows():
            team = row['team']
            p = probs[team]
            all_teams_output.append({
                'year': 2026,
                'team': team,
                'region': row.get('region', ''),
                'seed': row['seed'],
                'tier': row['tier'],
                'overall': row['overall'],
                'offense': row['offense'],
                'defense': row['defense'],
                'pct_round_2':      round(p.get('Round 2', 0), 4),
                'pct_sweet_16':     round(p.get('Sweet 16', 0), 4),
                'pct_elite_eight':  round(p.get('Elite Eight', 0), 4),
                'pct_final_four':   round(p.get('Final Four', 0), 4),
                'pct_championship': round(p.get('Championship', 0), 4),
                'pct_champion':     round(p.get('Champion', 0), 4),
                'bracket_value':    round(calc_bracket_value(p), 2),
            })

        matchups_out = matchups_raw[['year','game_id','round','team_region','team_seed',
                                     'team','opp_region','opp_seed','opponent',
                                     'win_prob_raw','win_prob']].copy()
        all_matchups_output.append(matchups_out)

    else:
        # Historical year
        yr_teams = teams_hist[teams_hist['year'] == year].copy()
        yr_torvik = torvik_hist[torvik_hist['year'] == year].copy()

        if len(yr_teams) == 0 or len(yr_torvik) == 0:
            print(f"    ⚠ Skipping {year} - missing data")
            continue

        # Merge
        merged = yr_teams.merge(yr_torvik, on='torvik_id', how='left', suffixes=('', '_torvik'))

        # Drop duplicate year cols
        if 'year_torvik' in merged.columns:
            merged = merged.drop(columns=['year_torvik'])
        merged = merged.rename(columns={'year_x': 'year'}) if 'year_x' in merged.columns else merged

        # Score composites and tiers
        scored = score_teams(merged)

        # Build matchups WITH DIFFERENTIALS
        matchups_yr = build_matchups_for_year(scored, year)
        matchups_yr = predict_win_probs(matchups_yr)

        # Advancement probs
        teams_list = scored['team'].tolist()
        probs = calc_advancement_probs(teams_list, matchups_yr)

        for _, row in scored.iterrows():
            team = row['team']
            p = probs[team]
            all_teams_output.append({
                'year': year,
                'team': team,
                'region': row.get('region', ''),
                'seed': row['seed'],
                'tier': row['tier'],
                'overall': row['overall'],
                'offense': row['offense'],
                'defense': row['defense'],
                'pct_round_2':      round(p.get('Round 2', 0), 4),
                'pct_sweet_16':     round(p.get('Sweet 16', 0), 4),
                'pct_elite_eight':  round(p.get('Elite Eight', 0), 4),
                'pct_final_four':   round(p.get('Final Four', 0), 4),
                'pct_championship': round(p.get('Championship', 0), 4),
                'pct_champion':     round(p.get('Champion', 0), 4),
                'bracket_value':    round(calc_bracket_value(p), 2),
            })

        matchups_out = matchups_yr[['year','game_id','round','team_region','team_seed',
                                    'team','opp_region','opp_seed','opponent',
                                    'win_prob_raw','win_prob']].copy()
        all_matchups_output.append(matchups_out)

        # Sanity check: print probability sums and top team
        total_champ = sum(probs[t].get('Champion', 0) for t in teams_list)
        top_team = max(teams_list, key=lambda t: probs[t].get('Champion', 0))
        top_bv = max(teams_list, key=lambda t: calc_bracket_value(probs[t]))
        print(f"    ✓ {len(scored)} teams, {len(matchups_yr)} matchups")
        print(f"      Champion prob sum: {total_champ:.4f} | Top: {top_team}")
        print(f"      Top bracket value: {top_bv} ({calc_bracket_value(probs[top_bv]):.2f})")

# ============================================================================
# STEP 6: SAVE OUTPUTS
# ============================================================================
print("\nSTEP 6: Saving outputs...")

teams_final    = pd.DataFrame(all_teams_output)
matchups_final = pd.concat(all_matchups_output, ignore_index=True)

teams_out_path    = os.path.join(script_dir, '..', 'data', 'women', 'women_teams_output.csv')
matchups_out_path = os.path.join(script_dir, '..', 'data', 'women', 'women_matchups_output.csv')

teams_final.to_csv(teams_out_path, index=False)
matchups_final.to_csv(matchups_out_path, index=False)

print(f"\n  ✓ Saved: {teams_out_path}")
print(f"    {len(teams_final)} rows, years {teams_final['year'].min()}-{teams_final['year'].max()}")

print(f"\n  ✓ Saved: {matchups_out_path}")
print(f"    {len(matchups_final)} rows")

# Show 2026 top 10
print("\n" + "="*80)
print("2026 TOP 10 BY BRACKET VALUE")
print("-"*80)
t2026 = teams_final[teams_final['year']==2026].nlargest(10, 'bracket_value')
print(t2026[['team','seed','tier','bracket_value','pct_champion','pct_championship']].to_string(index=False))
print(f"\n  Total champion probability (2026): {teams_final[teams_final['year']==2026]['pct_champion'].sum():.4f}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)