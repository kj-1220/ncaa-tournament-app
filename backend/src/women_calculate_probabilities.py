"""
Calculate win probabilities and bracket values for women's tournament
Uses trained matchup models with proper pairwise probability normalization
"""
import pandas as pd
import numpy as np
import os
import joblib

print("="*80)
print("CALCULATING WOMEN'S TOURNAMENT PROBABILITIES")
print("="*80 + "\n")

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')
models_dir = os.path.join(script_dir, '..', 'models')

# ============================================================================
# STEP 1: LOAD MODELS
# ============================================================================
print("STEP 1: Loading trained matchup models")
print("-" * 80 + "\n")

early_model = joblib.load(os.path.join(models_dir, 'womens_early_rounds.joblib'))
elite_model = joblib.load(os.path.join(models_dir, 'womens_elite_rounds.joblib'))

print("✓ Loaded early rounds model (Logistic Regression + Platt)")
print("✓ Loaded elite rounds model (XGBoost + Platt)\n")

# ============================================================================
# STEP 2: LOAD MATCHUP DATA
# ============================================================================
print("STEP 2: Loading matchup data")
print("-" * 80 + "\n")

matchups = pd.read_csv(os.path.join(data_dir, 'women_matchups_current.csv'))
print(f"✓ Loaded {len(matchups)} possible matchups")

# Verify game_id exists
if 'game_id' not in matchups.columns:
    print("ERROR: game_id column not found in matchups!")
    exit(1)

print(f"✓ Found {matchups['game_id'].nunique()} unique games\n")

# ============================================================================
# STEP 3: PREDICT WIN PROBABILITIES FOR ALL MATCHUPS
# ============================================================================
print("STEP 3: Predicting win probabilities for all matchups")
print("-" * 80 + "\n")

# Features for each model
early_features = ['barthag', 'adj_oe', 'adj_de', 'orb_pct', 'drb_pct', 'ftr', '2p_pct']
elite_features = ['wab', 'barthag', 'adj_oe', 'adj_de', 'efg_pct', 'efgd_pct',
                  'orb_pct', 'drb_pct', '2p_pct', '2pd_pct', '3p_pct', '3pd_pct', '3pr']

# Round names
early_rounds = ['Round 1', 'Round 2']
elite_rounds = ['Sweet 16', 'Elite Eight', 'Final Four', 'Championship']

# Add win probability column
matchups['win_prob_raw'] = 0.0

# Predict for early rounds
early_mask = matchups['round'].isin(early_rounds)
if early_mask.sum() > 0:
    X_early = matchups.loc[early_mask, early_features]
    matchups.loc[early_mask, 'win_prob_raw'] = early_model.predict_proba(X_early)[:, 1]
    print(f"✓ Predicted {early_mask.sum()} early round matchups")

# Predict for elite rounds
elite_mask = matchups['round'].isin(elite_rounds)
if elite_mask.sum() > 0:
    X_elite = matchups.loc[elite_mask, elite_features]
    matchups.loc[elite_mask, 'win_prob_raw'] = elite_model.predict_proba(X_elite)[:, 1]
    print(f"✓ Predicted {elite_mask.sum()} elite round matchups\n")

# ============================================================================
# STEP 4: NORMALIZE PROBABILITIES PAIRWISE
# ============================================================================
print("STEP 4: Normalizing probabilities for opposing perspectives")
print("-" * 80 + "\n")

matchups['win_prob'] = 0.0
matchups['normalized'] = False

# For each matchup, find its reverse and normalize the pair
for idx in matchups.index:
    # Skip if already normalized
    if matchups.at[idx, 'normalized']:
        continue
    
    row = matchups.loc[idx]
    team = row['team']
    opponent = row['opponent']
    game_id = row['game_id']
    
    # Find reverse matchup (opponent vs team, same game_id)
    reverse = matchups[
        (matchups['game_id'] == game_id) &
        (matchups['team'] == opponent) &
        (matchups['opponent'] == team)
    ]
    
    if len(reverse) == 0:
        # No reverse found, just use raw probability
        matchups.at[idx, 'win_prob'] = matchups.at[idx, 'win_prob_raw']
        matchups.at[idx, 'normalized'] = True
        continue
    
    reverse_idx = reverse.index[0]
    
    # Normalize the pair
    prob1 = matchups.at[idx, 'win_prob_raw']
    prob2 = matchups.at[reverse_idx, 'win_prob_raw']
    total = prob1 + prob2
    
    if total > 0:
        matchups.at[idx, 'win_prob'] = prob1 / total
        matchups.at[reverse_idx, 'win_prob'] = prob2 / total
    else:
        matchups.at[idx, 'win_prob'] = 0.5
        matchups.at[reverse_idx, 'win_prob'] = 0.5
    
    matchups.at[idx, 'normalized'] = True
    matchups.at[reverse_idx, 'normalized'] = True

matchups = matchups.drop(columns=['normalized'])

print("✓ Normalized all pairwise probabilities")

# Verify normalization
sample_game = matchups[matchups['game_id'] == 'r1_r1_g01']
print(f"✓ Sample verification: game_id 'r1_r1_g01'")
uconn = sample_game[sample_game['team'] == 'Connecticut']
if len(uconn) > 0:
    print(f"  Connecticut vs {uconn.iloc[0]['opponent']}: {uconn.iloc[0]['win_prob']:.4f}")
    opp_name = uconn.iloc[0]['opponent']
    reverse = sample_game[sample_game['team'] == opp_name]
    if len(reverse) > 0:
        print(f"  {opp_name} vs Connecticut: {reverse.iloc[0]['win_prob']:.4f}")
        print(f"  Sum: {uconn.iloc[0]['win_prob'] + reverse.iloc[0]['win_prob']:.4f}\n")

# Save matchups with probabilities
matchups_output = os.path.join(data_dir, 'women_matchups_with_probs.csv')
matchups.to_csv(matchups_output, index=False)
print(f"✓ Saved to: {matchups_output}\n")

# ============================================================================
# STEP 5: CALCULATE ADVANCEMENT PROBABILITIES
# ============================================================================
print("STEP 5: Calculating advancement probabilities")
print("-" * 80 + "\n")

# Load team composites
composites = pd.read_csv(os.path.join(data_dir, 'women_composites_current.csv'))

# Get unique teams
teams = composites['team'].unique()

# Initialize probability dictionary
probs = {team: {} for team in teams}

# Everyone starts in Round 1 with 100% probability
for team in teams:
    probs[team]['Round 1'] = 1.0

# Round progression
round_progression = [
    ('Round 1', 'Round 2', 'Round 1'),
    ('Round 2', 'Sweet 16', 'Round 2'),
    ('Sweet 16', 'Elite Eight', 'Sweet 16'),
    ('Elite Eight', 'Final Four', 'Elite Eight'),
    ('Final Four', 'Championship', 'Final Four')
]

print("Calculating round-by-round probabilities...")

# Process each round
for current_round, next_round, matchup_round in round_progression:
    print(f"  Processing {current_round} -> {next_round}")
    
    # For each team, calculate probability of advancing
    for team in teams:
        p_team_current = probs[team][current_round]
        
        if p_team_current == 0:
            probs[team][next_round] = 0.0
            continue
        
        # Get matchups for current round
        team_matchups = matchups[
            (matchups['team'] == team) & 
            (matchups['round'] == matchup_round)
        ]
        
        if len(team_matchups) == 0:
            probs[team][next_round] = 0.0
            continue
        
        # Calculate probability of advancing
        advance_prob = 0.0
        
        for _, matchup in team_matchups.iterrows():
            opponent = matchup['opponent']
            p_opponent_current = probs[opponent][current_round]
            p_team_wins = matchup['win_prob']
            
            advance_prob += p_opponent_current * p_team_wins
        
        probs[team][next_round] = p_team_current * advance_prob
    
    # Verify probability conservation
    total_prob = sum(probs[team][next_round] for team in teams)
    print(f"    Total probability in {next_round}: {total_prob:.4f}")

print("\n✓ Calculated advancement probabilities\n")

# ============================================================================
# STEP 6: CALCULATE CHAMPION PROBABILITIES
# ============================================================================
print("STEP 6: Calculating champion probabilities")
print("-" * 80 + "\n")

champion_probs = {}

for team in teams:
    p_in_final = probs[team]['Championship']
    
    if p_in_final == 0:
        champion_probs[team] = 0.0
        continue
    
    # Get all possible championship matchups
    final_matchups = matchups[
        (matchups['team'] == team) & 
        (matchups['round'] == 'Championship')
    ]
    
    # Calculate probability of winning championship
    win_prob = 0.0
    
    for _, matchup in final_matchups.iterrows():
        opponent = matchup['opponent']
        p_opponent_final = probs[opponent]['Championship']
        p_team_wins = matchup['win_prob']
        
        win_prob += p_opponent_final * p_team_wins
    
    champion_probs[team] = p_in_final * win_prob

print("✓ Calculated champion probabilities")
total_champion_prob = sum(champion_probs.values())
print(f"✓ Total champion probability: {total_champion_prob:.4f} (should be 1.0)\n")

# ============================================================================
# STEP 7: UPDATE COMPOSITE DATA
# ============================================================================
print("STEP 7: Updating composite data")
print("-" * 80 + "\n")

# Map round names to column names
round_to_col = {
    'Round 2': 'round_2_prob',
    'Sweet 16': 'sweet_16_prob',
    'Elite Eight': 'elite_8_prob',
    'Final Four': 'final_4_prob',
    'Championship': 'championship_prob',
}

# Initialize columns
for col in round_to_col.values():
    composites[col] = 0.0
composites['champion_prob'] = 0.0

# Fill probabilities
for idx, row in composites.iterrows():
    team = row['team']
    
    for round_name, col_name in round_to_col.items():
        composites.at[idx, col_name] = probs[team][round_name]
    
    composites.at[idx, 'champion_prob'] = champion_probs[team]

print("✓ Updated composite data\n")

# ============================================================================
# STEP 8: CALCULATE BRACKET VALUE
# ============================================================================
print("STEP 8: Calculating bracket value")
print("-" * 80 + "\n")

# Bracket value = expected points
composites['bracket_value'] = (
    composites['round_2_prob'] * 1 +
    composites['sweet_16_prob'] * 2 +
    composites['elite_8_prob'] * 4 +
    composites['final_4_prob'] * 8 +
    composites['championship_prob'] * 16 +
    composites['champion_prob'] * 32
).round(2)

print("✓ Calculated bracket values\n")

print("Top 10 Teams by Bracket Value:")
top_10 = composites.nlargest(10, 'bracket_value')[
    ['team', 'seed', 'tier', 'bracket_value', 'champion_prob', 'championship_prob']
]
print(top_10.to_string(index=False))

print(f"\n✓ Total champion probability: {composites['champion_prob'].sum():.4f}")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Saving results")
print("-" * 80 + "\n")

# Save current composites
current_output = os.path.join(data_dir, 'women_composites_current.csv')
composites.to_csv(current_output, index=False)
print(f"✓ Saved: {current_output}")

# Update historical file
historical = pd.read_csv(os.path.join(data_dir, 'women_composites_historical.csv'))

for idx, row in composites.iterrows():
    team = row['team']
    mask = (historical['year'] == 2026) & (historical['team'] == team)
    historical.loc[mask, 'bracket_value'] = row['bracket_value']

historical_output = os.path.join(data_dir, 'women_composites_historical.csv')
historical.to_csv(historical_output, index=False)
print(f"✓ Saved: {historical_output}")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nOutput files:")
print(f"  1. {matchups_output}")
print(f"  2. {current_output}")
print(f"  3. {historical_output}")
print("\nReady for API integration!")