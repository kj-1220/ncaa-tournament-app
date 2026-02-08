"""
Apply composite and tier models to current women's tournament teams
Creates output for tournament table display
"""
import pandas as pd
import os
import joblib

print("="*80)
print("APPLYING COMPOSITE & TIER MODELS TO CURRENT TEAMS")
print("="*80 + "\n")

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')
models_dir = os.path.join(script_dir, '..', 'models')

# Load trained predictor
print("Loading trained model...")
predictor_path = os.path.join(models_dir, 'womens_predictor.joblib')

if not os.path.exists(predictor_path):
    print("ERROR: Trained model not found!")
    print("Please run train_womens_models.py first.")
    exit(1)

predictor = joblib.load(predictor_path)
print("✓ Model loaded\n")

# Load current team data
print("Loading current teams...")
teams_current = pd.read_csv(os.path.join(data_dir, 'women_teams_current.csv'))

# Filter to only tournament teams
tournament_teams = teams_current[teams_current['seed'].notna()].copy()
print(f"✓ {len(tournament_teams)} tournament teams\n")

# Make predictions
print("Generating composite scores and tiers...")
predictions = predictor.batch_predict(tournament_teams)
print("✓ Complete\n")

# Create output with only needed columns
output = pd.DataFrame()
output['team'] = predictions['team']
output['region'] = predictions['region']
output['seed'] = predictions['seed'].astype(int)
output['tier'] = predictions['tier']
output['bracket_value'] = 0.0  # Placeholder - will be calculated later
output['overall'] = predictions['overall']
output['offense'] = predictions['offense']
output['defense'] = predictions['defense']
output['round_2_prob'] = 0.0  # Placeholder - will be calculated by matchup models
output['sweet_16_prob'] = 0.0
output['elite_8_prob'] = 0.0
output['final_4_prob'] = 0.0
output['championship_prob'] = 0.0
output['champion_prob'] = 0.0

# Sort by overall score descending
output = output.sort_values('overall', ascending=False).reset_index(drop=True)

# Save
output_path = os.path.join(data_dir, 'women_composites_current.csv')
output.to_csv(output_path, index=False)

print("="*80)
print("OUTPUT SAVED")
print("="*80)
print(f"\nFile: {output_path}")
print(f"Teams: {len(output)}")
print(f"\nTop 5 Teams:")
print(output[['team', 'seed', 'tier', 'overall', 'offense', 'defense']].head().to_string(index=False))
print(f"\nTier Breakdown:")
for tier in ['S', 'A', 'B', 'C', 'D']:
    count = len(output[output['tier'] == tier])
    if count > 0:
        print(f"  Tier {tier}: {count} teams")

print("\n" + "="*80)
print("NEXT STEP: Run matchup prediction models to calculate probabilities")
print("="*80)