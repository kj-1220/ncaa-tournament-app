"""
Generate historical composite scores for scatterplot (2021-2026)
"""
import pandas as pd
import os
import joblib

print("="*80)
print("GENERATING HISTORICAL COMPOSITE SCORES (2021-2026)")
print("="*80 + "\n")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')
models_dir = os.path.join(script_dir, '..', 'models')

# Load trained predictor
print("Loading trained model...")
predictor = joblib.load(os.path.join(models_dir, 'womens_predictor.joblib'))
print("✓ Model loaded\n")

# Load historical data files (2021-2025)
print("Loading historical data files...")
teams_historical = pd.read_csv(os.path.join(data_dir, 'women_teams_historical.csv'))
torvik_historical = pd.read_csv(os.path.join(data_dir, 'women_torvik_historical.csv'))

# Merge datasets
df = teams_historical.merge(torvik_historical, on='torvik_id', how='left', suffixes=('', '_torvik'))
print(f"✓ Loaded {len(df)} teams from {int(df['year'].min())}-{int(df['year'].max())}\n")

# Generate composite scores for historical data
print("Generating composite scores for historical data...")
historical_with_scores = predictor.batch_predict(df)
print("✓ Complete\n")

# Create historical output (2021-2025)
historical_output = pd.DataFrame({
    'year': historical_with_scores['year'].astype(int),
    'team': historical_with_scores['team'],
    'seed': historical_with_scores['seed'].astype(int),
    'bracket_value': 0.0,
    'overall': historical_with_scores['overall'],
    'offense': historical_with_scores['offense'],
    'defense': historical_with_scores['defense'],
    'finish': historical_with_scores['finish']
})

print(f"Historical data prepared: {len(historical_output)} teams\n")

# Load 2026 current teams
print("Loading 2026 teams...")
current_path = os.path.join(data_dir, 'women_composites_current.csv')

if not os.path.exists(current_path):
    print(f"ERROR: {current_path} not found!")
    print("Please run women_create_tiers_current.py first.")
    exit(1)

current_teams = pd.read_csv(current_path)
print(f"✓ Loaded {len(current_teams)} teams from women_composites_current.csv\n")

# Create 2026 output
current_output = pd.DataFrame({
    'year': 2026,
    'team': current_teams['team'],
    'seed': current_teams['seed'].astype(int),
    'bracket_value': current_teams['bracket_value'],
    'overall': current_teams['overall'],
    'offense': current_teams['offense'],
    'defense': current_teams['defense'],
    'finish': 'TBD'
})

print(f"2026 data prepared: {len(current_output)} teams\n")

# Combine both
print("Combining historical and current data...")
output = pd.concat([historical_output, current_output], ignore_index=True)
print(f"✓ Combined: {len(output)} total teams\n")

# Verify the combine worked
print("Verification:")
print(f"  Historical (2021-2025): {len(output[output['year'] < 2026])} teams")
print(f"  Current (2026): {len(output[output['year'] == 2026])} teams")
print(f"  Year range: {int(output['year'].min())}-{int(output['year'].max())}\n")

# Save
output_path = os.path.join(data_dir, 'women_composites_historical.csv')
output.to_csv(output_path, index=False)

print("="*80)
print("OUTPUT SAVED")
print("="*80)
print(f"\nFile: {output_path}")
print(f"Total teams: {len(output)}")
print(f"\nFirst 3 rows:")
print(output.head(3).to_string(index=False))
print(f"\nLast 3 rows (should be 2026 teams):")
print(output.tail(3).to_string(index=False))

print("\n" + "="*80)
print("COMPLETE - Ready for scatterplot visualization")
print("="*80)