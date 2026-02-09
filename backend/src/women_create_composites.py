"""
Generate composite scores and tiers for women's basketball
Trains models on historical data (2021-2025), then applies to all teams (2021-2026)
"""
import pandas as pd
import os
import joblib
from pathlib import Path
import sys

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import NCAAPredictor

print("="*80)
print("WOMEN'S BASKETBALL COMPOSITE SCORES & TIERS")
print("="*80 + "\n")

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')
models_dir = os.path.join(script_dir, '..', 'models')
Path(models_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: TRAIN MODELS ON HISTORICAL DATA (2021-2025)
# ============================================================================
print("STEP 1: Training models on historical data (2021-2025)")
print("-" * 80)

predictor = NCAAPredictor(historical_data_path=data_dir)

print("Loading historical data...")
predictor.load_historical_data(
    tournament_file='women_teams_historical.csv',
    torvik_file='women_torvik_historical.csv'
)

print("\nTraining composite scoring model...")
predictor.train_composite_model()

print("\nTraining tier clustering model...")
predictor.train_tier_model()

# Save trained model
model_path = os.path.join(models_dir, 'womens_predictor.joblib')
joblib.dump(predictor, model_path)
print(f"\n✓ Saved trained model to: {model_path}")

# ============================================================================
# STEP 2: GENERATE HISTORICAL COMPOSITES (2021-2025)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Generating composite scores for historical data (2021-2025)")
print("-" * 80)

historical_with_scores = predictor.batch_predict(predictor.historical_data)

historical_output = pd.DataFrame({
    'year': historical_with_scores['year'].astype(int),
    'team': historical_with_scores['team'],
    'seed': historical_with_scores['seed'].astype(int),
    'tier': historical_with_scores['tier'],
    'bracket_value': 0.0,
    'overall': historical_with_scores['overall'],
    'offense': historical_with_scores['offense'],
    'defense': historical_with_scores['defense'],
    'finish': historical_with_scores['finish']
})

print(f"✓ Generated scores for {len(historical_output)} historical teams")

# ============================================================================
# STEP 3: GENERATE CURRENT COMPOSITES (2026)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Generating composite scores for 2026 tournament teams")
print("-" * 80)

# Load 2026 teams
current_teams = pd.read_csv(os.path.join(data_dir, 'women_teams_enriched.csv'))
tournament_teams = current_teams[current_teams['seed'].notna()].copy()
print(f"Loaded {len(tournament_teams)} tournament teams")

# Generate predictions
predictions = predictor.batch_predict(tournament_teams)

# Create current output
current_output = pd.DataFrame({
    'team': predictions['team'],
    'region': predictions['region'],
    'seed': predictions['seed'].astype(int),
    'tier': predictions['tier'],
    'bracket_value': 0.0,
    'overall': predictions['overall'],
    'offense': predictions['offense'],
    'defense': predictions['defense'],
    'round_2_prob': 0.0,
    'sweet_16_prob': 0.0,
    'elite_8_prob': 0.0,
    'final_4_prob': 0.0,
    'championship_prob': 0.0,
    'champion_prob': 0.0
})

# Sort by overall score
current_output = current_output.sort_values('overall', ascending=False).reset_index(drop=True)

# Save current output
current_path = os.path.join(data_dir, 'women_composites_current.csv')
current_output.to_csv(current_path, index=False)
print(f"✓ Saved to: {current_path}")

# Show top teams
print(f"\nTop 5 Teams:")
print(current_output[['team', 'seed', 'tier', 'overall', 'offense', 'defense']].head().to_string(index=False))

# Show tier breakdown
print(f"\nTier Breakdown:")
for tier in ['S', 'A', 'B', 'C', 'D']:
    count = len(current_output[current_output['tier'] == tier])
    if count > 0:
        print(f"  Tier {tier}: {count} teams")

# ============================================================================
# STEP 4: COMBINE HISTORICAL + CURRENT FOR SCATTERPLOT
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Creating combined historical dataset (2021-2026)")
print("-" * 80)

# Prepare 2026 for historical file
current_for_scatter = pd.DataFrame({
    'year': 2026,
    'team': current_output['team'],
    'seed': current_output['seed'],
    'tier': current_output['tier'],
    'bracket_value': current_output['bracket_value'],
    'overall': current_output['overall'],
    'offense': current_output['offense'],
    'defense': current_output['defense'],
    'finish': 'TBD'
})

# Combine
combined = pd.concat([historical_output, current_for_scatter], ignore_index=True)

# Save
historical_path = os.path.join(data_dir, 'women_composites_historical.csv')
combined.to_csv(historical_path, index=False)
print(f"✓ Saved to: {historical_path}")
print(f"  Total teams: {len(combined)}")
print(f"  2021-2025: {len(combined[combined['year'] < 2026])} teams")
print(f"  2026: {len(combined[combined['year'] == 2026])} teams")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nOutput files:")
print(f"  1. {current_path}")
print(f"  2. {historical_path}")
print(f"  3. {model_path}")
print("\nNext step: Run matchup prediction models to calculate win probabilities")