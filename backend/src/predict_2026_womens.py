"""
Apply trained models to 2026 women's tournament teams
"""
import pandas as pd
import os
import joblib

print("="*80)
print("2026 WOMEN'S TOURNAMENT PREDICTIONS")
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
    print("Please run train_womens_models.py first to train the model.")
    exit(1)

predictor = joblib.load(predictor_path)
print("✓ Model loaded successfully\n")

# Load 2026 team data
print("Loading 2026 team data...")
teams_2026 = pd.read_csv(os.path.join(data_dir, '2026_team_stats.csv'))
print(f"✓ Loaded {len(teams_2026)} teams\n")

# Make predictions
print("Making predictions...")
predictions = predictor.batch_predict(teams_2026)
print("✓ Predictions complete\n")

# Display results
print("="*80)
print("TOP 10 TEAMS")
print("="*80 + "\n")

top_10 = predictions.nsmallest(10, 'rank')
display_cols = ['rank', 'team', 'seed', 'region', 'tier', 'overall', 'offense', 'defense']
print(top_10[display_cols].to_string(index=False))

# Tier breakdown
print("\n" + "="*80)
print("TIER BREAKDOWN")
print("="*80 + "\n")

tier_counts = predictions['tier'].value_counts().sort_index()
for tier, count in tier_counts.items():
    print(f"Tier {tier}: {count} teams")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80 + "\n")

output_path = os.path.join(data_dir, '2026_womens_predictions.csv')
predictions.to_csv(output_path, index=False)
print(f"✓ Saved predictions to: {output_path}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80 + "\n")

print(f"Average Overall Score: {predictions['overall'].mean():.2f}")
print(f"Average Offensive Score: {predictions['offense'].mean():.2f}")
print(f"Average Defensive Score: {predictions['defense'].mean():.2f}")
print(f"\nTop Team: {top_10.iloc[0]['team']}")
print(f"S Tier Teams: {len(predictions[predictions['tier']=='S'])}")

print("\n" + "="*80)
print("PREDICTIONS COMPLETE!")
print("="*80)
print(f"\nResults saved to: {output_path}")
print("Ready to use in the web app!")