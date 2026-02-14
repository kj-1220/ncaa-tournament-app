"""
Train matchup prediction models for women's basketball
- Logistic Regression for early rounds (with Platt scaling)
- XGBoost for elite rounds (with Platt scaling)
"""
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

print("="*80)
print("TRAINING WOMEN'S MATCHUP PREDICTION MODELS")
print("="*80 + "\n")

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')
models_dir = os.path.join(script_dir, '..', 'models')
Path(models_dir).mkdir(parents=True, exist_ok=True)

# Load training data
print("Loading training data...")
df = pd.read_csv(os.path.join(data_dir, 'women_matchups_training.csv'))
print(f"✓ Loaded {len(df)} games from {df['year'].min()}-{df['year'].max()}\n")

# Split into early and elite rounds
early_rounds = ['First Round', 'Second Round']
elite_rounds = ['Sweet 16', 'Elite Eight', 'Final Four', 'Championship']

early_df = df[df['round'].isin(early_rounds)].copy()
elite_df = df[df['round'].isin(elite_rounds)].copy()

print(f"Early rounds: {len(early_df)} games")
print(f"Elite rounds: {len(elite_df)} games\n")

# ============================================================================
# EARLY ROUNDS MODEL - LOGISTIC REGRESSION + PLATT SCALING
# ============================================================================
print("="*80)
print("TRAINING EARLY ROUNDS MODEL (Logistic Regression + Platt Scaling)")
print("="*80 + "\n")

early_features = ['barthag', 'adj_oe', 'adj_de', 'orb_pct', 'drb_pct', 'ftr', '2p_pct']

X_early = early_df[early_features]
y_early = early_df['win']

# 70/30 split
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    X_early, y_early, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train_early)} games")
print(f"Test set: {len(X_test_early)} games\n")

# Train base model
print("Training base Logistic Regression model...")
base_early_model = LogisticRegression(random_state=42, max_iter=1000)
base_early_model.fit(X_train_early, y_train_early)

# Apply Platt scaling
print("Applying Platt scaling calibration...")
early_model = CalibratedClassifierCV(base_early_model, method='sigmoid', cv='prefit')
early_model.fit(X_test_early, y_test_early)

# Evaluate
y_pred_early = early_model.predict(X_test_early)
y_prob_early = early_model.predict_proba(X_test_early)[:, 1]

print("\nEarly Rounds Model Performance:")
print(f"  Accuracy: {accuracy_score(y_test_early, y_pred_early):.4f}")
print(f"  ROC AUC: {roc_auc_score(y_test_early, y_prob_early):.4f}")
print(f"  Brier Score: {brier_score_loss(y_test_early, y_prob_early):.4f}")

# Save model
early_model_path = os.path.join(models_dir, 'womens_early_rounds.joblib')
joblib.dump(early_model, early_model_path)
print(f"\n✓ Saved calibrated model to: {early_model_path}\n")

# ============================================================================
# ELITE ROUNDS MODEL - XGBOOST + PLATT SCALING
# ============================================================================
print("="*80)
print("TRAINING ELITE ROUNDS MODEL (XGBoost + Platt Scaling)")
print("="*80 + "\n")

elite_features = ['wab', 'barthag', 'adj_oe', 'adj_de', 'efg_pct', 'efgd_pct',
                  'orb_pct', 'drb_pct', '2p_pct', '2pd_pct', '3p_pct', '3pd_pct', '3pr']

X_elite = elite_df[elite_features]
y_elite = elite_df['win']

# 70/30 split
X_train_elite, X_test_elite, y_train_elite, y_test_elite = train_test_split(
    X_elite, y_elite, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train_elite)} games")
print(f"Test set: {len(X_test_elite)} games\n")

# Train base model with your exact parameters
print("Training base XGBoost model...")
base_elite_model = XGBClassifier(
    learning_rate=0.2997738363859162,
    max_depth=9,
    min_child_weight=8.623522034407337,
    subsample=0.8324211691115178,
    colsample_bytree=0.9988769480719698,
    gamma=2.017715776385069,
    reg_alpha=0.9692563913308194,
    reg_lambda=2.5910989850621258,
    n_estimators=335,
    random_state=42,
    eval_metric='logloss'
)

base_elite_model.fit(X_train_elite, y_train_elite)

# Apply Platt scaling
print("Applying Platt scaling calibration...")
elite_model = CalibratedClassifierCV(base_elite_model, method='sigmoid', cv='prefit')
elite_model.fit(X_test_elite, y_test_elite)

# Evaluate
y_pred_elite = elite_model.predict(X_test_elite)
y_prob_elite = elite_model.predict_proba(X_test_elite)[:, 1]

print("\nElite Rounds Model Performance:")
print(f"  Accuracy: {accuracy_score(y_test_elite, y_pred_elite):.4f}")
print(f"  ROC AUC: {roc_auc_score(y_test_elite, y_prob_elite):.4f}")
print(f"  Brier Score: {brier_score_loss(y_test_elite, y_prob_elite):.4f}")

# Save model
elite_model_path = os.path.join(models_dir, 'womens_elite_rounds.joblib')
joblib.dump(elite_model, elite_model_path)
print(f"\n✓ Saved calibrated model to: {elite_model_path}\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nModels saved:")
print(f"  1. {early_model_path}")
print(f"  2. {elite_model_path}")
print(f"\nFeatures:")
print(f"  Early: {early_features}")
print(f"  Elite: {elite_features}")
print(f"\nBoth models calibrated with Platt scaling (sigmoid method)")
print("\nNext step: Run women_calculate_probabilities.py to predict tournament outcomes")