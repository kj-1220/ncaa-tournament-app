# Models Directory

This directory contains trained machine learning models for NCAA tournament predictions.

## Expected Files

- `composite_model.pkl` or `composite_model.joblib` - Composite score model
- `tier_model.pkl` or `tier_model.joblib` - Team tier classification model
- `offense_model.pkl` - Offensive rating model
- `defense_model.pkl` - Defensive rating model
- Any other trained models you've created

## Model Parameters

Store model hyperparameters and training configurations here or in a separate `model_params.json` file.

## Important Notes

- Model files are excluded from Git by default (see `.gitignore`)
- For collaboration, consider using Git LFS (Large File Storage) or storing models in cloud storage
- Document model versions and training dates
