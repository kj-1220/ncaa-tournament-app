# Data Directory

This directory contains datasets for training and prediction.

## Expected Files

### Historical Data (for model training)
- `historical_tournament_data.csv` - Past tournament results
- `historical_team_stats.csv` - Team statistics from previous seasons
- `historical_game_results.csv` - Game-by-game results

### Current Season Data (for 2026 predictions)
- `2026_team_stats.csv` - Current season team statistics
- `2026_schedule.csv` - Game schedule and results
- `2026_seeds.csv` - Tournament seeding information

## Data Schema

Document your data schemas here. Example:

### Team Stats Schema
- `team_id`: Unique team identifier
- `team_name`: Team name
- `conference`: Conference affiliation
- `wins`: Number of wins
- `losses`: Number of losses
- `offensive_rating`: Points per 100 possessions
- `defensive_rating`: Points allowed per 100 possessions
- etc.

## Important Notes

- Data files are excluded from Git by default (see `.gitignore`)
- Do not commit sensitive or proprietary data
- Consider using cloud storage for large datasets
