"""Flask API for NCAA Women's Basketball Tournament Predictions"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

DATA_DIR = Path(__file__).parent / 'data' / 'women'

teams_data = None
matchups_data = None
historical_data = None
bracket_template_data = None

def load_csv_data():
    global teams_data, matchups_data, historical_data, bracket_template_data
    
    teams_data = pd.read_csv(DATA_DIR / 'women_composites_current.csv')
    matchups_data = pd.read_csv(DATA_DIR / 'women_matchups_with_probs.csv')
    historical_data = pd.read_csv(DATA_DIR / 'women_composites_historical.csv')
    bracket_template_data = pd.read_csv(DATA_DIR / 'bracket_template.csv')
    
    print(f"✓ Loaded {len(teams_data)} teams")
    print(f"✓ Loaded {len(matchups_data)} matchups")
    print(f"✓ Loaded {len(historical_data)} historical records")
    print(f"✓ Loaded {len(bracket_template_data)} bracket template entries")

def clean_df(df):
    return df.where(pd.notnull(df), None).to_dict('records')

@app.route('/api/women/teams', methods=['GET'])
def get_teams():
    return jsonify(clean_df(teams_data)), 200

@app.route('/api/women/teams/<team_name>', methods=['GET'])
def get_team(team_name):
    team = teams_data[teams_data['team'].str.lower() == team_name.lower()]
    if team.empty:
        return jsonify({"error": f"Team '{team_name}' not found"}), 404
    return jsonify(clean_df(team)[0]), 200

@app.route('/api/women/matchups', methods=['GET'])
def get_matchups():
    team_filter = request.args.get('team')
    if team_filter:
        filtered = matchups_data[
            (matchups_data['team'].str.lower() == team_filter.lower()) |
            (matchups_data['opponent'].str.lower() == team_filter.lower())
        ]
        return jsonify(clean_df(filtered)), 200
    return jsonify(clean_df(matchups_data)), 200

@app.route('/api/women/matchups/<team_name>', methods=['GET'])
def get_team_matchups(team_name):
    team_matchups = matchups_data[matchups_data['team'].str.lower() == team_name.lower()]
    if team_matchups.empty:
        return jsonify({"error": f"No matchups found for '{team_name}'"}), 404
    return jsonify(clean_df(team_matchups)), 200

@app.route('/api/women/historical', methods=['GET'])
def get_historical():
    filtered = historical_data.copy()
    if request.args.get('year'):
        filtered = filtered[filtered['year'] == int(request.args.get('year'))]
    if request.args.get('tier'):
        filtered = filtered[filtered['tier'] == request.args.get('tier').upper()]
    return jsonify(clean_df(filtered)), 200

@app.route('/api/women/bracket-template', methods=['GET'])
def get_bracket_template():
    return jsonify(clean_df(bracket_template_data)), 200

@app.route('/api/women/stats', methods=['GET'])
def get_stats():
    stats = {
        "total_teams": len(teams_data),
        "total_matchups": len(matchups_data),
        "historical_records": len(historical_data),
        "tiers": teams_data['tier'].value_counts().to_dict(),
        "regions": teams_data['region'].value_counts().to_dict(),
        "top_5_teams": clean_df(teams_data.nlargest(5, 'bracket_value')[
            ['team', 'seed', 'tier', 'bracket_value', 'champion_prob']
        ])
    }
    return jsonify(stats), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "data_loaded": teams_data is not None}), 200

if __name__ == '__main__':
    print("Loading CSV data...")
    load_csv_data()
    print("\nStarting Flask API server...")
    print("API available at: http://localhost:5001\n")
    app.run(debug=True, port=5001, host='0.0.0.0')
