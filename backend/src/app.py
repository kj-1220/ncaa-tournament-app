"""
Flask API for NCAA Tournament Predictions
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os

# Import our custom modules (we'll create these next)
# from src.models import ModelPredictor
# from src.probabilities import calculate_win_probabilities

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
app.config['DEBUG'] = os.getenv('DEBUG', 'True') == 'True'
app.config['PORT'] = int(os.getenv('PORT', 5000))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'NCAA Tournament API is running'
    }), 200


@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get all teams with predictions"""
    # TODO: Implement this endpoint
    # This will load 2026 data and return predictions
    return jsonify({
        'message': 'Teams endpoint - to be implemented',
        'teams': []
    }), 200


@app.route('/api/teams/<int:team_id>', methods=['GET'])
def get_team(team_id):
    """Get specific team details"""
    # TODO: Implement this endpoint
    return jsonify({
        'message': f'Team {team_id} endpoint - to be implemented'
    }), 200


@app.route('/api/probabilities/heatmap', methods=['GET'])
def get_heatmap_data():
    """Get win probability heatmap for top seeds (1-3)"""
    # TODO: Implement this endpoint
    # This will calculate head-to-head probabilities for seeds 1, 2, 3
    return jsonify({
        'message': 'Heatmap endpoint - to be implemented',
        'data': []
    }), 200


@app.route('/api/scatterplot', methods=['GET'])
def get_scatterplot_data():
    """Get tier vs composite score data for scatterplot"""
    # TODO: Implement this endpoint
    return jsonify({
        'message': 'Scatterplot endpoint - to be implemented',
        'data': []
    }), 200


@app.route('/api/kpis', methods=['GET'])
def get_kpis():
    """Get KPI metrics for dashboard"""
    # TODO: Implement this endpoint
    # Return 4 key metrics to display at top
    return jsonify({
        'message': 'KPIs endpoint - to be implemented',
        'kpis': {
            'total_teams': 0,
            'average_score': 0,
            'top_seed_favorite': 'TBD',
            'upset_potential': 0
        }
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions for custom team data"""
    # TODO: Implement this endpoint
    # Accept team stats and return predictions
    data = request.get_json()
    return jsonify({
        'message': 'Prediction endpoint - to be implemented',
        'prediction': {}
    }), 200


if __name__ == '__main__':
    port = app.config['PORT']
    debug = app.config['DEBUG']
    print(f"Starting NCAA Tournament API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
