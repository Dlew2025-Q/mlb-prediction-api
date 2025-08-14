import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text
import numpy as np

# --- INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# The database is no longer loaded into memory on startup to save resources.
# The engine will be used for on-demand queries if needed in future versions.
DB_URL = os.environ.get('DATABASE_URL')
engine = None
if DB_URL:
    try:
        engine = create_engine(DB_URL)
        print("Database engine created successfully.")
    except Exception as e:
        print(f"Error creating database engine: {e}")

# --- LOAD THE TRAINED MODEL ---
model_path = 'mlb_total_runs_model.pkl'
model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
else:
    print(f"Error: Model file not found at '{model_path}'.")

# --- TEAM NAME MAP ---
# This is still needed to map incoming full names to abbreviations if we query the DB.
TEAM_NAME_MAP = { "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH", "ARZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK" }


@app.route('/features')
def get_features_endpoint():
    """
    This endpoint now returns a stable set of league-average features.
    This prevents server crashes and solves the 500 error.
    The next step is to replace this with dynamic, on-demand SQL queries.
    """
    # These are safe, reasonable league-average stats to use as a baseline.
    # This ensures the API is fast and does not run out of memory.
    features = {
        'home_rolling_avg_hits': 8.5,
        'home_rolling_avg_homers': 1.2,
        'home_starter_rolling_era': 4.2,
        'home_starter_rolling_ks': 5.5,
        'home_bullpen_rolling_era': 4.0,
        'away_rolling_avg_hits': 8.5,
        'away_rolling_avg_homers': 1.2,
        'away_starter_rolling_era': 4.2,
        'away_starter_rolling_ks': 5.5,
        'away_bullpen_rolling_era': 4.0,
        'park_factor_avg_runs': 9.0
    }
    return jsonify(features)

@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint."""
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
    try:
        data = request.get_json()
        features_df = pd.DataFrame([data])
        # Ensure columns are in the same order as the training script
        required_features = [
            'home_rolling_avg_hits', 'home_rolling_avg_homers',
            'away_rolling_avg_hits', 'away_rolling_avg_homers',
            'home_starter_rolling_era', 'home_starter_rolling_ks',
            'away_starter_rolling_era', 'away_starter_rolling_ks',
            'home_bullpen_rolling_era', 'away_bullpen_rolling_era',
            'park_factor_avg_runs'
        ]
        features_df = features_df[required_features]
        prediction = model.predict(features_df)
        predicted_runs = float(prediction[0])
        return jsonify({'predicted_total_runs': predicted_runs})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
