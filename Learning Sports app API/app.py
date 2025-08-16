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

# --- CONFIGURATION & DATABASE CONNECTION ---
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
TEAM_NAME_MAP = { "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH", "ARZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK" }

def get_team_features(team_abbr):
    """Calculates the latest features for a single team using direct, efficient SQL queries."""
    if engine is None: return {}
    
    default_features = {
        "rolling_avg_hits": 8.0, "rolling_avg_homers": 1.0,
        "starter_rolling_era": 4.5, "starter_rolling_ks": 5.0,
        "bullpen_rolling_era": 4.2, "park_factor_avg_runs": 9.0
    }

    try:
        with engine.connect() as conn:
            hitting_query = text("""
                SELECT AVG(total_hits), AVG(total_homers) FROM (
                    SELECT SUM(b.hits) as total_hits, SUM(b.home_runs) as total_homers
                    FROM games g JOIN batter_stats b ON g.game_id = b.game_id
                    WHERE b.team = :team_name
                    GROUP BY g.game_id, b.team
                    ORDER BY g.commence_time DESC LIMIT 10
                ) as recent_games;
            """)
            hitting_res = conn.execute(hitting_query, {"team_name": team_abbr}).fetchone()

            park_factor_query = text("SELECT AVG(home_score + away_score) FROM games WHERE home_team = :team_name;")
            park_factor_res = conn.execute(park_factor_query, {"team_name": team_abbr}).fetchone()

            features = default_features.copy()
            if hitting_res and hitting_res[0] is not None:
                features["rolling_avg_hits"] = hitting_res[0]
                features["rolling_avg_homers"] = hitting_res[1]
            if park_factor_res and park_factor_res[0] is not None:
                features["park_factor_avg_runs"] = park_factor_res[0]
            
            return features
            
    except Exception as e:
        print(f"Error calculating features for {team_abbr}: {e}")
        return default_features

@app.route('/features')
def get_features_endpoint():
    """API endpoint to get features for a home and away team."""
    home_team_full = request.args.get('home_team')
    away_team_full = request.args.get('away_team')

    if not all([home_team_full, away_team_full]):
        return jsonify({'error': 'Missing home_team or away_team parameter'}), 400

    home_team_abbr = TEAM_NAME_MAP.get(home_team_full, home_team_full)
    away_team_abbr = TEAM_NAME_MAP.get(away_team_full, away_team_full)

    home_feats = get_team_features(home_team_abbr)
    away_feats = get_team_features(away_team_abbr)

    # --- FIX: Standardize the feature names to match the model's training ---
    final_features = {
        'rolling_avg_hits_home': home_feats['rolling_avg_hits'],
        'rolling_avg_homers_home': home_feats['rolling_avg_homers'],
        'starter_rolling_era_home_starter': home_feats['starter_rolling_era'],
        'starter_rolling_ks_home_starter': home_feats['starter_rolling_ks'],
        'bullpen_rolling_era_home_bullpen': home_feats['bullpen_rolling_era'],
        'rolling_avg_hits_away': away_feats['rolling_avg_hits'],
        'rolling_avg_homers_away': away_feats['rolling_avg_homers'],
        'starter_rolling_era_away_starter': away_feats['starter_rolling_era'],
        'starter_rolling_ks_away_starter': away_feats['starter_rolling_ks'],
        'bullpen_rolling_era_away_bullpen': away_feats['bullpen_rolling_era'],
        'rolling_avg_hot_hitters_home_hotness': 10.0, # Placeholder
        'rolling_avg_hot_hitters_away_hotness': 10.0, # Placeholder
        'temperature': 70, # Placeholder
        'wind_speed': 5,   # Placeholder
        'humidity': 50,    # Placeholder
        'park_factor_avg_runs': home_feats['park_factor_avg_runs']
    }
    
    return jsonify(final_features)

@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint."""
    if model is None: return jsonify({'error': 'Model is not loaded.'}), 500
    try:
        data = request.get_json()
        features_df = pd.DataFrame([data])
        # Use the exact feature names the model was trained on
        required_features = model.get_booster().feature_names
        features_df = features_df[required_features]
        prediction = model.predict(features_df)
        predicted_runs = float(prediction[0])
        return jsonify({'predicted_total_runs': predicted_runs})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
