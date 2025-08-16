import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# --- LOAD THE TRAINED MODEL ---
model_path = 'mlb_total_runs_model.pkl'
model = None
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'.")
except Exception as e:
    print(f"An error occurred loading the model: {e}")

# --- LOAD THE PRE-COMPUTED FEATURES ---
features_path = 'latest_features.pkl'
features_df = None
try:
    with open(features_path, 'rb') as file:
        features_df = pickle.load(file)
    print("Pre-computed features loaded successfully.")
except FileNotFoundError:
    print(f"Error: Features file not found at '{features_path}'.")
except Exception as e:
    print(f"An error occurred loading features: {e}")


# --- TEAM NAME MAP ---
TEAM_NAME_MAP = { "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH", "ARZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK" }

@app.route('/features')
def get_features_endpoint():
    """API endpoint to get features for a home and away team from the pre-computed file."""
    home_team_full = request.args.get('home_team')
    away_team_full = request.args.get('away_team')

    if not all([home_team_full, away_team_full]):
        return jsonify({'error': 'Missing home_team or away_team parameter'}), 400

    if features_df is None:
        return jsonify({'error': 'Features file is not loaded on the server.'}), 500

    home_team_abbr = TEAM_NAME_MAP.get(home_team_full, home_team_full)
    away_team_abbr = TEAM_NAME_MAP.get(away_team_full, away_team_full)

    home_feats_list = features_df[features_df['team'] == home_team_abbr].to_dict('records')
    away_feats_list = features_df[features_df['team'] == away_team_abbr].to_dict('records')

    default_feats = {
        'rolling_avg_hits': 8.5, 'rolling_avg_homers': 1.2,
        'starter_rolling_era': 4.2, 'starter_rolling_ks': 5.5,
        'bullpen_rolling_era': 4.0, 'park_factor_avg_runs': 9.0,
        'rolling_avg_hot_hitters': 10.0
    }
    
    home_feats = home_feats_list[0] if home_feats_list else default_feats
    away_feats = away_feats_list[0] if away_feats_list else default_feats

    # --- FIX: Standardize the feature names to match the model's training ---
    final_features = {
        'rolling_avg_hits_home': home_feats.get('rolling_avg_hits', 8.0),
        'rolling_avg_homers_home': home_feats.get('rolling_avg_homers', 1.0),
        'starter_rolling_era_home_starter': home_feats.get('starter_rolling_era', 4.5),
        'starter_rolling_ks_home_starter': home_feats.get('starter_rolling_ks', 5.5),
        'bullpen_rolling_era_home_bullpen': home_feats.get('bullpen_rolling_era', 4.2),
        'rolling_avg_hot_hitters_home_hotness': home_feats.get('rolling_avg_hot_hitters', 10.0),
        'rolling_avg_hits_away': away_feats.get('rolling_avg_hits', 8.0),
        'rolling_avg_homers_away': away_feats.get('rolling_avg_homers', 1.0),
        'starter_rolling_era_away_starter': away_feats.get('starter_rolling_era', 4.5),
        'starter_rolling_ks_away_starter': away_feats.get('starter_rolling_ks', 5.5),
        'bullpen_rolling_era_away_bullpen': away_feats.get('bullpen_rolling_era', 4.2),
        'rolling_avg_hot_hitters_away_hotness': away_feats.get('rolling_avg_hot_hitters', 10.0),
        'temperature': 70, # Placeholder for weather
        'wind_speed': 5,   # Placeholder for weather
        'humidity': 50,    # Placeholder for weather
        'park_factor_avg_runs': home_feats.get('park_factor_avg_runs', 9.0)
    }
    
    return jsonify(final_features)

@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint."""
    if model is None: return jsonify({'error': 'Model is not loaded.'}), 500
    try:
        data = request.get_json()
        features_df = pd.DataFrame([data])
        # Use the model's expected feature names to ensure order
        required_features = model.get_booster().feature_names
        prediction = model.predict(features_df[required_features])
        predicted_runs = float(prediction[0])
        return jsonify({'predicted_total_runs': predicted_runs})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
