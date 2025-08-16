#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import date

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- LOAD MODEL AND FEATURES ON STARTUP ---
# Load the pre-trained XGBoost model.
try:
    with open('mlb_total_runs_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Store the expected feature names from the model. This is crucial for validation.
    MODEL_FEATURES = model.get_booster().feature_names
    print("Model loaded successfully.")
    print(f"Model expects {len(MODEL_FEATURES)} features.")
except Exception as e:
    model = None
    MODEL_FEATURES = []
    print(f"CRITICAL ERROR: Could not load model. API will not be able to make predictions. Error: {e}")

# Load the pre-computed features DataFrame.
try:
    with open('latest_features.pkl', 'rb') as file:
        features_df = pickle.load(file)
    print("Pre-computed features loaded successfully.")
except Exception as e:
    features_df = None
    print(f"CRITICAL ERROR: Could not load pre-computed features. API will not work. Error: {e}")

# --- CONFIGURATION & UTILITIES ---
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
# This mapping must be identical to the one in precompute_features.py
TEAM_NAME_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", 
    "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", 
    "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", 
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", 
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", 
    "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", 
    "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH",
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS",
    "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE",
    "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM",
    "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT",
    "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL",
    "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH",
    "ARZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK"
}

# --- API ENDPOINTS ---

@app.route('/games')
def get_games():
    """
    Proxy endpoint to securely fetch game odds from The Odds API.
    The frontend calls this instead of calling the API directly.
    """
    if not ODDS_API_KEY:
        return jsonify({'error': 'API key is not configured on the server.'}), 500
    
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?apiKey={ODDS_API_KEY}&regions=us&markets=totals"
    try:
        response = requests.get(url)
        response.raise_for_status()
        games = response.json()
        # Filter for today's games on the server-side.
        today_str = date.today().isoformat()
        today_games = [g for g in games if g['commence_time'].startswith(today_str)]
        return jsonify(today_games)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from The Odds API: {e}'}), 502 # Bad Gateway

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives game data, combines it with pre-computed features, and returns a model prediction.
    """
    if model is None or features_df is None:
        return jsonify({'error': 'Server is not ready; model or features not loaded.'}), 503 # Service Unavailable

    try:
        data = request.get_json()
        home_team_full = data.get('home_team')
        away_team_full = data.get('away_team')

        if not all([home_team_full, away_team_full]):
            return jsonify({'error': 'Missing home_team or away_team in request body'}), 400

        # Standardize team names to match our feature data.
        home_abbr = TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_abbr = TEAM_NAME_MAP.get(away_team_full, away_team_full)

        # Get features for each team.
        home_feats_row = features_df[features_df['team'] == home_abbr]
        away_feats_row = features_df[features_df['team'] == away_abbr]

        # If a team's features aren't found, we can't make a prediction.
        if home_feats_row.empty or away_feats_row.empty:
            missing_team = home_abbr if home_feats_row.empty else away_abbr
            return jsonify({'error': f'No pre-computed features found for team: {missing_team}'}), 404

        home_feats = home_feats_row.iloc[0].to_dict()
        away_feats = away_feats_row.iloc[0].to_dict()
        
        # --- IMPORTANT: Feature Vector Construction ---
        # This dictionary structure MUST EXACTLY MATCH the features the model was trained on.
        # Any mismatch in keys here will cause a prediction error.
        final_features_dict = {
            'rolling_avg_hits_home': home_feats.get('rolling_avg_hits'),
            'rolling_avg_homers_home': home_feats.get('rolling_avg_homers'),
            'starter_rolling_era_home': home_feats.get('starter_rolling_era'),
            'starter_rolling_ks_home': home_feats.get('starter_rolling_ks'),
            'bullpen_rolling_era_home': home_feats.get('bullpen_rolling_era'),
            'rolling_avg_hits_away': away_feats.get('rolling_avg_hits'),
            'rolling_avg_homers_away': away_feats.get('rolling_avg_homers'),
            'starter_rolling_era_away': away_feats.get('starter_rolling_era'),
            'starter_rolling_ks_away': away_feats.get('starter_rolling_ks'),
            'bullpen_rolling_era_away': away_feats.get('bullpen_rolling_era'),
            'park_factor_avg_runs': home_feats.get('park_factor_avg_runs'),
            # Adding placeholders for weather if the model needs them.
            'temperature': 70, 
            'wind_speed': 5,
            'humidity': 50,
        }

        # Create a DataFrame from the dictionary for prediction.
        # Ensure the column order matches what the model expects.
        prediction_df = pd.DataFrame([final_features_dict])[MODEL_FEATURES]
        
        prediction = model.predict(prediction_df)
        
        return jsonify({'predicted_total_runs': float(prediction[0])})

    except KeyError as e:
        # This is the most common error: a feature is missing or misnamed.
        return jsonify({
            'error': f'Feature mismatch error. The model expected a feature that was not provided: {e}',
            'provided_features': list(final_features_dict.keys()),
            'expected_features': MODEL_FEATURES
        }), 400
    except Exception as e:
        # Catch any other unexpected errors.
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
