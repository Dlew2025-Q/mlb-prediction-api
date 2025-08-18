#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- LOAD MODEL AND FEATURES ON STARTUP ---
try:
    with open('mlb_total_runs_model.pkl', 'rb') as file:
        model = pickle.load(file)
    MODEL_FEATURES = model.get_booster().feature_names
    print("Model loaded successfully.")
except Exception as e:
    model = None
    MODEL_FEATURES = []
    print(f"CRITICAL ERROR: Could not load model. Error: {e}")

try:
    with open('latest_features.pkl', 'rb') as file:
        features_df = pickle.load(file)
    print("Pre-computed features loaded successfully.")
except Exception as e:
    features_df = None
    print(f"CRITICAL ERROR: Could not load pre-computed features. Error: {e}")

# --- CONFIGURATION & UTILITIES ---
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
TEAM_NAME_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB",
    "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "D-backs": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS",
    "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT",
    "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH",
    "ARZ": "ARI", "AZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK"
}
CITY_MAP = {
    "ARI": "Phoenix,AZ", "ATL": "Atlanta,GA", "BAL": "Baltimore,MD", "BOS": "Boston,MA", "CHC": "Chicago,IL",
    "CHW": "Chicago,IL", "CIN": "Cincinnati,OH", "CLE": "Cleveland,OH", "COL": "Denver,CO", "DET": "Detroit,MI",
    "HOU": "Houston,TX", "KC": "Kansas City,MO", "LAA": "Anaheim,CA", "LAD": "Los Angeles,CA", "MIA": "Miami,FL",
    "MIL": "Milwaukee,WI", "MIN": "Minneapolis,MN", "NYM": "Queens,NY", "NYY": "Bronx,NY", "OAK": "Oakland,CA",
    "PHI": "Philadelphia,PA", "PIT": "Pittsburgh,PA", "SD": "San Diego,CA", "SF": "San Francisco,CA",
    "SEA": "Seattle,WA", "STL": "St. Louis,MO", "TB": "St. Petersburg,FL", "TEX": "Arlington,TX",
    "TOR": "Toronto,ON", "WSH": "Washington,DC"
}

def get_weather_for_game(city):
    if not WEATHER_API_KEY or not city:
        return {'temperature': 70.0, 'wind_speed': 5.0, 'humidity': 50.0}
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/today?unitGroup=us&include=current&key={WEATHER_API_KEY}&contentType=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        current_conditions = weather_data.get('currentConditions', {})
        return { 'temperature': float(current_conditions.get('temp', 70.0)), 'wind_speed': float(current_conditions.get('windspeed', 5.0)), 'humidity': float(current_conditions.get('humidity', 50.0)) }
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch weather data for {city}. Error: {e}")
        return {'temperature': 70.0, 'wind_speed': 5.0, 'humidity': 50.0}

@app.route('/games')
def get_games():
    if not ODDS_API_KEY:
        return jsonify({'error': 'API key is not configured on the server.'}), 500
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?apiKey={ODDS_API_KEY}&regions=us&markets=totals"
    try:
        response = requests.get(url)
        response.raise_for_status()
        games = response.json()
        now_utc = datetime.now(timezone.utc)
        upcoming_games = [g for g in games if datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')) > now_utc]
        return jsonify(upcoming_games)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from The Odds API: {e}'}), 502

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or features_df is None:
        return jsonify({'error': 'Server is not ready; model or features not loaded.'}), 503

    try:
        game_data = request.get_json()
        home_team_full = game_data.get('home_team')
        away_team_full = game_data.get('away_team')

        if not all([home_team_full, away_team_full]):
            return jsonify({'error': 'Missing team data in request body'}), 400

        home_abbr = TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_abbr = TEAM_NAME_MAP.get(away_team_full, away_team_full)

        home_city = CITY_MAP.get(home_abbr)
        weather = get_weather_for_game(home_city)

        home_feats_row = features_df[features_df['team'] == home_abbr]
        away_feats_row = features_df[features_df['team'] == away_abbr]

        if home_feats_row.empty or away_feats_row.empty:
            missing_team = home_abbr if home_feats_row.empty else away_abbr
            return jsonify({'error': f'No pre-computed features found for team: {missing_team}'}), 404

        home_feats = home_feats_row.iloc[0].to_dict()
        away_feats = away_feats_row.iloc[0].to_dict()
        
        # --- FINAL FIX: Construct the feature vector to match the NEW retrained model ---
        final_features_dict = {
            'rolling_avg_adj_hits_home_perf': float(home_feats.get('rolling_avg_adj_hits_home_perf', 8.0)),
            'rolling_avg_adj_homers_home_perf': float(home_feats.get('rolling_avg_adj_homers_home_perf', 1.0)),
            'rolling_avg_adj_walks_home_perf': float(home_feats.get('rolling_avg_adj_walks_home_perf', 3.0)),
            'rolling_avg_adj_strikeouts_home_perf': float(home_feats.get('rolling_avg_adj_strikeouts_home_perf', 8.0)),
            
            'rolling_avg_adj_hits_away_perf': float(away_feats.get('rolling_avg_adj_hits_away_perf', 8.0)),
            'rolling_avg_adj_homers_away_perf': float(away_feats.get('rolling_avg_adj_homers_away_perf', 1.0)),
            'rolling_avg_adj_walks_away_perf': float(away_feats.get('rolling_avg_adj_walks_away_perf', 3.0)),
            'rolling_avg_adj_strikeouts_away_perf': float(away_feats.get('rolling_avg_adj_strikeouts_away_perf', 8.0)),
            
            # (Assuming the rest of the features from the last training are still expected)
            'starter_rolling_adj_era_home': float(home_feats.get('starter_rolling_adj_era', 4.5)),
            'starter_rolling_adj_era_away': float(away_feats.get('starter_rolling_adj_era', 4.5)),
            'park_factor': float(home_feats.get('park_factor', 9.0)),
            'bullpen_ip_last_3_days_home': float(home_feats.get('bullpen_ip_last_3_days', 0.0)),
            'bullpen_ip_last_3_days_away': float(away_feats.get('bullpen_ip_last_3_days', 0.0)),
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'humidity': weather['humidity'],
        }

        prediction_df = pd.DataFrame([final_features_dict])[MODEL_FEATURES]
        prediction = model.predict(prediction_df)
        
        return jsonify({'predicted_total_runs': float(prediction[0])})

    except KeyError as e:
        return jsonify({'error': f'Feature mismatch error: {e}', 'expected': MODEL_FEATURES}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
