#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)

# --- LOAD MODELS AND FEATURES ---
def load_pickle(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Warning: {path} not found. This sport will be unavailable.")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load {path}. Error: {e}")
        return None

# Load all models and feature files for both sports
mlb_model = load_pickle('mlb_total_runs_model.pkl')
nfl_model = load_pickle('nfl_total_points_model.pkl')
mlb_features_df = load_pickle('latest_features.pkl') 
nfl_features_df = load_pickle('latest_nfl_features.pkl')

# --- CONFIGURATION ---
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')

MLB_TEAM_NAME_MAP = { "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "D-backs": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH", "ARZ": "ARI", "AZ": "ARI", "CWS": "CHW", "NY Mets": "NYM", "WSH Nationals": "WSH", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK" }
CITY_MAP = { "ARI": "Phoenix,AZ", "ATL": "Atlanta,GA", "BAL": "Baltimore,MD", "BOS": "Boston,MA", "CHC": "Chicago,IL", "CHW": "Chicago,IL", "CIN": "Cincinnati,OH", "CLE": "Cleveland,OH", "COL": "Denver,CO", "DET": "Detroit,MI", "HOU": "Houston,TX", "KC": "Kansas City,MO", "LAA": "Anaheim,CA", "LAD": "Los Angeles,CA", "MIA": "Miami,FL", "MIL": "Milwaukee,WI", "MIN": "Minneapolis,MN", "NYM": "Queens,NY", "NYY": "Bronx,NY", "OAK": "Oakland,CA", "PHI": "Philadelphia,PA", "PIT": "Pittsburgh,PA", "SD": "San Diego,CA", "SF": "San Francisco,CA", "SEA": "Seattle,WA", "STL": "St. Louis,MO", "TB": "St. Petersburg,FL", "TEX": "Arlington,TX", "TOR": "Toronto,ON", "WSH": "Washington,DC" }

def get_weather_for_game(city):
    if not WEATHER_API_KEY or not city: return {'temperature': 70.0, 'wind_speed': 5.0, 'humidity': 50.0}
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/today?unitGroup=us&include=current&key={WEATHER_API_KEY}&contentType=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        current_conditions = weather_data.get('currentConditions', {})
        return { 'temperature': float(current_conditions.get('temp', 70.0)), 'wind_speed': float(current_conditions.get('windspeed', 5.0)), 'humidity': float(current_conditions.get('humidity', 50.0)) }
    except requests.exceptions.RequestException: return {'temperature': 70.0, 'wind_speed': 5.0, 'humidity': 50.0}

@app.route('/games/<sport>')
def get_games(sport):
    sport_key_map = {"mlb": "baseball_mlb", "nfl": "americanfootball_nfl"}
    sport_key = sport_key_map.get(sport.lower())
    if not sport_key: return jsonify({'error': 'Invalid sport specified'}), 400
    if not ODDS_API_KEY: return jsonify({'error': 'API key not configured'}), 500
    
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds?apiKey={ODDS_API_KEY}&regions=us&markets=totals,spreads"
    try:
        response = requests.get(url)
        response.raise_for_status()
        games = response.json()
        now_utc = datetime.now(timezone.utc)
        upcoming_games = [g for g in games if datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')) > now_utc]
        return jsonify(upcoming_games)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from The Odds API: {e}'}), 502

@app.route('/predict/<sport>', methods=['POST'])
def predict(sport):
    sport = sport.lower()
    game_data = request.get_json()
    home_team_full = game_data.get('home_team')
    away_team_full = game_data.get('away_team')

    if not all([home_team_full, away_team_full]):
        return jsonify({'error': 'Missing team data in request body'}), 400

    if sport == "mlb":
        if mlb_model is None or mlb_features_df is None:
            return jsonify({'error': 'MLB model or features not loaded.'}), 503
        
        home_abbr = MLB_TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_abbr = MLB_TEAM_NAME_MAP.get(away_team_full, away_team_full)
        home_city = CITY_MAP.get(home_abbr)
        weather = get_weather_for_game(home_city)

        home_feats_row = mlb_features_df[mlb_features_df['team'] == home_abbr]
        away_feats_row = mlb_features_df[mlb_features_df['team'] == away_abbr]

        if home_feats_row.empty or away_feats_row.empty:
            return jsonify({'error': f'No MLB features found for a team'}), 404

        home_feats = home_feats_row.iloc[0].to_dict()
        away_feats = away_feats_row.iloc[0].to_dict()
        
        final_features = {
            'rolling_avg_adj_hits_home_perf': float(home_feats.get('rolling_avg_adj_hits_home_perf', 8.0)),
            'rolling_avg_adj_homers_home_perf': float(home_feats.get('rolling_avg_adj_homers_home_perf', 1.0)),
            'rolling_avg_adj_walks_home_perf': float(home_feats.get('rolling_avg_adj_walks_home_perf', 3.0)),
            'rolling_avg_adj_strikeouts_home_perf': float(home_feats.get('rolling_avg_adj_strikeouts_home_perf', 8.0)),
            'rolling_avg_adj_hits_away_perf': float(away_feats.get('rolling_avg_adj_hits_away_perf', 8.0)),
            'rolling_avg_adj_homers_away_perf': float(away_feats.get('rolling_avg_adj_homers_away_perf', 1.0)),
            'rolling_avg_adj_walks_away_perf': float(away_feats.get('rolling_avg_adj_walks_away_perf', 3.0)),
            'rolling_avg_adj_strikeouts_away_perf': float(away_feats.get('rolling_avg_adj_strikeouts_away_perf', 8.0)),
            'starter_rolling_adj_era_home': float(home_feats.get('starter_rolling_adj_era', 4.5)),
            'starter_rolling_adj_era_away': float(away_feats.get('starter_rolling_adj_era', 4.5)),
            'park_factor': float(home_feats.get('park_factor', 9.0)),
            'bullpen_ip_last_3_days_home': float(home_feats.get('bullpen_ip_last_3_days', 0.0)),
            'bullpen_ip_last_3_days_away': float(away_feats.get('bullpen_ip_last_3_days', 0.0)),
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'humidity': weather['humidity'],
        }
        
    elif sport == "nfl":
        if nfl_model is None or nfl_features_df is None:
            return jsonify({'error': 'NFL model or features not loaded.'}), 503

        home_feats_row = nfl_features_df[nfl_features_df['team'] == home_team_full]
        away_feats_row = nfl_features_df[nfl_features_df['team'] == away_team_full]
        
        if home_feats_row.empty or away_feats_row.empty:
            return jsonify({'error': f'No NFL features found for a team'}), 404
        
        home_feats = home_feats_row.iloc[0].to_dict()
        away_feats = away_feats_row.iloc[0].to_dict()

        final_features = {
            'rolling_avg_adj_pts_scored_home': float(home_feats.get('rolling_avg_adj_pts_scored_home', 21.0)),
            'rolling_avg_adj_pts_allowed_home': float(home_feats.get('rolling_avg_adj_pts_allowed_home', 21.0)),
            'rolling_avg_adj_pts_scored_away': float(away_feats.get('rolling_avg_adj_pts_scored_away', 21.0)),
            'rolling_avg_adj_pts_allowed_away': float(away_feats.get('rolling_avg_adj_pts_allowed_away', 21.0)),
        }
        
    else:
        return jsonify({'error': 'Invalid sport specified'}), 400

    try:
        model = mlb_model if sport == 'mlb' else nfl_model
        prediction_df = pd.DataFrame([final_features])[model.get_booster().feature_names]
        prediction = model.predict(prediction_df)
        return jsonify({'predicted_total_runs': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
