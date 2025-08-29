#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
import pytz
import numpy as np
import warnings

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow requests from different domains
CORS(app)

# Suppress the UserWarning from pandas about merging on non-unique columns
warnings.filterwarnings('ignore', category=UserWarning, message='You are merging on an intermediate')

# --- LOAD MODELS AND FEATURES ---
def load_pickle(path):
    """
    Loads a pickled object from a given file path.
    Handles FileNotFoundError gracefully by printing a warning.
    """
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Warning: {path} not found. This sport will be unavailable.")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load {path}. Error: {e}")
        return None

# Attempt to load the trained models and feature dataframes for MLB and NFL
mlb_model = load_pickle('mlb_total_runs_model.pkl')
mlb_calibration_model = load_pickle('mlb_calibration_model.pkl')
mlb_features_df = load_pickle('latest_mlb_features.pkl')

nfl_model = load_pickle('nfl_total_points_model.pkl')
nfl_calibration_model = load_pickle('nfl_calibration_model.pkl')
nfl_features_df = load_pickle('latest_nfl_features.pkl')


# --- CONFIGURATION ---
# Get API keys from environment variables for security
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')

# Maps various MLB team name formats to a standard full name
MLB_TEAM_NAME_MAP = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals", "KC": "Kansas City Royals", "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees",
    "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SD": "San Diego Padres", "SFG": "San Francisco Giants", "SF": "San Francisco Giants", "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals", "WAS": "Washington Nationals",
    "Arizona Diamondbacks": "Arizona Diamondbacks", "Atlanta Braves": "Atlanta Braves", "Baltimore Orioles": "Baltimore Orioles", "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs", "Chicago White Sox": "Chicago White Sox", "Cincinnati Reds": "Cincinnati Reds", "Cleveland Guardians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies", "Detroit Tigers": "Detroit Tigers", "Houston Astros": "Houston Astros", "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels", "Los Angeles Dodgers": "Los Angeles Dodgers", "Miami Marlins": "Miami Marlins", "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins", "New York Mets": "New York Mets", "New York Yankees": "New York Yankees", "Oakland Athletics": "Oakland Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies", "Pittsburgh Pirates": "Pittsburgh Pirates", "San Diego Padres": "San Diego Padres", "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners", "St. Louis Cardinals": "St. Louis Cardinals", "Tampa Bay Rays": "Tampa Bay Rays", "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays", "Washington Nationals": "Washington Nationals",
    "Diamondbacks": "Arizona Diamondbacks", "D-backs": "Arizona Diamondbacks", "Braves": "Atlanta Braves", "Orioles": "Baltimore Orioles", "Red Sox": "Boston Red Sox", "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox", "Reds": "Cincinnati Reds", "Guardians": "Cleveland Guardians", "Indians": "Cleveland Guardians", "Rockies": "Colorado Rockies",
    "Angels": "Los Angeles Angels", "Dodgers": "Los Angeles Dodgers", "Marlins": "Miami Marlins", "Brewers": "Milwaukee Brewers", "Twins": "Minnesota Twins",
    "Mets": "New York Mets", "Yankees": "New York Yankees", "Athletics": "Oakland Athletics", "Phillies": "Philadelphia Phillies", "Pirates": "Pittsburgh Pirates",
    "Padres": "San Diego Padres", "Giants": "San Francisco Giants", "Mariners": "Seattle Mariners", "Cardinals": "St. Louis Cardinals", "Rays": "Tampa Bay Rays",
    "Rangers": "Texas Rangers", "Blue Jays": "Toronto Blue Jays", "Nationals": "Washington Nationals",
    "ARZ": "Arizona Diamondbacks", "AZ": "Arizona Diamondbacks", "CWS": "Chicago White Sox", "NY Mets": "New York Mets", "WSH Nationals": "Washington Nationals",
    "METS": "New York Mets", "YANKEES": "New York Yankees", "ATH": "Oakland Athletics", "WSH": "Washington Nationals"
}

# Maps various NFL team name formats to a standard full name
NFL_TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "OAK": "Las Vegas Raiders", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
    "Houston Oilers": "Houston Texans",
    "San Diego Chargers": "Los Angeles Chargers",
    "St. Louis Cardinals": "Arizona Cardinals",
    "Washington Redskins": "Washington Commanders",
    "Baltimore Colts": "Indianapolis Colts",
    "Boston Patriots": "New England Patriots",
    "Los Angeles Raiders": "Las Vegas Raiders",
    "Phoenix Cardinals": "Arizona Cardinals",
    "St. Louis Rams": "Los Angeles Rams",
    "Tennessee Oilers": "Tennessee Titans",
    "Washington Football Team": "Washington Commanders",
}

# Maps team names to city and state for weather lookup
CITY_MAP = {
    "Arizona Diamondbacks": "Phoenix,AZ", "Atlanta Braves": "Atlanta,GA", "Baltimore Orioles": "Baltimore,MD", "Boston Red Sox": "Boston,MA",
    "Chicago Cubs": "Chicago,IL", "Chicago White Sox": "Chicago,IL", "Cincinnati Reds": "Cincinnati,OH", "Cleveland Guardians": "Cleveland,OH",
    "Colorado Rockies": "Denver,CO", "Detroit Tigers": "Detroit,MI", "Houston Astros": "Houston,TX", "Kansas City Royals": "Kansas City,MO",
    "Los Angeles Angels": "Anaheim,CA", "Los Angeles Dodgers": "Los Angeles,CA", "Miami Marlins": "Miami,FL", "Milwaukee Brewers": "Milwaukee,WI",
    "Minnesota Twins": "Minneapolis,MN", "New York Mets": "Queens,NY", "New York Yankees": "Bronx,NY", "Oakland Athletics": "Oakland,CA",
    "Philadelphia Phillies": "Philadelphia,PA", "Pittsburgh Pirates": "Pittsburgh,PA", "San Diego Padres": "San Diego,CA",
    "San Francisco Giants": "San Francisco,CA", "Seattle Mariners": "Seattle,WA", "St. Louis Cardinals": "St. Louis,MO",
    "Tampa Bay Rays": "St. Petersburg,FL", "Texas Rangers": "Arlington,TX", "Toronto Blue Jays": "Toronto,ON", "Washington Nationals": "Washington,DC"
}

def get_weather_for_game(city):
    """
    Fetches current weather conditions for a given city from the Visual Crossing API.
    Returns default values if the API key is missing or the request fails.
    """
    if not WEATHER_API_KEY or not city:
        return {'temperature': 70.0, 'wind_speed': 5.0, 'humidity': 50.0}
    
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/today?unitGroup=us&include=current&key={WEATHER_API_KEY}&contentType=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        weather_data = response.json()
        current_conditions = weather_data.get('currentConditions', {})
        return {
            'temperature': float(current_conditions.get('temp', 70.0)),
            'wind_speed': float(current_conditions.get('windspeed', 5.0)),
            'humidity': float(current_conditions.get('humidity', 50.0))
        }
    except requests.exceptions.RequestException:
        # Return default values if the API call fails
        return {'temperature': 70.0, 'wind_speed': 5.0, 'humidity': 50.0}

@app.route('/games/<sport>')
def get_games(sport):
    """
    API endpoint to fetch a list of upcoming games for a specified sport.
    Fetches data from The Odds API and filters for future games.
    """
    sport_key_map = {"mlb": "baseball_mlb", "nfl": "americanfootball_nfl"}
    sport_key = sport_key_map.get(sport.lower())

    if not sport_key:
        return jsonify({'error': 'Invalid sport specified. Must be "mlb" or "nfl".'}), 400
    if not ODDS_API_KEY:
        return jsonify({'error': 'Odds API key not configured.'}), 500
    
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds?apiKey={ODDS_API_KEY}&regions=us&markets=totals,spreads"

    try:
        response = requests.get(url)
        response.raise_for_status()
        games = response.json()
        
        # Filter for upcoming games
        now_utc = datetime.now(timezone.utc)
        upcoming_games = [g for g in games if datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')) > now_utc]
        
        return jsonify(upcoming_games)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from The Odds API: {e}'}), 502

@app.route('/predict/<sport>', methods=['POST'])
def predict(sport):
    """
    API endpoint to make a total score prediction for a game based on a POST request.
    It takes home and away team names and returns a predicted total and confidence score.
    """
    sport = sport.lower()
    game_data = request.get_json()
    home_team_full = game_data.get('home_team')
    away_team_full = game_data.get('away_team')
    commence_time_str = game_data.get('commence_time')

    if not all([home_team_full, away_team_full, commence_time_str]):
        return jsonify({'error': 'Missing team or commence time data in request body.'}), 400

    commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
    
    final_features = {}
    if sport == "mlb":
        if mlb_model is None or mlb_calibration_model is None or mlb_features_df is None:
            return jsonify({'error': 'MLB model or features not loaded.'}), 503
        
        home_team_standard = MLB_TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_team_standard = MLB_TEAM_NAME_MAP.get(away_team_full, away_team_full)

        # FIX: The logic was not correctly distinguishing between home and away stats.
        # This new logic gets the correct stats for each team's role in the upcoming game.

        # --- Get LATEST OVERALL game to calculate true rest days ---
        home_team_last_game_df = mlb_features_df[(mlb_features_df['home_team'] == home_team_standard) | (mlb_features_df['away_team'] == home_team_standard)]
        away_team_last_game_df = mlb_features_df[(mlb_features_df['home_team'] == away_team_standard) | (mlb_features_df['away_team'] == away_team_standard)]

        # --- Get LATEST HOME/AWAY game for performance stats ---
        home_team_latest_home_game_df = mlb_features_df[mlb_features_df['home_team'] == home_team_standard]
        away_team_latest_away_game_df = mlb_features_df[mlb_features_df['away_team'] == away_team_standard]

        if home_team_last_game_df.empty or away_team_last_game_df.empty or home_team_latest_home_game_df.empty or away_team_latest_away_game_df.empty:
            return jsonify({'error': f"Could not find sufficient historical game data for {home_team_full} or {away_team_full}."}), 404

        # Get the last game record for each context
        home_team_last_game = home_team_last_game_df.iloc[-1]
        away_team_last_game = away_team_last_game_df.iloc[-1]
        home_feats = home_team_latest_home_game_df.iloc[-1].to_dict()
        away_feats = away_team_latest_away_game_df.iloc[-1].to_dict()

        # Calculate days of rest until the upcoming game
        home_rest = (commence_time - pd.to_datetime(home_team_last_game['commence_time'])).days
        away_rest = (commence_time - pd.to_datetime(away_team_last_game['commence_time'])).days

        home_city = CITY_MAP.get(home_team_standard)
        weather = get_weather_for_game(home_city)
        
        final_features = {
            'in_series_hits_lag_home': home_feats.get('in_series_hits_lag_home', 8.0),
            'in_series_homers_lag_home': home_feats.get('in_series_homers_lag_home', 1.0),
            'in_series_walks_lag_home': home_feats.get('in_series_walks_lag_home', 3.0),
            'in_series_strikeouts_lag_home': home_feats.get('in_series_strikeouts_lag_home', 8.0),
            'starter_rolling_adj_era_home': home_feats.get('starter_rolling_adj_era_home', 4.5),
            'park_factor': home_feats.get('park_factor', 9.0),
            'bullpen_ip_last_3_days_home': home_feats.get('bullpen_ip_last_3_days_home', 0.0),
            
            'in_series_hits_lag_away': away_feats.get('in_series_hits_lag_away', 8.0),
            'in_series_homers_lag_away': away_feats.get('in_series_homers_lag_away', 1.0),
            'in_series_walks_lag_away': away_feats.get('in_series_walks_lag_away', 3.0),
            'in_series_strikeouts_lag_away': away_feats.get('in_series_strikeouts_lag_away', 8.0),
            'starter_rolling_adj_era_away': away_feats.get('starter_rolling_adj_era_away', 4.5),
            'bullpen_ip_last_3_days_away': away_feats.get('bullpen_ip_last_3_days_away', 0.0),
            
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'humidity': weather['humidity'],
            'home_days_rest': home_rest,
            'away_days_rest': away_rest,
            'game_of_season': home_feats.get('game_of_season', 1.0), # Game of season can be taken from home team's perspective
            'travel_factor': away_feats.get('travel_factor', 0.0) # Travel factor applies to the away team
        }
    elif sport == "nfl":
        if nfl_model is None or nfl_calibration_model is None or nfl_features_df is None:
            return jsonify({'error': 'NFL model or features not loaded.'}), 503

        home_team_standard = NFL_TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_team_standard = NFL_TEAM_NAME_MAP.get(away_team_full, away_team_full)

        # --- Get LATEST OVERALL game for rest days ---
        home_team_last_game_df = nfl_features_df[(nfl_features_df['home_team'] == home_team_standard) | (nfl_features_df['away_team'] == home_team_standard)]
        away_team_last_game_df = nfl_features_df[(nfl_features_df['home_team'] == away_team_standard) | (nfl_features_df['away_team'] == away_team_standard)]

        # --- Get LATEST HOME/AWAY game for performance stats ---
        home_team_latest_home_game_df = nfl_features_df[nfl_features_df['home_team'] == home_team_standard]
        away_team_latest_away_game_df = nfl_features_df[nfl_features_df['away_team'] == away_team_standard]

        if home_team_last_game_df.empty or away_team_last_game_df.empty or home_team_latest_home_game_df.empty or away_team_latest_away_game_df.empty:
            return jsonify({'error': f'Could not find sufficient historical game data for {home_team_full} or {away_team_full}.'}), 404

        home_team_last_game = home_team_last_game_df.iloc[-1]
        away_team_last_game = away_team_last_game_df.iloc[-1]
        home_feats = home_team_latest_home_game_df.iloc[-1].to_dict()
        away_feats = away_team_latest_away_game_df.iloc[-1].to_dict()

        home_rest = (commence_time - pd.to_datetime(home_team_last_game['commence_time'])).days
        away_rest = (commence_time - pd.to_datetime(away_team_last_game['commence_time'])).days

        home_city = CITY_MAP.get(home_team_standard)
        weather = get_weather_for_game(home_city)

        final_features = {
            'rolling_avg_adj_pts_scored_home': home_feats.get('rolling_avg_adj_pts_scored_home', 21.0),
            'rolling_avg_adj_pts_allowed_home': home_feats.get('rolling_avg_adj_pts_allowed_home', 21.0),
            'rolling_avg_adj_pts_scored_away': away_feats.get('rolling_avg_adj_pts_scored_away', 21.0),
            'rolling_avg_adj_pts_allowed_away': away_feats.get('rolling_avg_adj_pts_allowed_away', 21.0),
            'home_days_rest': home_rest,
            'away_days_rest': away_rest,
            'game_of_season': home_feats.get('game_of_season', 1.0),
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'humidity': weather['humidity']
        }
    else:
        return jsonify({'error': 'Invalid sport specified. Must be "mlb" or "nfl".'}), 400

    try:
        model = mlb_model if sport == 'mlb' else nfl_model
        calibration_model = mlb_calibration_model if sport == 'mlb' else nfl_calibration_model

        feature_order = model.get_booster().feature_names
        
        prediction_df = pd.DataFrame([final_features], columns=feature_order)

        raw_prediction = model.predict(prediction_df)[0]
        
        confidence_df = pd.DataFrame([{'raw_prediction': raw_prediction}])
        confidence_score = calibration_model.predict_proba(confidence_df.values.reshape(-1, 1))[0][1]

        return jsonify({
            'predicted_total_runs': float(raw_prediction),
            'confidence': float(confidence_score)
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

