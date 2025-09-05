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
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')

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

# Pre-process dataframes at startup for efficiency
if mlb_features_df is not None:
    mlb_features_df['commence_time'] = pd.to_datetime(mlb_features_df['commence_time'], utc=True)
if nfl_features_df is not None:
    nfl_features_df['commence_time'] = pd.to_datetime(nfl_features_df['commence_time'], utc=True)


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

# Add NFL team name map
NFL_TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "OAK": "Las Vegas Raiders", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
    # Add full names for consistency
    "Arizona Cardinals": "Arizona Cardinals", "Atlanta Falcons": "Atlanta Falcons", "Baltimore Ravens": "Baltimore Ravens", "Buffalo Bills": "Buffalo Bills", "Carolina Panthers": "Carolina Panthers", "Chicago Bears": "Chicago Bears",
    "Cincinnati Bengals": "Cincinnati Bengals", "Cleveland Browns": "Cleveland Browns", "Dallas Cowboys": "Dallas Cowboys", "Denver Broncos": "Denver Broncos", "Detroit Lions": "Detroit Lions", "Green Bay Packers": "Green Bay Packers",
    "Houston Texans": "Houston Texans", "Indianapolis Colts": "Indianapolis Colts", "Jacksonville Jaguars": "Jacksonville Jaguars", "Kansas City Chiefs": "Kansas City Chiefs", "Las Vegas Raiders": "Las Vegas Raiders", "Los Angeles Chargers": "Los Angeles Chargers",
    "Los Angeles Rams": "Los Angeles Rams", "Miami Dolphins": "Miami Dolphins", "Minnesota Vikings": "Minnesota Vikings", "New England Patriots": "New England Patriots", "New Orleans Saints": "New Orleans Saints", "New York Giants": "New York Giants",
    "New York Jets": "New York Jets", "Philadelphia Eagles": "Philadelphia Eagles", "Pittsburgh Steelers": "Pittsburgh Steelers", "San Francisco 49ers": "San Francisco 49ers", "Seattle Seahawks": "Seattle Seahawks",
    "Tampa Bay Buccaneers": "Tampa Bay Buccaneers", "Tennessee Titans": "Tennessee Titans", "Washington Commanders": "Washington Commanders",
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
    "Tampa Bay Rays": "St. Petersburg,FL", "Texas Rangers": "Arlington,TX", "Toronto Blue Jays": "Toronto,ON", "Washington Nationals": "Washington,DC",
    # NFL Cities
    "Arizona Cardinals": "Glendale,AZ", "Atlanta Falcons": "Atlanta,GA", "Baltimore Ravens": "Baltimore,MD", "Buffalo Bills": "Orchard Park,NY", "Carolina Panthers": "Charlotte,NC",
    "Chicago Bears": "Chicago,IL", "Cincinnati Bengals": "Cincinnati,OH", "Cleveland Browns": "Cleveland,OH", "Dallas Cowboys": "Arlington,TX", "Denver Broncos": "Denver,CO",
    "Detroit Lions": "Detroit,MI", "Green Bay Packers": "Green Bay,WI", "Houston Texans": "Houston,TX", "Indianapolis Colts": "Indianapolis,IN", "Jacksonville Jaguars": "Jacksonville,FL",
    "Kansas City Chiefs": "Kansas City,MO", "Las Vegas Raiders": "Las Vegas,NV", "Los Angeles Chargers": "Inglewood,CA", "Los Angeles Rams": "Inglewood,CA", "Miami Dolphins": "Miami Gardens,FL",
    "Minnesota Vikings": "Minneapolis,MN", "New England Patriots": "Foxborough,MA", "New Orleans Saints": "New Orleans,LA", "New York Giants": "East Rutherford,NJ", "New York Jets": "East Rutherford,NJ",
    "Philadelphia Eagles": "Philadelphia,PA", "Pittsburgh Steelers": "Pittsburgh,PA", "San Francisco 49ers": "Santa Clara,CA", "Seattle Seahawks": "Seattle,WA",
    "Tampa Bay Buccaneers": "Tampa,FL", "Tennessee Titans": "Nashville,TN", "Washington Commanders": "Landover,MD"
}

# Maps team names to city timezones for calculating travel factor
CITY_TIMEZONE_MAP = {
    "Arizona Diamondbacks": "America/Phoenix", "Atlanta Braves": "America/New_York", "Baltimore Orioles": "America/New_York", "Boston Red Sox": "America/New_York",
    "Chicago Cubs": "America/Chicago", "Chicago White Sox": "America/Chicago", "Cincinnati Reds": "America/New_York", "Cleveland Guardians": "America/New_York",
    "Colorado Rockies": "America/Denver", "Detroit Tigers": "America/New_York", "Houston Astros": "America/Chicago", "Kansas City Royals": "America/Chicago",
    "Los Angeles Angels": "America/Los_Angeles", "Los Angeles Dodgers": "America/Los_Angeles", "Miami Marlins": "America/New_York", "Milwaukee Brewers": "America/Chicago",
    "Minnesota Twins": "America/Chicago", "New York Mets": "America/New_York", "New York Yankees": "America/New_York", "Oakland Athletics": "America/Los_Angeles",
    "Philadelphia Phillies": "America/New_York", "Pittsburgh Pirates": "America/New_York", "San Diego Padres": "America/Los_Angeles", "San Francisco Giants": "America/Los_Angeles",
    "Seattle Mariners": "America/Los_Angeles", "St. Louis Cardinals": "America/Chicago", "Tampa Bay Rays": "America/New_York", "Texas Rangers": "America/Chicago",
    "Toronto Blue Jays": "America/New_York", "Washington Nationals": "America/New_York"
}

PARK_FACTOR_MAP = {
    "Cincinnati Reds": 1.15, "Colorado Rockies": 1.12, "Kansas City Royals": 1.08, "Boston Red Sox": 1.07,
    "Los Angeles Angels": 1.05, "Chicago White Sox": 1.04, "Atlanta Braves": 1.03, "Texas Rangers": 1.02,
    "Philadelphia Phillies": 1.01, "Chicago Cubs": 1.01, "Houston Astros": 1.00, "Los Angeles Dodgers": 1.00,
    "Toronto Blue Jays": 1.00, "Detroit Tigers": 0.99, "Washington Nationals": 0.98, "Minnesota Twins": 0.98,
    "Arizona Diamondbacks": 0.97, "New York Yankees": 0.97, "Milwaukee Brewers": 0.96, "Baltimore Orioles": 0.96,
    "New York Mets": 0.95, "St. Louis Cardinals": 0.94, "Tampa Bay Rays": 0.93, "Pittsburgh Pirates": 0.92,
    "Miami Marlins": 0.91, "Cleveland Guardians": 0.90, "San Diego Padres": 0.89, "Oakland Athletics": 0.88,
    "Seattle Mariners": 0.87, "San Francisco Giants": 0.86
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
        
        now_utc = datetime.now(timezone.utc)
        upcoming_games_data = [g for g in games if datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')) > now_utc]
        
        return jsonify(upcoming_games_data)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from The Odds API: {e}'}), 502

def get_feature(feat_dict, key, default):
    """Safely gets a feature, returning default if key is missing or value is NaN."""
    val = feat_dict.get(key)
    if val is None or pd.isna(val):
        return default
    return val

@app.route('/predict/<sport>', methods=['POST'])
def predict(sport):
    """
    API endpoint to make a total score prediction for a game based on a POST request.
    It takes home and away team names, and the market line, then returns a predicted
    total, confidence score, edge, and a suggestion (Over/Under/Hold).
    """
    sport = sport.lower()
    game_data = request.get_json()
    home_team_full = game_data.get('home_team')
    away_team_full = game_data.get('away_team')
    commence_time_str = game_data.get('commence_time')
    market_line = game_data.get('market_line') 
    
    if not all([home_team_full, away_team_full, commence_time_str]):
        return jsonify({'error': 'Missing team or commence time data in request body.'}), 400
    
    commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))

    final_features = {}
    if sport == "mlb":
        if mlb_model is None or mlb_calibration_model is None or mlb_features_df is None:
            return jsonify({'error': 'MLB model or features not loaded.'}), 503
        
        home_team_standard = MLB_TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_team_standard = MLB_TEAM_NAME_MAP.get(away_team_full, away_team_full)

        sorted_mlb_features = mlb_features_df.sort_values('commence_time')

        last_home_game = sorted_mlb_features[sorted_mlb_features['home_team'] == home_team_standard]
        last_away_game = sorted_mlb_features[sorted_mlb_features['away_team'] == away_team_standard]
        
        if last_home_game.empty or last_away_game.empty:
            return jsonify({'error': f"No historical MLB features found for {home_team_full} (home) or {away_team_full} (away). Please run pre-computation script."}), 404

        home_feats = last_home_game.iloc[-1].to_dict()
        away_feats = last_away_game.iloc[-1].to_dict()
        
        last_home_game_time = pd.to_datetime(home_feats['commence_time'], utc=True)
        last_away_game_time = pd.to_datetime(away_feats['commence_time'], utc=True)
        home_days_rest = (commence_time - last_home_game_time).days
        away_days_rest = (commence_time - last_away_game_time).days

        current_year = commence_time.year
        home_games_this_season = sorted_mlb_features[
            (sorted_mlb_features['commence_time'].dt.year == current_year) & 
            ((sorted_mlb_features['home_team'] == home_team_standard) | (sorted_mlb_features['away_team'] == home_team_standard))
        ]
        game_of_season = len(home_games_this_season) + 1
        
        def get_tz_offset(team_name):
            tz_name = CITY_TIMEZONE_MAP.get(team_name)
            if not tz_name: return 0
            try: return datetime(2023, 7, 1, tzinfo=pytz.timezone(tz_name)).utcoffset().total_seconds() / 3600
            except pytz.UnknownTimeZoneError: return 0

        away_team_previous_games = sorted_mlb_features[
            (sorted_mlb_features['home_team'] == away_team_standard) | (sorted_mlb_features['away_team'] == away_team_standard)
        ]
        if not away_team_previous_games.empty:
            last_game = away_team_previous_games.iloc[-1]
            last_game_location_team = last_game['home_team']
            previous_city_tz = get_tz_offset(last_game_location_team)
            current_city_tz = get_tz_offset(home_team_standard)
            travel_factor = abs(current_city_tz - previous_city_tz)
        else:
            travel_factor = 0.0
        
        park_factor = PARK_FACTOR_MAP.get(home_team_standard, 1.0)

        # FIX: Align feature set with the latest precompute script
        final_features = {
            'rolling_avg_adj_hits_home': get_feature(home_feats, 'rolling_avg_adj_hits_home', 8.0),
            'rolling_avg_adj_homers_home': get_feature(home_feats, 'rolling_avg_adj_homers_home', 1.0),
            'rolling_avg_adj_walks_home': get_feature(home_feats, 'rolling_avg_adj_walks_home', 3.0),
            'rolling_avg_adj_strikeouts_home': get_feature(home_feats, 'rolling_avg_adj_strikeouts_home', 8.0),
            'starter_rolling_adj_era_home': get_feature(home_feats, 'starter_rolling_adj_era_home', 4.5),
            'starter_rolling_whip_home': get_feature(home_feats, 'starter_rolling_whip_home', 1.3),
            'starter_rolling_k_per_9_home': get_feature(home_feats, 'starter_rolling_k_per_9_home', 8.5),
            'rolling_bullpen_era_home': get_feature(home_feats, 'rolling_bullpen_era_home', 4.5),
            'park_factor': park_factor,
            'bullpen_ip_last_3_days_home': get_feature(home_feats, 'bullpen_ip_last_3_days_home', 0.0),
            'rolling_avg_adj_hits_away': get_feature(away_feats, 'rolling_avg_adj_hits_away', 8.0),
            'rolling_avg_adj_homers_away': get_feature(away_feats, 'rolling_avg_adj_homers_away', 1.0),
            'rolling_avg_adj_walks_away': get_feature(away_feats, 'rolling_avg_adj_walks_away', 3.0),
            'rolling_avg_adj_strikeouts_away': get_feature(away_feats, 'rolling_avg_adj_strikeouts_away', 8.0),
            'starter_rolling_adj_era_away': get_feature(away_feats, 'starter_rolling_adj_era_away', 4.5),
            'starter_rolling_whip_away': get_feature(away_feats, 'starter_rolling_whip_away', 1.3),
            'starter_rolling_k_per_9_away': get_feature(away_feats, 'starter_rolling_k_per_9_away', 8.5),
            'rolling_bullpen_era_away': get_feature(away_feats, 'rolling_bullpen_era_away', 4.5),
            'bullpen_ip_last_3_days_away': get_feature(away_feats, 'bullpen_ip_last_3_days_away', 0.0),
            'home_days_rest': home_days_rest,
            'away_days_rest': away_days_rest,
            'game_of_season': game_of_season,
            'travel_factor': travel_factor,
            'starter_era_diff': get_feature(away_feats, 'starter_rolling_adj_era_away', 4.5) - get_feature(home_feats, 'starter_rolling_adj_era_home', 4.5),
            'bullpen_era_diff': get_feature(away_feats, 'rolling_bullpen_era_away', 4.5) - get_feature(home_feats, 'rolling_bullpen_era_home', 4.5),
            'home_offense_vs_away_defense': get_feature(away_feats, 'pitching_rank', 15.5) - get_feature(home_feats, 'hitting_rank', 15.5),
            'away_offense_vs_home_defense': get_feature(home_feats, 'pitching_rank', 15.5) - get_feature(away_feats, 'hitting_rank', 15.5)
        }

    elif sport == "nfl":
        if nfl_model is None or nfl_calibration_model is None or nfl_features_df is None:
            return jsonify({'error': 'NFL model or features not loaded.'}), 503

        home_team_standard = NFL_TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_team_standard = NFL_TEAM_NAME_MAP.get(away_team_full, away_team_full)

        sorted_nfl_features = nfl_features_df.sort_values('commence_time')

        last_home_game = sorted_nfl_features[sorted_nfl_features['home_team'] == home_team_standard]
        last_away_game = sorted_nfl_features[sorted_nfl_features['away_team'] == away_team_standard]

        if last_home_game.empty or last_away_game.empty:
            return jsonify({'error': f'No NFL features found for {home_team_full} or {away_team_full} in the loaded features file.'}), 404
        
        home_feats = last_home_game.iloc[-1].to_dict()
        away_feats = last_away_game.iloc[-1].to_dict()

        home_city = CITY_MAP.get(home_team_standard)
        weather = get_weather_for_game(home_city)

        final_features = {
            'rolling_avg_adj_pts_scored_home': get_feature(home_feats, 'rolling_avg_adj_pts_scored_home', 21.0),
            'rolling_avg_adj_pts_allowed_home': get_feature(home_feats, 'rolling_avg_adj_pts_allowed_home', 21.0),
            'rolling_avg_adj_pts_scored_away': get_feature(away_feats, 'rolling_avg_adj_pts_scored_away', 21.0),
            'rolling_avg_adj_pts_allowed_away': get_feature(away_feats, 'rolling_avg_adj_pts_allowed_away', 21.0),
            'home_days_rest': (commence_time - pd.to_datetime(home_feats['commence_time'], utc=True)).days,
            'away_days_rest': (commence_time - pd.to_datetime(away_feats['commence_time'], utc=True)).days,
            'game_of_season': home_feats.get('game_of_season', 1.0) + 1,
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

        suggestion = "Hold"
        edge = 0
        if market_line is not None:
            try:
                 market_line_float = float(market_line)
                 edge = raw_prediction - market_line_float
                 
                 # FIX: Implement the "Alpha Strategy" thresholds from the analysis
                 min_confidence = 0.35
                 min_edge = 1.5
                 
                 if edge > min_edge and confidence_score > min_confidence:
                     suggestion = "Over"
                 elif edge < -min_edge and confidence_score > min_confidence:
                     suggestion = "Under"
            except (ValueError, TypeError):
                pass

        return jsonify({
            'predicted_total_runs': float(raw_prediction),
            'confidence': float(confidence_score),
            'suggestion': suggestion,
            'edge': float(edge)
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

