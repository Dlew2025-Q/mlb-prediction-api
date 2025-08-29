#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
import pytz

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow requests from different domains
CORS(app)

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
nfl_model = load_pickle('nfl_total_points_model.pkl')
mlb_features_df = load_pickle('latest_features.pkl')
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
    "METS": "New York Mets", "YANKEES": "New York Yankees", "ATH": "Oakland Athletics"
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
    It takes home and away team names and returns a predicted total.
    """
    sport = sport.lower()
    game_data = request.get_json()
    home_team_full = game_data.get('home_team')
    away_team_full = game_data.get('away_team')
    commence_time_str = game_data.get('commence_time')

    if not all([home_team_full, away_team_full, commence_time_str]):
        return jsonify({'error': 'Missing team or commence time data in request body.'}), 400

    commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))

    if sport == "mlb":
        if mlb_model is None or mlb_features_df is None:
            return jsonify({'error': 'MLB model or features not loaded.'}), 503
        
        # Standardize team names to match the feature files
        home_team_standard = MLB_TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_team_standard = MLB_TEAM_NAME_MAP.get(away_team_full, away_team_full)

        home_city = CITY_MAP.get(home_team_standard)
        weather = get_weather_for_game(home_city)

        home_feats_row = mlb_features_df[mlb_features_df['team'] == home_team_standard]
        away_feats_row = mlb_features_df[mlb_features_df['team'] == away_team_standard]

        if home_feats_row.empty or away_feats_row.empty:
            return jsonify({'error': f'No MLB features found for {home_team_full} or {away_team_full}. Check team names.'}), 404

        home_feats = home_feats_row.iloc[0].to_dict()
        away_feats = away_feats_row.iloc[0].to_dict()
        
        # Calculate new fatigue features
        # The column names in the pkl file are based on how they were created.
        # Home team's features get a '_x' suffix and away team's get a '_y'
        home_days_rest = float(home_feats.get('home_days_rest_x', 3.0))
        away_days_rest = float(away_feats.get('away_days_rest_y', 3.0))
        game_of_season = float(home_feats.get('game_of_season_x', 1.0))
        
        home_tz_name = CITY_TIMEZONE_MAP.get(home_team_standard, 'UTC')
        away_tz_name = CITY_TIMEZONE_MAP.get(away_team_standard, 'UTC')
        
        try:
            home_tz = pytz.timezone(home_tz_name)
            away_tz = pytz.timezone(away_tz_name)
            # Calculate the time difference in hours and use it to determine travel factor
            # Eastward travel (positive difference) is more fatiguing
            travel_factor = (home_tz.utcoffset(commence_time) - away_tz.utcoffset(commence_time)).total_seconds() / 3600
        except pytz.UnknownTimeZoneError:
            travel_factor = 0.0

        final_features = {
            'rolling_avg_adj_hits_home_perf': float(home_feats.get('rolling_avg_adj_hits_home_perf', 8.0)),
            'rolling_avg_adj_homers_home_perf': float(home_feats.get('rolling_avg_adj_homers_home_perf', 1.0)),
            'rolling_avg_adj_walks_home_perf': float(home_feats.get('rolling_avg_adj_walks_home_perf', 3.0)),
            'rolling_avg_adj_strikeouts_home_perf': float(home_feats.get('rolling_avg_adj_strikeouts_home_perf', 8.0)),
            'rolling_avg_adj_hits_away_perf': float(away_feats.get('rolling_avg_adj_hits_away_perf', 8.0)),
            'rolling_avg_adj_homers_away_perf': float(away_feats.get('rolling_avg_adj_homers_away_perf', 1.0)),
            'rolling_avg_adj_walks_away_perf': float(away_feats.get('rolling_avg_adj_walks_away_perf', 3.0)),
            'rolling_avg_adj_strikeouts_away_perf': float(away_feats.get('rolling_avg_adj_strikeouts_away_perf', 8.0)),
            'starter_rolling_adj_era_home': float(home_feats.get('starter_rolling_adj_era_home', 4.5)),
            'starter_rolling_adj_era_away': float(away_feats.get('starter_rolling_adj_era_away', 4.5)),
            'park_factor': float(home_feats.get('park_factor_x', 9.0)),
            'bullpen_ip_last_3_days_home': float(home_feats.get('bullpen_ip_last_3_days_x', 0.0)),
            'bullpen_ip_last_3_days_away': float(away_feats.get('bullpen_ip_last_3_days_y', 0.0)),
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'humidity': weather['humidity'],
            'home_days_rest': home_days_rest,
            'away_days_rest': away_days_rest,
            'game_of_season': game_of_season,
            'travel_factor': travel_factor
        }
        
    elif sport == "nfl":
        if nfl_model is None or nfl_features_df is None:
            return jsonify({'error': 'NFL model or features not loaded.'}), 503
        
        home_team_standard = NFL_TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_team_standard = NFL_TEAM_NAME_MAP.get(away_team_full, away_team_full)

        home_feats_row = nfl_features_df[nfl_features_df['team'] == home_team_standard]
        away_feats_row = nfl_features_df[nfl_features_df['team'] == away_team_standard]
        
        if home_feats_row.empty or away_feats_row.empty:
            return jsonify({'error': f'No NFL features found for {home_team_full} or {away_team_full} in the loaded features file. The file may be out of date or the team names are not available.'}), 404
        
        home_feats = home_feats_row.iloc[0].to_dict()
        away_feats = away_feats_row.iloc[0].to_dict()
        
        # Calculate new fatigue features
        home_days_rest = float(home_feats.get('home_days_rest_x', 7.0))
        away_days_rest = float(away_feats.get('away_days_rest_y', 7.0))
        game_of_season = float(home_feats.get('game_of_season_x', 1.0))

        final_features = {
            'rolling_avg_adj_pts_scored_home': float(home_feats.get('rolling_avg_adj_pts_scored_home', 21.0)),
            'rolling_avg_adj_pts_allowed_home': float(home_feats.get('rolling_avg_adj_pts_allowed_home', 21.0)),
            'rolling_avg_adj_pts_scored_away': float(away_feats.get('rolling_avg_adj_pts_scored_away', 21.0)),
            'rolling_avg_adj_pts_allowed_away': float(away_feats.get('rolling_avg_adj_pts_allowed_away', 21.0)),
            'home_days_rest': home_days_rest,
            'away_days_rest': away_days_rest,
            'game_of_season': game_of_season
        }
        
    else:
        return jsonify({'error': 'Invalid sport specified. Must be "mlb" or "nfl".'}), 400

    try:
        model = mlb_model if sport == 'mlb' else nfl_model
        
        prediction_df = pd.DataFrame([final_features])[model.get_booster().feature_names]
        
        predicted_total = model.predict(prediction_df)
        
        return jsonify({'predicted_total_runs': float(predicted_total[0])})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
