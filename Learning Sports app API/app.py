import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
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

# --- DATA CACHING ---
games_df = None
batter_stats_df = None
pitcher_stats_df = None

def load_data_from_db():
    """Loads all necessary data from the database into memory."""
    global games_df, batter_stats_df, pitcher_stats_df
    if engine is None:
        print("Database not connected. Cannot load data.")
        return
    try:
        print("Loading historical data from database...")
        games_df = pd.read_sql("SELECT * FROM games", engine)
        batter_stats_df = pd.read_sql("SELECT * FROM batter_stats", engine)
        pitcher_stats_df = pd.read_sql("SELECT * FROM pitcher_stats", engine)
        print("All historical data loaded into memory.")
    except Exception as e:
        print(f"Error loading data from database: {e}")

# Load data when the app starts
load_data_from_db()

# --- TEAM NAME MAP (Should match the one from training) ---
TEAM_NAME_MAP = { "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH", "ARZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK" }

@app.route('/features')
def get_features():
    """Calculates and returns the features for a given game."""
    home_team_full = request.args.get('home_team')
    away_team_full = request.args.get('away_team')

    if not all([home_team_full, away_team_full]):
        return jsonify({'error': 'Missing home_team or away_team parameter'}), 400

    # Standardize names
    home_team = TEAM_NAME_MAP.get(home_team_full, home_team_full)
    away_team = TEAM_NAME_MAP.get(away_team_full, away_team_full)

    # This is a simplified feature calculation for a live game.
    # A production system would have a more robust way of getting the latest stats.
    try:
        # Simulate getting the most recent rolling stats for each team
        home_hitting_stats = batter_stats_df[batter_stats_df['team'] == home_team].tail(10).agg(total_hits=('hits', 'mean'), total_homers=('home_runs', 'mean'))
        away_hitting_stats = batter_stats_df[batter_stats_df['team'] == away_team].tail(10).agg(total_hits=('hits', 'mean'), total_homers=('home_runs', 'mean'))
        
        # Simulate getting starter, bullpen, and park factor data
        # In a real app, you would identify the actual starters for today's game.
        home_starter_era = pitcher_stats_df[pitcher_stats_df['team'] == home_team]['earned_runs'].mean() * 9 / 5
        home_starter_ks = pitcher_stats_df[pitcher_stats_df['team'] == home_team]['strikeouts'].mean()
        away_starter_era = pitcher_stats_df[pitcher_stats_df['team'] == away_team]['earned_runs'].mean() * 9 / 5
        away_starter_ks = pitcher_stats_df[pitcher_stats_df['team'] == away_team]['strikeouts'].mean()
        home_bullpen_era = 4.0 
        away_bullpen_era = 4.0
        park_factor = games_df[games_df['home_team'] == home_team]['total_runs'].mean() if 'total_runs' in games_df else 9.0

        features = {
            'home_rolling_avg_hits': home_hitting_stats['total_hits'],
            'home_rolling_avg_homers': home_hitting_stats['total_homers'],
            'away_rolling_avg_hits': away_hitting_stats['total_hits'],
            'away_rolling_avg_homers': away_hitting_stats['total_homers'],
            'home_starter_rolling_era': home_starter_era,
            'home_starter_rolling_ks': home_starter_ks,
            'away_starter_rolling_era': away_starter_era,
            'away_starter_rolling_ks': away_starter_ks,
            'home_bullpen_rolling_era': home_bullpen_era,
            'away_bullpen_rolling_era': away_bullpen_era,
            'park_factor_avg_runs': park_factor
        }
        return jsonify(features)
    except Exception as e:
        return jsonify({'error': f'Could not calculate features: {str(e)}'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint."""
    if model is None: return jsonify({'error': 'Model is not loaded.'}), 500
    try:
        data = request.get_json()
        features_df = pd.DataFrame([data])
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
