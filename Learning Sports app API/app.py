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

# --- DATA CACHING & PRE-CALCULATION ---
games_df = None
batter_stats_df = None
pitcher_stats_df = None
team_features = None # Will store the final pre-calculated features

TEAM_NAME_MAP = { "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH", "ARZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK" }

def calculate_all_features():
    """
    Loads all data and runs the full feature engineering pipeline,
    caching the final features for each team.
    """
    global games_df, batter_stats_df, pitcher_stats_df, team_features
    if engine is None:
        print("Database not connected. Cannot calculate features.")
        return
    try:
        print("Loading historical data from database...")
        games_df = pd.read_sql("SELECT * FROM games", engine)
        batter_stats_df = pd.read_sql("SELECT * FROM batter_stats", engine)
        pitcher_stats_df = pd.read_sql("SELECT * FROM pitcher_stats", engine)

        # --- Data Cleaning ---
        games_df['game_id'] = games_df['game_id'].astype(str)
        batter_stats_df['game_id'] = batter_stats_df['game_id'].astype(str)
        pitcher_stats_df['game_id'] = pitcher_stats_df['game_id'].astype(str)
        stat_game_ids = set(batter_stats_df['game_id'])
        games_df = games_df[games_df['game_id'].isin(stat_game_ids)].copy()
        games_df['home_team'] = games_df['home_team'].str.strip().map(TEAM_NAME_MAP)
        games_df['away_team'] = games_df['away_team'].str.strip().map(TEAM_NAME_MAP)
        batter_stats_df['team'] = batter_stats_df['team'].str.strip().map(TEAM_NAME_MAP)
        pitcher_stats_df['team'] = pitcher_stats_df['team'].str.strip().map(TEAM_NAME_MAP)
        
        # --- Feature Engineering ---
        print("Starting feature engineering for all teams...")
        
        # Team Hitting
        batter_agg = batter_stats_df.groupby(['game_id', 'team']).agg(total_hits=('hits', 'sum'), total_homers=('home_runs', 'sum')).reset_index()
        team_game_stats = pd.merge(games_df[['game_id', 'commence_time']], batter_agg, on='game_id', how='left')
        team_game_stats.sort_values('commence_time', inplace=True)
        team_game_stats.dropna(subset=['team'], inplace=True)
        team_game_stats['rolling_avg_hits'] = team_game_stats.groupby('team')['total_hits'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        team_game_stats['rolling_avg_homers'] = team_game_stats.groupby('team')['total_homers'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        
        # Starters
        starters = pitcher_stats_df.loc[pitcher_stats_df.groupby(['game_id', 'team'])['innings_pitched'].idxmax()]
        starters = pd.merge(starters, games_df[['game_id', 'commence_time']], on='game_id', how='left')
        starters.sort_values('commence_time', inplace=True)
        epsilon = 1e-6
        rolling_ip = starters.groupby('player_name')['innings_pitched'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        rolling_er = starters.groupby('player_name')['earned_runs'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        starters['starter_rolling_era'] = (rolling_er * 9) / (rolling_ip + epsilon)
        starters['starter_rolling_ks'] = starters.groupby('player_name')['strikeouts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        # Bullpen
        bullpen_df = pitcher_stats_df[~pitcher_stats_df.index.isin(starters.index)]
        bullpen_agg = bullpen_df.groupby(['game_id', 'team']).agg(bullpen_ip=('innings_pitched', 'sum'), bullpen_er=('earned_runs', 'sum')).reset_index()
        bullpen_agg = pd.merge(games_df[['game_id', 'commence_time']], bullpen_agg, on='game_id', how='left')
        bullpen_agg.sort_values('commence_time', inplace=True)
        bullpen_agg.dropna(subset=['team'], inplace=True)
        rolling_bullpen_ip = bullpen_agg.groupby('team')['bullpen_ip'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        rolling_bullpen_er = bullpen_agg.groupby('team')['bullpen_er'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        bullpen_agg['bullpen_rolling_era'] = (rolling_bullpen_er * 9) / (rolling_bullpen_ip + epsilon)

        # Park Factors
        games_df['total_runs'] = games_df['home_score'] + games_df['away_score']
        park_factors = games_df.groupby('home_team')['total_runs'].mean().reset_index().rename(columns={'home_team': 'team', 'total_runs': 'park_factor_avg_runs'})
        
        # --- Cache the latest features for each team ---
        team_features = team_game_stats.groupby('team').last().reset_index()
        team_features = pd.merge(team_features, park_factors, on='team', how='left')
        
        # This is a simplification; a real app would also cache starter and bullpen data
        # For now, we'll use team-level averages as a proxy
        latest_starters = starters.groupby('team').last().reset_index()
        team_features = pd.merge(team_features, latest_starters[['team', 'starter_rolling_era', 'starter_rolling_ks']], on='team', how='left')
        
        latest_bullpen = bullpen_agg.groupby('team').last().reset_index()
        team_features = pd.merge(team_features, latest_bullpen[['team', 'bullpen_rolling_era']], on='team', how='left')
        
        team_features.fillna(0, inplace=True)
        print("Feature calculation complete. API is ready for predictions.")

    except Exception as e:
        print(f"An error occurred during feature calculation: {e}")

# Calculate features when the app starts
calculate_all_features()

@app.route('/features')
def get_features():
    """Returns the most recent pre-calculated features for a given team."""
    home_team_full = request.args.get('home_team')
    away_team_full = request.args.get('away_team')

    if not all([home_team_full, away_team_full]):
        return jsonify({'error': 'Missing home_team or away_team parameter'}), 400

    home_team = TEAM_NAME_MAP.get(home_team_full, home_team_full)
    away_team = TEAM_NAME_MAP.get(away_team_full, away_team_full)

    home_feats = team_features[team_features['team'] == home_team].to_dict('records')[0]
    away_feats = team_features[team_features['team'] == away_team].to_dict('records')[0]

    features = {
        'home_rolling_avg_hits': home_feats.get('rolling_avg_hits', 8.0),
        'home_rolling_avg_homers': home_feats.get('rolling_avg_homers', 1.0),
        'away_rolling_avg_hits': away_feats.get('rolling_avg_hits', 8.0),
        'away_rolling_avg_homers': away_feats.get('rolling_avg_homers', 1.0),
        'home_starter_rolling_era': home_feats.get('starter_rolling_era', 4.0),
        'home_starter_rolling_ks': home_feats.get('starter_rolling_ks', 5.0),
        'away_starter_rolling_era': away_feats.get('starter_rolling_era', 4.0),
        'away_starter_rolling_ks': away_feats.get('starter_rolling_ks', 5.0),
        'home_bullpen_rolling_era': home_feats.get('bullpen_rolling_era', 4.0),
        'away_bullpen_rolling_era': away_feats.get('bullpen_rolling_era', 4.0),
        'park_factor_avg_runs': home_feats.get('park_factor_avg_runs', 9.0)
    }
    return jsonify(features)

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
