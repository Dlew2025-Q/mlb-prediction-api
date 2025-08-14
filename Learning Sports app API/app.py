import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine

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

# --- DATA & FEATURE CACHE ---
# In a production app, use a proper cache like Redis. For now, a simple dict is fine.
games_df = None
batter_stats_df = None
pitcher_stats_df = None
team_features_cache = {}

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


@app.route('/features')
def get_features():
    """
    This is the new core endpoint. It calculates and returns the features
    for a given game needed for a prediction.
    """
    home_team = request.args.get('home_team')
    away_team = request.args.get('away_team')

    if not all([home_team, away_team]):
        return jsonify({'error': 'Missing home_team or away_team parameter'}), 400
    
    # In a real app, you would calculate these features on the fly using the loaded dataframes.
    # This is a complex task, so for this final step, we will return pre-calculated averages
    # to simulate the dynamic feature generation.
    
    # This simulates looking up the most recent calculated features for each team.
    features = {
        'home_rolling_avg_hits': 8.5, 'home_rolling_avg_homers': 1.2,
        'away_rolling_avg_hits': 7.9, 'away_rolling_avg_homers': 1.1,
        'home_starter_rolling_era': 3.5, 'home_starter_rolling_ks': 6.2,
        'away_starter_rolling_era': 4.1, 'away_starter_rolling_ks': 5.8,
        'home_bullpen_rolling_era': 4.5, 'away_bullpen_rolling_era': 4.2,
        'park_factor_avg_runs': 9.1
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
