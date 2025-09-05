#!/usr/bin/env python3
import os
import pickle
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import numpy as np

# Load environment variables from your .env file
load_dotenv()
DB_URL = os.environ.get('DATABASE_URL')

# --- CONFIGURATION ---
BACKTEST_DAYS = 30
SIMULATED_MARKET_LINE = 8.5 # Used if no market line is found in the data

# --- STRATEGIES TO TEST ---
# The script will loop through these and test each one.
STRATEGIES = [
    {"name": "High Edge, Low Confidence", "min_edge": 1.5, "min_confidence": 0.20},
    {"name": "Balanced Edge and Confidence", "min_edge": 1.0, "min_confidence": 0.30},
    {"name": "High Confidence", "min_edge": 0.5, "min_confidence": 0.40},
    {"name": "Original Strategy", "min_edge": 1.0, "min_confidence": 0.0}, # Edge only
]


# --- LOAD MODELS AND FEATURES ---
def load_pickle(path):
    """Loads a pickled object from a given file path."""
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {path} not found. Please run precompute_features.py first.")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load {path}. Error: {e}")
        return None

def get_feature(feat_dict, key, default):
    """Safely gets a feature, returning default if key is missing or value is NaN."""
    val = feat_dict.get(key)
    if val is None or pd.isna(val):
        return default
    return val

def run_backtest():
    """
    Runs a backtest of multiple strategies on the past 30 days of MLB games.
    """
    print("--- Starting Full Strategy Backtest ---")

    # Load all necessary files
    mlb_model = load_pickle('mlb_total_runs_model.pkl')
    mlb_calibration_model = load_pickle('mlb_calibration_model.pkl')
    mlb_features_df = load_pickle('latest_mlb_features.pkl')

    if mlb_model is None or mlb_calibration_model is None or mlb_features_df is None:
        return

    # Prepare data
    mlb_features_df['commence_time'] = pd.to_datetime(mlb_features_df['commence_time'], utc=True)
    
    # Connect to DB and get recent games
    if not DB_URL:
        print("Error: DATABASE_URL environment variable not found.")
        return
        
    try:
        engine = create_engine(DB_URL)
        games_query = "SELECT * FROM games"
        recent_games_df = pd.read_sql(games_query, engine)
        recent_games_df['commence_time'] = pd.to_datetime(recent_games_df['commence_time'], utc=True)
        
        if 'total' in recent_games_df.columns:
            recent_games_df.rename(columns={'total': 'market_line'}, inplace=True)
        elif 'over_under' in recent_games_df.columns:
            recent_games_df.rename(columns={'over_under': 'market_line'}, inplace=True)

        if 'market_line' not in recent_games_df.columns:
            print(f"\nWarning: Could not find a 'market_line' column in your 'games' table.")
            print(f"Proceeding with a simulated market line of {SIMULATED_MARKET_LINE} for all games.\n")
            recent_games_df['market_line'] = SIMULATED_MARKET_LINE
            
    except Exception as e:
        print(f"Error connecting to the database or fetching games: {e}")
        return

    # Backtest logic
    if recent_games_df.empty:
        print("No games found in the database to backtest.")
        return
        
    last_game_date = recent_games_df['commence_time'].max().date()
    print(f"Most recent game data found for: {last_game_date.strftime('%Y-%m-%d')}")
    print(f"Backtesting the {BACKTEST_DAYS} days of data leading up to and including this date...")

    # Calculate all predictions once to be efficient
    all_predictions = []
    for _, game in recent_games_df.iterrows():
        if game['commence_time'].date() > last_game_date or game['commence_time'].date() < last_game_date - timedelta(days=BACKTEST_DAYS):
            continue

        historical_features = mlb_features_df[mlb_features_df['commence_time'] < game['commence_time']]
        home_team, away_team = game['home_team'], game['away_team']
        last_home_game = historical_features[historical_features['home_team'] == home_team]
        last_away_game = historical_features[historical_features['away_team'] == away_team]

        if last_home_game.empty or last_away_game.empty:
            continue

        home_feats, away_feats = last_home_game.iloc[-1].to_dict(), last_away_game.iloc[-1].to_dict()
        
        final_features = {
            'rolling_avg_adj_hits_home': get_feature(home_feats, 'rolling_avg_adj_hits_home', 8.0), 'rolling_avg_adj_homers_home': get_feature(home_feats, 'rolling_avg_adj_homers_home', 1.0), 'rolling_avg_adj_walks_home': get_feature(home_feats, 'rolling_avg_adj_walks_home', 3.0), 'rolling_avg_adj_strikeouts_home': get_feature(home_feats, 'rolling_avg_adj_strikeouts_home', 8.0), 'starter_rolling_adj_era_home': get_feature(home_feats, 'starter_rolling_adj_era_home', 4.5), 'starter_rolling_whip_home': get_feature(home_feats, 'starter_rolling_whip_home', 1.3), 'starter_rolling_k_per_9_home': get_feature(home_feats, 'starter_rolling_k_per_9_home', 8.5), 'rolling_bullpen_era_home': get_feature(home_feats, 'rolling_bullpen_era_home', 4.5), 'park_factor': get_feature(home_feats, 'park_factor', 1.0), 'bullpen_ip_last_3_days_home': get_feature(home_feats, 'bullpen_ip_last_3_days_home', 0.0), 'rolling_avg_adj_hits_away': get_feature(away_feats, 'rolling_avg_adj_hits_away', 8.0), 'rolling_avg_adj_homers_away': get_feature(away_feats, 'rolling_avg_adj_homers_away', 1.0), 'rolling_avg_adj_walks_away': get_feature(away_feats, 'rolling_avg_adj_walks_away', 3.0), 'rolling_avg_adj_strikeouts_away': get_feature(away_feats, 'rolling_avg_adj_strikeouts_away', 8.0), 'starter_rolling_adj_era_away': get_feature(away_feats, 'starter_rolling_adj_era_away', 4.5), 'starter_rolling_whip_away': get_feature(away_feats, 'starter_rolling_whip_away', 1.3), 'starter_rolling_k_per_9_away': get_feature(away_feats, 'starter_rolling_k_per_9_away', 8.5), 'rolling_bullpen_era_away': get_feature(away_feats, 'rolling_bullpen_era_away', 4.5), 'bullpen_ip_last_3_days_away': get_feature(away_feats, 'bullpen_ip_last_3_days_away', 0.0), 'home_days_rest': get_feature(game, 'home_days_rest', 3), 'away_days_rest': get_feature(game, 'away_days_rest', 3), 'game_of_season': get_feature(game, 'game_of_season', 50), 'travel_factor': get_feature(game, 'travel_factor', 0), 'starter_era_diff': get_feature(away_feats, 'starter_rolling_adj_era_away', 4.5) - get_feature(home_feats, 'starter_rolling_adj_era_home', 4.5), 'bullpen_era_diff': get_feature(away_feats, 'rolling_bullpen_era_away', 4.5) - get_feature(home_feats, 'rolling_bullpen_era_home', 4.5), 'home_offense_vs_away_defense': get_feature(away_feats, 'pitching_rank', 15.5) - get_feature(home_feats, 'hitting_rank', 15.5), 'away_offense_vs_home_defense': get_feature(home_feats, 'pitching_rank', 15.5) - get_feature(away_feats, 'hitting_rank', 15.5)
        }
        feature_order = mlb_model.get_booster().feature_names
        prediction_df = pd.DataFrame([final_features], columns=feature_order)
        raw_prediction = mlb_model.predict(prediction_df)[0]
        
        confidence_df = pd.DataFrame([{'raw_prediction': raw_prediction}])
        confidence_score = mlb_calibration_model.predict_proba(confidence_df.values.reshape(-1, 1))[0][1]
        
        all_predictions.append({
            "game": f"{game['away_team']} @ {game['home_team']}",
            "market_line": game['market_line'],
            "prediction": raw_prediction,
            "edge": raw_prediction - game['market_line'],
            "confidence": confidence_score,
            "actual_score": game['home_score'] + game['away_score']
        })

    # Now, test each strategy against the pre-calculated predictions
    for strategy in STRATEGIES:
        print(f"\n--- Strategy: {strategy['name']} ---")
        
        qualified_bets = [p for p in all_predictions if abs(p['edge']) >= strategy['min_edge'] and p['confidence'] >= strategy['min_confidence']]

        if not qualified_bets:
            print("No games met the criteria for this strategy.")
            continue

        for bet in qualified_bets:
            bet['bet_type'] = "Over" if bet['edge'] > 0 else "Under"
            if bet['actual_score'] > bet['market_line']:
                bet['result'] = "WIN" if bet['bet_type'] == "Over" else "LOSS"
            elif bet['actual_score'] < bet['market_line']:
                bet['result'] = "WIN" if bet['bet_type'] == "Under" else "LOSS"
            else:
                bet['result'] = "PUSH"

        total_wins = len([b for b in qualified_bets if b['result'] == 'WIN'])
        total_losses = len([b for b in qualified_bets if b['result'] == 'LOSS'])
        total_pushes = len([b for b in qualified_bets if b['result'] == 'PUSH'])
        
        print(f"Total Bets Placed: {len(qualified_bets)}")
        print(f"Record (W-L-P): {total_wins}-{total_losses}-{total_pushes}")
        if (total_wins + total_losses) > 0:
            win_rate = total_wins / (total_wins + total_losses)
            print(f"Win Rate: {win_rate:.2%}")

if __name__ == '__main__':
    run_backtest()

