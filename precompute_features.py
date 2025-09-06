#!/usr/bin/env python3
import os
import pickle
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np
from xgboost import XGBRegressor
import pytz
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
import warnings

# Load environment variables from your .env file
load_dotenv()
DB_URL = os.environ.get('DATABASE_URL')
REDEPLOY_HOOK_URL = os.environ.get('REDEPLOY_HOOK_URL')

# Suppress the UserWarning from pandas about merging on non-unique columns
warnings.filterwarnings('ignore', category=UserWarning, message='You are merging on an intermediate')

# Mapping for MLB team names
MLB_TEAM_NAME_MAP = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals", "KC": "Kansas City Royals", "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees",
    "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SD": "San Diego Padres", "SFG": "San Francisco Giants", "SF": "San Francisco Giants", "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals", "WAS": "Washington Nationals",
    "Arizona Diamondbacks": "Arizona Diamondbacks", "Atlanta Braves": "Atlanta Braves", "Baltimore Orioles": "Baltimore Orioles", "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs", "Chicago White Sox": "Chicago White Sox", "Cincinnati Reds": "Cincinnati Reds", "Cleveland Guardians": "Cleveland Guardians", "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers", "Houston Astros": "Houston Astros", "Kansas City Royals": "Kansas City Royals", "Los Angeles Angels": "Los Angeles Angels", "Los Angeles Dodgers": "Los Angeles Dodgers", "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers", "Minnesota Twins": "Minnesota Twins", "New York Mets": "New York Mets", "New York Yankees": "New York Yankees", "Oakland Athletics": "Oakland Athletics", "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates", "San Diego Padres": "San Diego Padres", "San Francisco Giants": "San Francisco Giants", "Seattle Mariners": "Seattle Mariners", "St. Louis Cardinals": "St. Louis Cardinals", "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers", "Toronto Blue Jays": "Toronto Blue Jays", "Washington Nationals": "Washington Nationals",
    "Diamondbacks": "Arizona Diamondbacks", "D-backs": "Arizona Diamondbacks", "Braves": "Atlanta Braves", "Orioles": "Baltimore Orioles", "Red Sox": "Boston Red Sox", "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox", "Reds": "Cincinnati Reds", "Guardians": "Cleveland Guardians", "Indians": "Cleveland Guardians", "Rockies": "Colorado Rockies", "Angels": "Los Angeles Angels",
    "Dodgers": "Los Angeles Dodgers", "Marlins": "Miami Marlins", "Brewers": "Milwaukee Brewers", "Twins": "Minnesota Twins", "Mets": "New York Mets", "Yankees": "New York Yankees",
    "Athletics": "Oakland Athletics", "Phillies": "Philadelphia Phillies", "Pirates": "Pittsburgh Pirates", "Padres": "San Diego Padres", "Giants": "San Francisco Giants", "Mariners": "Seattle Mariners",
    "Cardinals": "St. Louis Cardinals", "Rays": "Tampa Bay Rays", "Rangers": "Texas Rangers", "Blue Jays": "Toronto Blue Jays", "Nationals": "Washington Nationals",
    "ARZ": "Arizona Diamondbacks", "AZ": "Arizona Diamondbacks", "CWS": "Chicago White Sox", "NY Mets": "New York Mets", "WSH Nationals": "Washington Nationals",
    "METS": "New York Mets", "YANKEES": "New York Yankees", "ATH": "Oakland Athletics", "WSH": "Washington Nationals"
}

# Mapping for NFL team names
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

# Add realistic park factors (based on 3-year averages for runs)
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


def precompute_mlb_features(engine):
    print("--- Starting MLB Feature Pre-computation ---")
    try:
        # Load raw data
        games_df = pd.read_sql("SELECT * FROM games", engine)
        batter_stats_df = pd.read_sql("SELECT * FROM batter_stats", engine)
        pitcher_stats_df = pd.read_sql("SELECT * FROM pitcher_stats", engine)
        print("MLB data loaded successfully.")
        
        # --- Data Cleaning & Standardization ---
        print("Standardizing team names...")
        
        games_df['home_team'] = games_df['home_team'].str.strip().map(MLB_TEAM_NAME_MAP)
        games_df['away_team'] = games_df['away_team'].str.strip().map(MLB_TEAM_NAME_MAP)
        batter_stats_df['team'] = batter_stats_df['team'].str.strip().map(MLB_TEAM_NAME_MAP)
        pitcher_stats_df['team'] = pitcher_stats_df['team'].str.strip().map(MLB_TEAM_NAME_MAP)
        
        missing_teams_mlb = set(games_df['home_team'].dropna().unique()) | set(games_df['away_team'].dropna().unique())
        missing_from_city_map = [team for team in missing_teams_mlb if team not in CITY_TIMEZONE_MAP]
        if missing_from_city_map:
            raise ValueError(f"The following standardized team names are in your data but not in CITY_TIMEZONE_MAP: {', '.join(missing_from_city_map)}")

        batter_stats_df.fillna(0, inplace=True)
        pitcher_stats_df.fillna(0, inplace=True)
        
        games_df['commence_time'] = pd.to_datetime(games_df['commence_time'].astype(str), utc=True)
        games_df.sort_values('commence_time', inplace=True)

        # --- Create all feature DataFrames ---
        print("Calculating and merging features...")
        
        pitcher_stats_df_temp = pd.merge(pitcher_stats_df, games_df[['game_id', 'commence_time']], on='game_id', how='left')
        pitcher_stats_df_temp.sort_values('commence_time', inplace=True)
        
        bullpen_df = pitcher_stats_df_temp[pitcher_stats_df_temp['innings_pitched'] < 3.0].copy()
        bullpen_df['bullpen_ip_last_3_days'] = bullpen_df.groupby('team')['innings_pitched'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        
        bullpen_game_stats = bullpen_df.groupby(['game_id', 'team']).agg(
            bullpen_er=('earned_runs', 'sum'),
            bullpen_ip=('innings_pitched', 'sum')
        ).reset_index()
        
        bullpen_game_stats['rolling_bullpen_er'] = bullpen_game_stats.groupby('team')['bullpen_er'].transform(lambda x: x.shift(1).rolling(15, min_periods=1).sum())
        bullpen_game_stats['rolling_bullpen_ip'] = bullpen_game_stats.groupby('team')['bullpen_ip'].transform(lambda x: x.shift(1).rolling(15, min_periods=1).sum())
        bullpen_game_stats['rolling_bullpen_era'] = (bullpen_game_stats['rolling_bullpen_er'] * 9) / (bullpen_game_stats['rolling_bullpen_ip'] + 1e-6)
        
        bullpen_agg = bullpen_df.groupby(['game_id', 'team']).agg(bullpen_ip_last_3_days=('bullpen_ip_last_3_days', 'sum')).reset_index()
        bullpen_agg = pd.merge(bullpen_agg, bullpen_game_stats[['game_id', 'team', 'rolling_bullpen_era']], on=['game_id', 'team'], how='left')


        starters_df = pitcher_stats_df_temp[pitcher_stats_df_temp['innings_pitched'] >= 3.0].copy()
        
        starters_df['whip_numerator'] = starters_df['walks'] + starters_df['hits_allowed']
        starters_df['starter_rolling_era'] = starters_df.groupby('team')['earned_runs'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / (starters_df.groupby('team')['innings_pitched'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) + 1e-6) * 9
        starters_df['starter_rolling_whip'] = starters_df.groupby('team')['whip_numerator'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / (starters_df.groupby('team')['innings_pitched'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) + 1e-6)
        starters_df['starter_rolling_k_per_9'] = starters_df.groupby('team')['strikeouts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / (starters_df.groupby('team')['innings_pitched'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) + 1e-6) * 9
        
        starters_agg = starters_df.groupby(['game_id', 'team']).last().reset_index()
        
        batter_stats_df_merged = pd.merge(batter_stats_df, games_df[['game_id', 'commence_time']], on='game_id', how='left').sort_values('commence_time')
        
        batter_agg = batter_stats_df_merged.groupby(['game_id', 'team']).agg(
            total_hits=('hits', 'sum'),
            total_homers=('home_runs', 'sum'),
            total_walks=('walks', 'sum'),
            total_strikeouts=('strikeouts', 'sum')
        ).reset_index()
        batter_agg = pd.merge(batter_agg, games_df[['game_id', 'commence_time', 'home_team', 'away_team']], on='game_id', how='left').sort_values('commence_time')
        
        batter_agg['location'] = np.where(batter_agg['team'] == batter_agg['home_team'], 'Home', 'Away')
        
        batter_agg['rolling_avg_hits_loc'] = batter_agg.groupby(['team', 'location'])['total_hits'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        batter_agg['rolling_avg_homers_loc'] = batter_agg.groupby(['team', 'location'])['total_homers'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        batter_agg['rolling_avg_walks_loc'] = batter_agg.groupby(['team', 'location'])['total_walks'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        batter_agg['rolling_avg_strikeouts_loc'] = batter_agg.groupby(['team', 'location'])['total_strikeouts'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

        # --- Consolidate all data into a single master DataFrame ---
        mlb_final_df = games_df.copy()
        
        mlb_final_df = pd.merge(mlb_final_df, bullpen_agg.rename(columns={'team': 'home_team'}), on=['game_id', 'home_team'], how='left')
        mlb_final_df = pd.merge(mlb_final_df, bullpen_agg.rename(columns={'team': 'away_team'}), on=['game_id', 'away_team'], how='left', suffixes=('', '_away'))
        mlb_final_df.rename(columns={'bullpen_ip_last_3_days': 'bullpen_ip_last_3_days_home', 'rolling_bullpen_era': 'rolling_bullpen_era_home'}, inplace=True)
        mlb_final_df.rename(columns={'bullpen_ip_last_3_days_away': 'bullpen_ip_last_3_days_away', 'rolling_bullpen_era_away': 'rolling_bullpen_era_away'}, inplace=True)

        mlb_final_df['bullpen_ip_last_3_days_home'].fillna(0, inplace=True)
        mlb_final_df['bullpen_ip_last_3_days_away'].fillna(0, inplace=True)
        mlb_final_df['rolling_bullpen_era_home'].fillna(4.5, inplace=True)
        mlb_final_df['rolling_bullpen_era_away'].fillna(4.5, inplace=True)
        
        starter_features_to_merge = ['game_id', 'team', 'starter_rolling_era', 'starter_rolling_whip', 'starter_rolling_k_per_9']
        mlb_final_df = pd.merge(mlb_final_df, starters_agg[starter_features_to_merge].rename(columns={'team': 'home_team', 'starter_rolling_era': 'starter_rolling_era_home', 'starter_rolling_whip': 'starter_rolling_whip_home', 'starter_rolling_k_per_9': 'starter_rolling_k_per_9_home'}), on=['game_id', 'home_team'], how='left')
        mlb_final_df = pd.merge(mlb_final_df, starters_agg[starter_features_to_merge].rename(columns={'team': 'away_team', 'starter_rolling_era': 'starter_rolling_era_away', 'starter_rolling_whip': 'starter_rolling_whip_away', 'starter_rolling_k_per_9': 'starter_rolling_k_per_9_away'}), on=['game_id', 'away_team'], how='left')
        mlb_final_df['starter_rolling_era_home'].fillna(4.5, inplace=True)
        mlb_final_df['starter_rolling_era_away'].fillna(4.5, inplace=True)
        mlb_final_df['starter_rolling_whip_home'].fillna(1.3, inplace=True)
        mlb_final_df['starter_rolling_whip_away'].fillna(1.3, inplace=True)
        mlb_final_df['starter_rolling_k_per_9_home'].fillna(8.5, inplace=True)
        mlb_final_df['starter_rolling_k_per_9_away'].fillna(8.5, inplace=True)
        
        home_batter_rolling = batter_agg[batter_agg['location'] == 'Home'].copy().rename(columns={
            'rolling_avg_hits_loc': 'rolling_avg_hits_home',
            'rolling_avg_homers_loc': 'rolling_avg_homers_home',
            'rolling_avg_walks_loc': 'rolling_avg_walks_home',
            'rolling_avg_strikeouts_loc': 'rolling_avg_strikeouts_home',
        })[['game_id', 'home_team', 'rolling_avg_hits_home', 'rolling_avg_homers_home', 'rolling_avg_walks_home', 'rolling_avg_strikeouts_home']]
        
        away_batter_rolling = batter_agg[batter_agg['location'] == 'Away'].copy().rename(columns={
            'rolling_avg_hits_loc': 'rolling_avg_hits_away',
            'rolling_avg_homers_loc': 'rolling_avg_homers_away',
            'rolling_avg_walks_loc': 'rolling_avg_walks_away',
            'rolling_avg_strikeouts_loc': 'rolling_avg_strikeouts_away',
        })[['game_id', 'away_team', 'rolling_avg_hits_away', 'rolling_avg_homers_away', 'rolling_avg_walks_away', 'rolling_avg_strikeouts_away']]
        
        mlb_final_df = pd.merge(mlb_final_df, home_batter_rolling, on=['game_id', 'home_team'], how='left')
        mlb_final_df = pd.merge(mlb_final_df, away_batter_rolling, on=['game_id', 'away_team'], how='left')

        mlb_final_df['rolling_avg_hits_home'].fillna(8.0, inplace=True)
        mlb_final_df['rolling_avg_homers_home'].fillna(1.0, inplace=True)
        mlb_final_df['rolling_avg_walks_home'].fillna(3.0, inplace=True)
        mlb_final_df['rolling_avg_strikeouts_home'].fillna(8.0, inplace=True)
        mlb_final_df['rolling_avg_hits_away'].fillna(8.0, inplace=True)
        mlb_final_df['rolling_avg_homers_away'].fillna(1.0, inplace=True)
        mlb_final_df['rolling_avg_walks_away'].fillna(3.0, inplace=True)
        mlb_final_df['rolling_avg_strikeouts_away'].fillna(8.0, inplace=True)

        all_games_for_rest = pd.melt(games_df, id_vars=['game_id', 'commence_time'], value_vars=['home_team', 'away_team'], value_name='team').sort_values('commence_time')
        all_games_for_rest['days_rest'] = all_games_for_rest.groupby('team')['commence_time'].diff().dt.days
        home_rest = all_games_for_rest[all_games_for_rest['variable'] == 'home_team'][['game_id', 'days_rest']].rename(columns={'days_rest': 'home_days_rest'})
        away_rest = all_games_for_rest[all_games_for_rest['variable'] == 'away_team'][['game_id', 'days_rest']].rename(columns={'days_rest': 'away_days_rest'})

        mlb_final_df = pd.merge(mlb_final_df, home_rest, on='game_id', how='left')
        mlb_final_df = pd.merge(mlb_final_df, away_rest, on='game_id', how='left')
        mlb_final_df['home_days_rest'].fillna(3, inplace=True)
        mlb_final_df['away_days_rest'].fillna(3, inplace=True)
        mlb_final_df['game_of_season'] = mlb_final_df.groupby(['home_team', mlb_final_df['commence_time'].dt.year])['commence_time'].rank(method='first')
        
        def get_tz_offset(team_name):
            tz_name = CITY_TIMEZONE_MAP.get(team_name)
            if not tz_name: return 0
            try: return datetime(2023, 7, 1, tzinfo=pytz.timezone(tz_name)).utcoffset().total_seconds() / 3600
            except pytz.UnknownTimeZoneError: return 0

        travel_df = all_games_for_rest.copy()
        travel_df['previous_game_id'] = travel_df.groupby('team')['game_id'].shift(1)
        
        game_locations = games_df[['game_id', 'home_team']].set_index('game_id')
        travel_df = travel_df.join(game_locations.rename(columns={'home_team': 'previous_location'}), on='previous_game_id')
        
        travel_df['current_location'] = travel_df['game_id'].map(game_locations['home_team'])
        
        away_travel_df = travel_df[travel_df['variable'] == 'away_team'].copy()
        away_travel_df['travel_factor'] = away_travel_df.apply(lambda row: abs(get_tz_offset(row['current_location']) - get_tz_offset(row['previous_location'])), axis=1)

        mlb_final_df = pd.merge(mlb_final_df, away_travel_df[['game_id', 'travel_factor']], on='game_id', how='left')
        mlb_final_df['travel_factor'].fillna(0, inplace=True)

        mlb_final_df['park_factor'] = mlb_final_df['home_team'].map(PARK_FACTOR_MAP).fillna(1.0)
        
        mlb_final_df['starter_era_diff'] = mlb_final_df['starter_rolling_era_away'] - mlb_final_df['starter_rolling_era_home']
        mlb_final_df['bullpen_era_diff'] = mlb_final_df['rolling_bullpen_era_away'] - mlb_final_df['rolling_bullpen_era_home']


        print(f"Final MLB training DataFrame columns: {mlb_final_df.columns.tolist()}")

        mlb_api_features = mlb_final_df.copy()
        output_filename_api = 'latest_mlb_features.pkl'
        with open(output_filename_api, 'wb') as file:
            pickle.dump(mlb_api_features, file)
        print(f"\nSuccessfully pre-computed and saved MLB features to '{output_filename_api}'")

        # --- MLB Model Training and Saving ---
        mlb_training_df = mlb_final_df.copy()
        mlb_training_df['total_runs'] = mlb_training_df['home_score'] + mlb_training_df['away_score']

        mlb_model_features = [
            'rolling_avg_hits_home', 'rolling_avg_homers_home', 'rolling_avg_walks_home',
            'rolling_avg_strikeouts_home', 'starter_rolling_era_home', 
            'starter_rolling_whip_home', 'starter_rolling_k_per_9_home', 
            'rolling_bullpen_era_home', 'park_factor',
            'bullpen_ip_last_3_days_home', 'rolling_avg_hits_away', 'rolling_avg_homers_away',
            'rolling_avg_walks_away', 'rolling_avg_strikeouts_away',
            'starter_rolling_era_away', 'starter_rolling_whip_away', 'starter_rolling_k_per_9_away',
            'rolling_bullpen_era_away', 'bullpen_ip_last_3_days_away',
            'home_days_rest', 'away_days_rest',
            'game_of_season', 'travel_factor',
            'starter_era_diff', 'bullpen_era_diff'
        ]
        
        mlb_training_df.dropna(subset=mlb_model_features + ['total_runs'], inplace=True)
        X_mlb = mlb_training_df[mlb_model_features]
        y_mlb = mlb_training_df['total_runs']

        mlb_model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=3, 
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=-1
        )
        mlb_model.fit(X_mlb, y_mlb)

        with open('mlb_total_runs_model.pkl', 'wb') as f:
            pickle.dump(mlb_model, f)
        
        print("\nSuccessfully trained and saved MLB model.")
        
        print("\n--- MLB Feature Importance ---")
        feature_importance = pd.DataFrame({
            'feature': mlb_model.get_booster().feature_names,
            'importance': mlb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance)
        
        mlb_training_df['raw_prediction'] = mlb_model.predict(X_mlb)
        mlb_training_df['is_accurate'] = np.where(abs(mlb_training_df['raw_prediction'] - mlb_training_df['total_runs']) <= 1.5, 1, 0)
        
        X_cal_mlb = mlb_training_df[['raw_prediction']]
        y_cal_mlb = mlb_training_df['is_accurate']
        
        if len(y_cal_mlb.unique()) > 1:
            mlb_calibration_model = CalibratedClassifierCV(LogisticRegression(), method='isotonic', cv=5)
            mlb_calibration_model.fit(X_cal_mlb, y_cal_mlb)
            with open('mlb_calibration_model.pkl', 'wb') as f:
                pickle.dump(mlb_calibration_model, f)
            print("\nSuccessfully trained and saved MLB calibration model.")
        else:
            print("Warning: Not enough unique classes to train MLB calibration model. Skipping.")

    except Exception as e:
        print(f"An error occurred during MLB feature pre-computation: {e}")
        raise

def precompute_nfl_features(engine):
    print("\n--- Starting NFL Feature Pre-computation ---")
    try:
        nfl_games_df = pd.read_sql("SELECT * FROM nfl_games", engine)
        print("NFL data loaded successfully.")
        
        nfl_games_df.dropna(subset=['home_score', 'away_score'], inplace=True)
        nfl_games_df['commence_time'] = pd.to_datetime(nfl_games_df['commence_time'])
        nfl_games_df.sort_values('commence_time', inplace=True)

        nfl_games_df['home_team'] = nfl_games_df['home_team'].str.strip().map(NFL_TEAM_NAME_MAP).fillna(nfl_games_df['home_team'])
        nfl_games_df['away_team'] = nfl_games_df['away_team'].str.strip().map(NFL_TEAM_NAME_MAP).fillna(nfl_games_df['away_team'])
        
        nfl_games_df['game_of_season'] = nfl_games_df.groupby(['home_team', nfl_games_df['commence_time'].dt.year])['commence_time'].rank(method='first')
        nfl_games_df['home_days_rest'] = nfl_games_df.groupby('home_team')['commence_time'].diff().dt.days.fillna(7)
        nfl_games_df['away_days_rest'] = nfl_games_df.groupby('away_team')['commence_time'].diff().dt.days.fillna(7)
        
        home_games = nfl_games_df[['game_id', 'commence_time', 'home_team', 'away_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'points_scored', 'away_score': 'points_allowed'})
        home_games['is_home_game'] = True
        away_games = nfl_games_df[['game_id', 'commence_time', 'away_team', 'home_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'points_scored', 'home_score': 'points_allowed'})
        away_games['is_home_game'] = False
        team_game_stats = pd.concat([home_games, away_games]).sort_values('commence_time')
        
        offensive_rank_df = team_game_stats.groupby('team')['points_scored'].mean().rank(ascending=False, method='first').reset_index(name='offensive_rank')
        defensive_rank_df = team_game_stats.groupby('team')['points_allowed'].mean().rank(ascending=True, method='first').reset_index(name='defensive_rank')

        opponent_ranks_df = pd.merge(offensive_rank_df, defensive_rank_df, on='team')
        opponent_ranks_df.rename(columns={'team': 'opponent'}, inplace=True)
        team_game_stats = pd.merge(team_game_stats, opponent_ranks_df, on='opponent', how='left')

        team_game_stats['adj_points_scored'] = team_game_stats['points_scored'] * (1 + (16.5 - team_game_stats['defensive_rank']) / 16.5)
        team_game_stats['adj_points_allowed'] = team_game_stats['points_allowed'] * (1 + (16.5 - team_game_stats['offensive_rank']) / 16.5)
        
        team_game_stats['rolling_avg_adj_pts_scored'] = team_game_stats.groupby('team')['adj_points_scored'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        team_game_stats['rolling_avg_adj_pts_allowed'] = team_game_stats.groupby('team')['adj_points_allowed'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())

        home_stats = team_game_stats[team_game_stats['is_home_game']].copy()[['game_id', 'team', 'rolling_avg_adj_pts_scored', 'rolling_avg_adj_pts_allowed']].rename(columns={'team': 'home_team', 'rolling_avg_adj_pts_scored': 'rolling_avg_adj_pts_scored_home', 'rolling_avg_adj_pts_allowed': 'rolling_avg_adj_pts_allowed_home'})
        away_stats = team_game_stats[~team_game_stats['is_home_game']].copy()[['game_id', 'team', 'rolling_avg_adj_pts_scored', 'rolling_avg_adj_pts_allowed']].rename(columns={'team': 'away_team', 'rolling_avg_adj_pts_scored': 'rolling_avg_adj_pts_scored_away', 'rolling_avg_adj_pts_allowed': 'rolling_avg_adj_pts_allowed_away'})
        
        latest_features_nfl = nfl_games_df.copy()
        latest_features_nfl = pd.merge(latest_features_nfl, home_stats, on=['game_id', 'home_team'], how='left')
        latest_features_nfl = pd.merge(latest_features_nfl, away_stats, on=['game_id', 'away_team'], how='left')

        latest_features_nfl['temperature'] = 70.0
        latest_features_nfl['wind_speed'] = 5.0
        latest_features_nfl['humidity'] = 50.0

        output_filename = 'latest_nfl_features.pkl'
        with open(output_filename, 'wb') as file:
            pickle.dump(latest_features_nfl, file)
        
        print(f"\nSuccessfully pre-computed and saved NFL features to '{output_filename}'")
        
        nfl_model_features = [
            'rolling_avg_adj_pts_scored_home', 'rolling_avg_adj_pts_allowed_home',
            'rolling_avg_adj_pts_scored_away', 'rolling_avg_adj_pts_allowed_away',
            'home_days_rest', 'away_days_rest', 'game_of_season',
            'temperature', 'wind_speed', 'humidity'
        ]
        
        nfl_training_df = latest_features_nfl.copy()
        
        nfl_training_df.dropna(subset=['home_score', 'away_score'], inplace=True)
        nfl_training_df['total_points'] = nfl_training_df['home_score'] + nfl_training_df['away_score']
        
        nfl_training_df.dropna(subset=nfl_model_features + ['total_points'], inplace=True)
        X_nfl = nfl_training_df[nfl_model_features]
        y_nfl = nfl_training_df['total_points']

        nfl_model = XGBRegressor(objective='reg:squarederror')
        nfl_model.fit(X_nfl, y_nfl)

        with open('nfl_total_points_model.pkl', 'wb') as f:
            pickle.dump(nfl_model, f)
            
        print("\nSuccessfully trained and saved NFL model.")
        
        nfl_training_df['raw_prediction'] = nfl_model.predict(X_nfl)

        nfl_training_df['is_accurate'] = np.where(abs(nfl_training_df['raw_prediction'] - nfl_training_df['total_points']) <= 3.0, 1, 0)
        
        X_cal_nfl = nfl_training_df[['raw_prediction']]
        y_cal_nfl = nfl_training_df['is_accurate']
        
        if len(y_cal_nfl.unique()) > 1:
            nfl_calibration_model = CalibratedClassifierCV(LogisticRegression(), method='isotonic', cv=5)
            nfl_calibration_model.fit(X_cal_nfl, y_cal_nfl)
            with open('nfl_calibration_model.pkl', 'wb') as f:
                pickle.dump(nfl_calibration_model, f)
            print("\nSuccessfully trained and saved NFL calibration model.")
        else:
            print("Warning: Not enough unique classes to train NFL calibration model. Skipping.")

    except Exception as e:
        print(f"An error occurred during NFL feature pre-computation: {e}")
        raise

def main():
    if not DB_URL:
        print("Error: DATABASE_URL environment variable not found.")
        return

    try:
        engine = create_engine(DB_URL)
        precompute_mlb_features(engine)
        precompute_nfl_features(engine)
        
        if REDEPLOY_HOOK_URL:
            print("Attempting to trigger web service redeployment...")
            response = requests.post(REDEPLOY_HOOK_URL)
            if response.status_code == 200:
                print("Web service redeployment successfully triggered.")
            else:
                print(f"Failed to trigger redeployment. Status code: {response.status_code}")
        else:
            print("REDEPLOY_HOOK_URL not set. Skipping redeployment.")

    except Exception as e:
        print(f"A critical error occurred: {e}")
        raise

if __name__ == '__main__':
    main()

