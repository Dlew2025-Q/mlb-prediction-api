import os
import pickle
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- SETUP ---
# This script should be run locally, not on Render.
# It needs a .env file with your DATABASE_URL.
load_dotenv()
DB_URL = os.environ.get('DATABASE_URL')

TEAM_NAME_MAP = { "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KC", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDP": "SD", "SD": "SD", "SFG": "SF", "SF": "SF", "SEA": "SEA", "STL": "STL", "TBR": "TB", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSN": "WSH", "WAS": "WSH", "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH", "ARZ": "ARI", "CWS": "CHW", "METS": "NYM", "YANKEES": "NYY", "ATH": "OAK" }

def main():
    """Connects to the DB, computes the latest features for all teams, and saves them to a file."""
    if not DB_URL:
        print("Error: DATABASE_URL not found in .env file.")
        return

    try:
        engine = create_engine(DB_URL)
        print("Connecting to database...")
        games_df = pd.read_sql("SELECT * FROM games", engine)
        batter_stats_df = pd.read_sql("SELECT * FROM batter_stats", engine)
        pitcher_stats_df = pd.read_sql("SELECT * FROM pitcher_stats", engine)
        print("Data loaded successfully.")

        # --- Data Cleaning ---
        games_df['game_id'] = games_df['game_id'].astype(str)
        batter_stats_df['game_id'] = batter_stats_df['game_id'].astype(str)
        pitcher_stats_df['game_id'] = pitcher_stats_df['game_id'].astype(str)
        games_df['home_team'] = games_df['home_team'].str.strip().map(TEAM_NAME_MAP)
        games_df['away_team'] = games_df['away_team'].str.strip().map(TEAM_NAME_MAP)
        batter_stats_df['team'] = batter_stats_df['team'].str.strip().map(TEAM_NAME_MAP)
        pitcher_stats_df['team'] = pitcher_stats_df['team'].str.strip().map(TEAM_NAME_MAP)
        
        # --- Feature Engineering ---
        print("Calculating features for all teams...")
        batter_agg = batter_stats_df.groupby(['game_id', 'team']).agg(total_hits=('hits', 'sum'), total_homers=('home_runs', 'sum')).reset_index()
        team_game_stats = pd.merge(games_df[['game_id', 'commence_time']], batter_agg, on='game_id', how='left')
        team_game_stats.sort_values('commence_time', inplace=True)
        team_game_stats['rolling_avg_hits'] = team_game_stats.groupby('team')['total_hits'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        team_game_stats['rolling_avg_homers'] = team_game_stats.groupby('team')['total_homers'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        
        # --- Get the most recent calculated stats for each team ---
        latest_features = team_game_stats.groupby('team').last().reset_index()
        
        # Add simplified placeholder stats for pitching and park factors
        latest_features['starter_rolling_era'] = 4.5
        latest_features['starter_rolling_ks'] = 5.5
        latest_features['bullpen_rolling_era'] = 4.2
        latest_features['park_factor_avg_runs'] = 9.0

        # --- Save to File ---
        output_filename = 'latest_features.pkl'
        with open(output_filename, 'wb') as file:
            pickle.dump(latest_features, file)
        
        print(f"\nSuccessfully pre-computed and saved features to '{output_filename}'")
        print("You can now add this file to your API's GitHub repository.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
