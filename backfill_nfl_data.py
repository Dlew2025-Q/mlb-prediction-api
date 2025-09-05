#!/usr/bin/env python3
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from uuid import uuid4

def backfill_nfl_data():
    """
    Script to load historical NFL data from a CSV file with specific column names,
    clean the data, and insert it into the 'nfl_games' table in the database.
    """
    # Load environment variables
    load_dotenv()
    DB_URL = os.environ.get('DATABASE_URL')

    if not DB_URL:
        print("Error: DATABASE_URL environment variable not found. Please set it in your .env file.")
        return

    # Path to your historical NFL data file
    # Make sure this file exists in your project directory
    data_file_path = 'nfl_historical_data.csv'
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return

    try:
        # Create a database engine
        engine = create_engine(DB_URL)
        print("Successfully connected to the database.")

        # Load data from the CSV file
        print(f"Loading data from {data_file_path}...")
        df = pd.read_csv(data_file_path)

        # --- Data Cleaning and Standardization ---
        print("Renaming and standardizing columns...")

        # Rename columns to match the expected database schema
        df.rename(columns={
            'schedule_date': 'commence_time',
            'team_home': 'home_team',
            'score_home': 'home_score',
            'score_away': 'away_score',
            'team_away': 'away_team',
        }, inplace=True)
        
        # NFL team mapping to handle abbreviations and standardize to full names
        # This mapping ensures consistency with the Odds API and feature computation
        nfl_team_name_map = {
            "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", 
            "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
            "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", 
            "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers", 
            "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", 
            "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
            "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", 
            "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
            "NYJ": "New York Jets", "OAK": "Las Vegas Raiders", "PHI": "Philadelphia Eagles", 
            "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", 
            "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
        }
        
        # Apply the mapping to both home and away team columns
        df['home_team'] = df['home_team'].str.strip().map(nfl_team_name_map).fillna(df['home_team'])
        df['away_team'] = df['away_team'].str.strip().map(nfl_team_name_map).fillna(df['away_team'])
        
        # Convert 'commence_time' to a proper datetime object
        df['commence_time'] = pd.to_datetime(df['commence_time'])
        
        # Generate a unique game_id for each row
        df['game_id'] = [str(uuid4()) for _ in range(len(df))]

        # Select only the columns to be inserted into the database
        columns_to_insert = [
            'game_id', 'commence_time', 'home_team', 'away_team', 'home_score', 'away_score'
        ]
        df_to_insert = df[columns_to_insert]
        
        # --- Database Insertion ---
        print("Inserting data into the 'nfl_games' table...")
        df_to_insert.to_sql('nfl_games', engine, if_exists='append', index=False)
        
        print("NFL historical data backfill complete.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    backfill_nfl_data()
