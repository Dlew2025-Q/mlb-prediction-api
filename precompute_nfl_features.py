#!/usr/bin/env python3
import os
import pickle
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np

# Load environment variables from your .env file
load_dotenv()
DB_URL = os.environ.get('DATABASE_URL')

def main():
    """
    Connects to the database, computes the latest NFL features for all teams,
    and saves them to a pickle file for the API.
    """
    print("--- Starting NFL Feature Pre-computation ---")
    if not DB_URL:
        print("Error: DATABASE_URL environment variable not found.")
        return

    try:
        engine = create_engine(DB_URL)
        print("Connecting to database and loading NFL data...")
        nfl_games_df = pd.read_sql("SELECT * FROM nfl_games", engine)
        print("Data loaded successfully.")

        # --- 1. Data Cleaning & Initial Prep ---
        print("Cleaning and preparing data...")
        nfl_games_df.dropna(subset=['home_score', 'away_score'], inplace=True)
        nfl_games_df['commence_time'] = pd.to_datetime(nfl_games_df['commence_time'])
        nfl_games_df.sort_values('commence_time', inplace=True)
        
        home_games = nfl_games_df[['game_id', 'commence_time', 'home_team', 'away_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'points_scored', 'away_score': 'points_allowed'})
        away_games = nfl_games_df[['game_id', 'commence_time', 'away_team', 'home_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'points_scored', 'home_score': 'points_allowed'})
        team_game_stats = pd.concat([home_games, away_games]).sort_values('commence_time')

        # --- 2. Calculate Team Strength Rankings ---
        print("Calculating team strength rankings...")
        offensive_rank_df = team_game_stats.groupby('team')['points_scored'].mean().rank(ascending=False, method='first').reset_index(name='offensive_rank')
        defensive_rank_df = team_game_stats.groupby('team')['points_allowed'].mean().rank(ascending=True, method='first').reset_index(name='defensive_rank')

        # --- 3. Map Opponent Strength to Each Game ---
        opponent_ranks_df = pd.merge(offensive_rank_df, defensive_rank_df, on='team')
        opponent_ranks_df.rename(columns={'team': 'opponent'}, inplace=True)
        team_game_stats = pd.merge(team_game_stats, opponent_ranks_df, on='opponent', how='left')

        # --- 4. Calculate Opponent-Adjusted Stats ---
        team_game_stats['adj_points_scored'] = team_game_stats['points_scored'] * (1 + (16.5 - team_game_stats['defensive_rank']) / 16.5)
        team_game_stats['adj_points_allowed'] = team_game_stats['points_allowed'] * (1 + (16.5 - team_game_stats['offensive_rank']) / 16.5)

        # --- 5. Feature Engineering (Rolling Averages) ---
        print("Engineering rolling average features...")
        team_game_stats['rolling_avg_adj_pts_scored'] = team_game_stats.groupby('team')['adj_points_scored'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        team_game_stats['rolling_avg_adj_pts_allowed'] = team_game_stats.groupby('team')['adj_points_allowed'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        
        # --- 6. Assemble Latest Features ---
        print("Assembling the latest features for each team...")
        latest_features = team_game_stats.groupby('team').last().reset_index()
        
        # --- 7. Save Features to File ---
        output_filename = 'latest_nfl_features.pkl'
        with open(output_filename, 'wb') as file:
            pickle.dump(latest_features, file)
        
        print(f"\nSuccessfully pre-computed and saved NFL features to '{output_filename}'")
        print("Feature columns saved:")
        print(latest_features.columns.tolist())

    except Exception as e:
        print(f"An error occurred during feature pre-computation: {e}")

if __name__ == '__main__':
    main()
