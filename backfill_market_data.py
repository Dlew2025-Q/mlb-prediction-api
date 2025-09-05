import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import requests

# --- CONFIGURATION ---
load_dotenv()
DB_URL = os.environ.get('DATABASE_URL')
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')we need to 
BATCH_SIZE = 100 # Keep this batch size small due to high API cost
TARGET_YEAR = 2025

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        print(f"Error: Could not connect to the database. {e}")
        return None

def get_historical_line(game_date, game_id):
    """Fetches the average 'totals' line for a game at a specific timestamp."""
    date_str = game_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # --- FIX: Corrected the API endpoint URL ---
    url = f"https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=totals&date={date_str}"
    
    try:
        response = requests.get(url)
        if response.status_code == 429:
            print(" -> Rate limit hit. Pausing for 60 seconds.")
            time.sleep(60)
            response = requests.get(url) # Retry once
        
        response.raise_for_status()
        json_data = response.json()
        
        # --- FIX: Correctly parse the JSON response ---
        # The list of games is inside the 'data' key.
        games_list = json_data.get('data', [])
        if not games_list:
            return None

        # Find the specific game in the response list
        game_data = next((g for g in games_list if g['id'] == game_id), None)

        if not game_data:
            return None

        # Calculate the average line from all bookmakers for that game
        lines = [
            outcome['point'] 
            for bookmaker in game_data.get('bookmakers', []) 
            for market in bookmaker.get('markets', []) 
            if market.get('key') == 'totals' 
            for outcome in market.get('outcomes', [])
        ]
        
        return sum(lines) / len(lines) if lines else None

    except Exception as e:
        print(f"  -> Could not fetch historical odds for {game_id}: {e}")
        return None

def insert_market_data(conn, game_id, opening_line, closing_line):
    """Inserts the opening and closing lines into the database."""
    query = "INSERT INTO betting_market_data (game_id, opening_line, closing_line) VALUES (%s, %s, %s) ON CONFLICT (game_id) DO NOTHING;"
    with conn.cursor() as cur:
        cur.execute(query, (game_id, opening_line, closing_line))
    conn.commit()

def main():
    """Main function to run the market data backfill."""
    if not DB_URL or not ODDS_API_KEY:
        print("Error: DB_URL and ODDS_API_KEY must be set in your .env file.")
        return

    conn = get_db_connection()
    if not conn: return

    try:
        print(f"Finding games from the {TARGET_YEAR} season missing market data...")
        query = f"""
        SELECT g.game_id, g.commence_time FROM games g
        LEFT JOIN betting_market_data bmd ON g.game_id = bmd.game_id
        WHERE bmd.game_id IS NULL AND EXTRACT(YEAR FROM g.commence_time) = {TARGET_YEAR}
        ORDER BY g.commence_time ASC LIMIT {BATCH_SIZE};
        """
        games_to_fetch_df = pd.read_sql(query, conn)
        
        if games_to_fetch_df.empty:
            print("No more games to fetch market data for this year. Backfill is complete!")
            return

        print(f"Found {len(games_to_fetch_df)} games in this batch. Fetching lines...")

        for _, game in games_to_fetch_df.iterrows():
            game_id = str(game['game_id'])
            commence_time = game['commence_time']
            
            # Get opening line (approximated as 24 hours before game time)
            opening_time = commence_time - timedelta(hours=24)
            opening_line = get_historical_line(opening_time, game_id)
            time.sleep(2) # Pause between requests

            # Get closing line (approximated as 5 minutes before game time)
            closing_time = commence_time - timedelta(minutes=5)
            closing_line = get_historical_line(closing_time, game_id)
            time.sleep(2)

            if opening_line is not None and closing_line is not None:
                print(f"  -> Game {game_id}: Opening Line={opening_line:.2f}, Closing Line={closing_line:.2f}")
                insert_market_data(conn, game_id, opening_line, closing_line)
            else:
                print(f"  -> Skipping Game {game_id} due to missing line data.")

    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")
    
    print("Market data batch processing finished.")

if __name__ == "__main__":
    main()
