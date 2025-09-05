#!/usr/bin/env python3
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

# --- CONFIGURATION ---
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
# A specific date in the past that should have NFL games
HISTORICAL_DATE = "2023-10-22" 
SPORT_KEY = "americanfootball_nfl"

def main():
    print("--- API Response Debugger ---")
    
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY environment variable not found.")
        return

    # Construct the URL for a specific historical date
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/scores/?apiKey={ODDS_API_KEY}&date={HISTORICAL_DATE}"
    
    print(f"\nRequesting data for a single date: {HISTORICAL_DATE}")
    print(f"Request URL: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        games_data = response.json()
        
        print("\n--- API Call Successful ---")
        print(f"Found {len(games_data)} games in the response.")
        
        if games_data:
            # Print the details of the first game in the response
            first_game = games_data[0]
            print("\n--- Details of the First Game Returned by API ---")
            print(json.dumps(first_game, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\n--- Error ---")
        print(f"An error occurred while fetching data: {e}")
        print("This could be due to an invalid API key or a network issue.")

if __name__ == "__main__":
    main()
