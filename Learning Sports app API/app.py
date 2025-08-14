import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS # Import the CORS library

# --- INITIALIZATION ---
# Initialize the Flask application
app = Flask(__name__)
# --- FIX: Enable CORS ---
# This will allow your front-end dashboard to make requests to this API.
CORS(app)

# --- LOAD THE TRAINED MODEL ---
# The model is loaded once when the API server starts.
model_path = 'mlb_total_runs_model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# --- API ENDPOINTS ---

@app.route('/')
def home():
    """A simple endpoint to confirm the API is running."""
    return "MLB Prediction API is live."

@app.route('/predict', methods=['POST'])
def predict():
    """
    The main prediction endpoint.
    It expects a JSON payload with the features for a game.
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Convert the incoming data into a pandas DataFrame
        features_df = pd.DataFrame([data])
        
        # Ensure the columns are in the correct order
        required_features = [
            'home_rolling_avg_hits', 'home_rolling_avg_homers',
            'away_rolling_avg_hits', 'away_rolling_avg_homers',
            'home_starter_rolling_era', 'home_starter_rolling_ks',
            'away_starter_rolling_era', 'away_starter_rolling_ks',
            'home_bullpen_rolling_era', 'away_bullpen_rolling_era',
            'park_factor_avg_runs'
        ]
        features_df = features_df[required_features]

        # Make a prediction using the loaded model
        prediction = model.predict(features_df)
        
        # Extract the first element
        predicted_runs = float(prediction[0])
        
        # Return the prediction in a JSON response
        return jsonify({'predicted_total_runs': predicted_runs})

    except Exception as e:
        # Return a detailed error message if something goes wrong
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 400

@app.route('/predict_test')
def predict_test():
    """
    A simple test endpoint to verify the model can make a prediction.
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
        
    # Create a sample data point
    test_data = {
        'home_rolling_avg_hits': 8.5, 'home_rolling_avg_homers': 1.2,
        'away_rolling_avg_hits': 7.9, 'away_rolling_avg_homers': 1.1,
        'home_starter_rolling_era': 3.5, 'home_starter_rolling_ks': 6.2,
        'away_starter_rolling_era': 4.1, 'away_starter_rolling_ks': 5.8,
        'home_bullpen_rolling_era': 4.5, 'away_bullpen_rolling_era': 4.2,
        'park_factor_avg_runs': 9.1
    }
    
    features_df = pd.DataFrame([test_data])
    prediction = model.predict(features_df)
    predicted_runs = float(prediction[0])
    
    return jsonify({
        'message': 'This is a test prediction.',
        'input_features': test_data,
        'predicted_total_runs': predicted_runs
    })

# This is required for Render's Gunicorn server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
