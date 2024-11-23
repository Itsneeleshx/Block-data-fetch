# Imports for core functionality
import os
import schedule
import json
import requests
from datetime import datetime, timezone, timedelta
import time
import logging
import threading

# Imports for handling data processing (if applicable)
import numpy as np
import pandas as pd

# Imports for ML (if needed in the future)
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Imports for Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# Flask imports for creating the API
from flask import Flask, jsonify, request

# Environment variable management
from dotenv import load_dotenv

# Suppress TensorFlow CPU logs (if applicable)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables
load_dotenv()

# Ensure all required environment variables are set
required_vars = ["GOOGLE_CREDENTIALS", "OKLINK_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Environment variable '{var}' is not set.")

# Google Sheets setup
try:
    credentials_info = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
    credentials = Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    )
    client = gspread.authorize(credentials)
except json.JSONDecodeError as e:
    raise ValueError("GOOGLE_CREDENTIALS environment variable contains invalid JSON.") from e

# Set default Google Sheet name
sheet_name = os.getenv("GOOGLE_SHEET_NAME", "91club-api")
sheet = client.open(sheet_name).sheet1

# OKLink API setup
API_BASE_URL = "https://www.oklink.com/api/v5/explorer"
API_KEY = os.getenv("OKLINK_API_KEY")
CHAIN_SHORT_NAME = "TRON"

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Flask app setup
app = Flask(__name__)

# Global variable to store the LSTM model (loaded once)
lstm_model = None

# Ensure global variables are defined
sequence = []  # Store historical sequence of last digits
SEQUENCE_LENGTH = 30  # Length of sequence for LSTM input
latest_probability = None  # Store the latest prediction

# Function to load the LSTM model
def load_lstm_model():
    global lstm_model
    if lstm_model is None:
        lstm_model = tf.keras.models.load_model("model.h5")
        print("LSTM model loaded successfully.")

# Function to predict the next last digit using the LSTM model
def predict_next_digit(last_digits):
    # Ensure the model is loaded
    load_lstm_model()

    # Preprocess data for prediction
    input_data = preprocess_data_for_lstm(last_digits)

    # Make prediction
    prediction = lstm_model.predict(input_data)
    predicted_digit = np.argmax(prediction)  # Get the digit with the highest probability
    probability = prediction[0][predicted_digit]  # Probability of the predicted digit

    return predicted_digit, probability

# Route: Home endpoint
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask API"})

# Route: Fetch block data
@app.route('/get_block', methods=['GET'])
def get_block():
    try:
        # Calculate target timestamp
        target_time, target_timestamp_ms = calculate_target_timestamp()

        # Fetch block height
        block_height, block_time = get_block_height_by_time(target_timestamp_ms)
        if block_height is None or block_time is None:
            return jsonify({"error": "Failed to fetch block data"}), 500

        # Fetch block hash
        block_hash = get_block_hash_by_height(block_height)
        if block_hash is None:
            return jsonify({"error": "Failed to fetch block hash"}), 500

        # Convert the last character of the block hash to an integer
        try:
            last_digit = int(block_hash[-1], 16)
        except ValueError as e:
            # Handle invalid conversions (e.g., if block_hash[-1] is not a valid hex character)
            logging.error(f"Error converting block hash to last digit: {e}")
            last_digit = None

        # Update Google Sheet with the last digit
        if last_digit is not None:
            try:
                sheet.append_row([str(datetime.now()), block_height, block_hash, last_digit])
                logging.info(f"Last digit {last_digit} sent to Google Sheet successfully.")
            except Exception as e:
                logging.error(f"Error updating Google Sheet: {str(e)}")
                return jsonify({"error": "Failed to update Google Sheet"}), 500

            # Send last digit to LSTM model and get prediction
            try:
                predicted_next_digit = update_and_predict_lstm(last_digit)
                logging.info(f"LSTM predicted next digit: {predicted_next_digit}")
                return jsonify({
                    "block_height": block_height,
                    "block_hash": block_hash,
                    "last_digit": last_digit,
                    "predicted_next_digit": predicted_next_digit
                })
            except Exception as e:
                logging.error(f"Error predicting next digit with LSTM: {str(e)}")
                return jsonify({"error": "Failed to predict next digit"}), 500
        else:
            logging.warning("Invalid last digit. Skipping Google Sheet update and prediction.")
            return jsonify({"error": "Invalid last digit"}), 500

    except Exception as e:
        logging.error(f"Error fetching block: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Shared variable to store the latest probability
latest_probability = None

# Route: Get the latest prediction
@app.route('/get-latest-prediction', methods=['GET'])
def get_latest_prediction():
    return jsonify({"latest_probability": latest_probability})

# Function to log block data to Google Sheet
def log_to_google_sheet(last_digit):
    timestamp = datetime.datetime.now().isoformat()
    sheet.append_row([timestamp, "Last Digit", last_digit])

# Function to log predictions to Google Sheet
def log_to_google_sheet_prediction(predicted_digit, probabilities):
    timestamp = datetime.datetime.now().isoformat()
    sheet.append_row([timestamp, "Prediction", predicted_digit, probabilities])

# Function: Calculate target timestamp
def calculate_target_timestamp():
    now = datetime.now(timezone.utc)
    target_time = now.replace(second=54, microsecond=0)
    if now.second >= 54:
        target_time += timedelta(minutes=1)
    return target_time, int(target_time.timestamp() * 1000)

# Function: Fetch block height by time
def get_block_height_by_time(target_timestamp_ms):
    delay = 8  # Wait to ensure the block is indexed
    logging.info(f"Delaying for {delay} seconds to ensure the block is created...")
    time.sleep(delay)

    params = {"chainShortName": CHAIN_SHORT_NAME, "time": target_timestamp_ms}
    headers = {"Ok-Access-Key": API_KEY}
    url = f"{API_BASE_URL}/block/block-height-by-time"
    response = requests.get(url, headers=headers, params=params)
    response_data = response.json()

    if response.status_code == 200 and response_data["code"] == "0":
        block_data = response_data.get("data", [])
        if block_data:
            block_height = block_data[0]["height"]
            block_time = int(block_data[0]["blockTime"])
            return int(block_height), datetime.fromtimestamp(block_time / 1000, tz=timezone.utc)
    logging.error(f"Error fetching block height: {response.text}")
    return None, None

# Function: Fetch block hash by height
def get_block_hash_by_height(block_height):
    params = {"chainShortName": CHAIN_SHORT_NAME, "height": block_height}
    headers = {"Ok-Access-Key": API_KEY}
    url = f"{API_BASE_URL}/block/block-fills"
    response = requests.get(url, headers=headers, params=params)
    response_data = response.json()

    if response.status_code == 200 and response_data["code"] == "0":
        block_data = response_data.get("data", [])
        if block_data:
            block_hash = block_data[0]["hash"]
            return block_hash
    logging.error(f"Error fetching block hash: {response.text}")
    return None



# Extract the last numerical digit
def extract_last_numerical_digit(block_hash):
    for char in reversed(block_hash):
        if char.isdigit():
            return char
    return None

# Function to fetch data from Google Sheets
def fetch_data():
    data = pd.DataFrame(sheet.get_all_records())
    if 'Last Numerical Digit' not in data.columns:
        raise ValueError("The column 'Last Numerical Digit' is missing in the data.")
    return data

# Preprocess data for LSTM
def preprocess_data(data, sequence_length=9):
    data['Last Numerical Digit'] = pd.to_numeric(data['Last Numerical Digit'], errors='coerce').fillna(0).astype(int)
    data['Category'] = data['Last Numerical Digit'].apply(lambda x: 0 if x <= 4 else 1)
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data['Category'].iloc[i:i + sequence_length].values)
        labels.append(data['Category'].iloc[i + sequence_length])
    return np.array(sequences), np.array(labels)

# Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the LSTM model
def train_model(model, X, y, epochs=10, batch_size=32):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

# Predict the next digit and category
def predict_next_digit(model, sequence):
    sequence = np.expand_dims(sequence, axis=0)
    sequence = np.expand_dims(sequence, axis=2)
    prob = model.predict(sequence)[0][0]
    category = 'Small' if prob <= 0.5 else 'Big'
    predicted_digit = np.round(prob * 9).astype(int)
    return predicted_digit, category, prob * 100

# Main function
def main():
    sequence_length = 9
    try:
        # Fetch and preprocess initial data
        data = fetch_data()
        X, y = preprocess_data(data, sequence_length=sequence_length)
        X = np.expand_dims(X, axis=2)
        model = build_model(input_shape=(X.shape[1], X.shape[2]))
        model = train_model(model, X, y, epochs=20, batch_size=32)

        while True:
            target_time, target_timestamp_ms = calculate_target_timestamp()
            while datetime.now(timezone.utc) < target_time:
                time.sleep(0.1)

            block_height, block_time = get_block_height_by_time(target_timestamp_ms)
            if block_height and block_time and block_time.second == 54:
                block_hash = get_block_hash_by_height(block_height)
                if block_hash:
                    last_digit = extract_last_numerical_digit(block_hash)
                    if last_digit:
                        sheet.append_row([datetime.now().isoformat(), block_height, last_digit])
                        data = fetch_data()
                        X_new, _ = preprocess_data(data, sequence_length=sequence_length)
                        X_new = np.expand_dims(X_new[-1], axis=0)
                        X_new = np.expand_dims(X_new, axis=2)
                        digit, category, prob = predict_next_digit(model, X_new[-1])
                        print(f"Predicted Next Digit: {digit} ({prob:.2f}%)")
                        print(f"Predicted Category: {category}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Process terminated by user.")

def fetch_and_log_block_data():
    try:
        # Calculate target timestamp
        target_time, target_timestamp_ms = calculate_target_timestamp()

        # Fetch block height
        block_height, block_time = get_block_height_by_time(target_timestamp_ms)
        if block_height is None or block_time is None:
            logging.error("Failed to fetch block data.")
            return

        # Fetch block hash
        block_hash = get_block_hash_by_height(block_height)
        if block_hash is None:
            logging.error("Failed to fetch block hash.")
            return

        # Extract last digit of block hash
        last_digit = int(block_hash[-1], 16)

        # Log last digit to Google Sheets
        log_to_google_sheet(last_digit)

        # Process prediction using the LSTM model
        process_prediction(last_digit)

        logging.info(f"Block height: {block_height}, Block time: {block_time}, Block hash: {block_hash}")
    except Exception as e:
        logging.error(f"Error in fetch_and_log_block_data: {str(e)}")

# Function to start the scheduling cycle
def start_cycle():
    # Schedule the block-fetching process to run every minute at the 54th second
    schedule.every().minute.at(":54").do(fetch_and_log_block_data)

    # Continuously check for scheduled tasks
    while True:
        schedule.run_pending()
        sleep(1)

# Start the cycle in a separate thread
background_thread = threading.Thread(target=start_cycle, daemon=True)
background_thread.start()

# Run Flask app (for local testing only)
if __name__ == "__main__":
    app.run(debug=True)