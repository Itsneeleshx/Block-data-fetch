# Full Code with Main Function

# Required Imports
from time import sleep
import os
import schedule
import json
import requests
from datetime import datetime, timezone, timedelta
import time
import logging
import threading
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from google.oauth2.service_account import Credentials
import gspread
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# Suppress TensorFlow CPU logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables
load_dotenv()

# Ensure all required environment variables are set
required_vars = ["GOOGLE_CREDENTIALS", "OKLINK_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Environment variable '{var}' is not set.")

# Google Sheets setup
credentials_info = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
credentials = Credentials.from_service_account_info(
    credentials_info,
    scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
)
client = gspread.authorize(credentials)
sheet_name = os.getenv("GOOGLE_SHEET_NAME", "block-data")
sheet = client.open(sheet_name).sheet1

# OKLink API setup
API_BASE_URL = "https://www.oklink.com/api/v5/explorer"
API_KEY = os.getenv("OKLINK_API_KEY")
CHAIN_SHORT_NAME = "TRON"

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Global variables
SEQUENCE_LENGTH = 30
sequence = []
lstm_model = None

# Flask app setup
app = Flask(__name__)

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

# Extract last numerical digit from block hash
def extract_last_numerical_digit(block_hash):
    for char in reversed(block_hash):
        if char.isdigit():
            return int(char)
    return None

# Log data to Google Sheets
def log_to_google_sheet(timestamp, block_height, block_hash, last_digit):
    try:
        sheet.append_row([timestamp, block_height, block_hash, last_digit])
        logging.info(f"Logged to Google Sheets: {last_digit}")
    except Exception as e:
        logging.error(f"Failed to log to Google Sheets: {e}")

# Load LSTM model
def load_lstm_model():
    global lstm_model
    if lstm_model is None:
        lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(SEQUENCE_LENGTH, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logging.info("LSTM model initialized.")

# Update LSTM sequence and predict
def process_prediction(last_digit):
    global sequence, lstm_model
    sequence.append(last_digit)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)
    padded_sequence = [0] * (SEQUENCE_LENGTH - len(sequence)) + sequence
    input_data = np.array(padded_sequence).reshape(1, SEQUENCE_LENGTH, 1)
    probabilities = lstm_model.predict(input_data)
    predicted_digit = np.argmax(probabilities)
    return predicted_digit, probabilities[0][predicted_digit] * 100

# Log and train workflow
def log_and_train():
    while True:
        try:
            target_time, target_timestamp_ms = calculate_target_timestamp()
            while datetime.now(timezone.utc) < target_time:
                time.sleep(0.1)

            block_height, block_time = get_block_height_by_time(target_timestamp_ms)
            if block_height:
                block_hash = get_block_hash_by_height(block_height)
                if block_hash:
                    last_digit = extract_last_numerical_digit(block_hash)
                    if last_digit is not None:
                        log_to_google_sheet(datetime.now().isoformat(), block_height, block_hash, last_digit)
                        predicted_digit, confidence = process_prediction(last_digit)
                        logging.info(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}%")
        except Exception as e:
            logging.error(f"Error in log_and_train: {e}")

# Flask API: Get latest prediction
@app.route('/get-latest-prediction', methods=['GET'])
def get_latest_prediction():
    if sequence:
        predicted_digit, confidence = process_prediction(sequence[-1])
        return jsonify({"predicted_digit": predicted_digit, "confidence": confidence})
    else:
        return jsonify({"error": "No data available for prediction"}), 500

# Main function
def main():
    logging.info("Starting the script...")
    load_lstm_model()
    background_thread = threading.Thread(target=log_and_train, daemon=True)
    background_thread.start()
    app.run(debug=True)

# Entry point
if __name__ == "__main__":
    main()