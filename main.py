# Imports for core functionality
from time import sleep
import os
import schedule
import json
import requests
from datetime import datetime, timezone, timedelta
import time
import logging
import threading

# Imports for handling data processing
import numpy as np
import pandas as pd

# Imports for ML
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Imports for Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# Environment variable management
from dotenv import load_dotenv

# Flask imports for creating the API
from flask import Flask, jsonify

# Threading lock for shared state
from threading import Lock
state_lock = Lock()

# Shared state dictionary for thread-safe data sharing
shared_state = {}


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

# Global variables for model training
SEQUENCE_LENGTH = 30  # Sequence length for LSTM input
MODEL_PATH = "model.h5"  # Path to save the LSTM model
scaler = MinMaxScaler()  # Scaler for normalization
lstm_model = None  # The LSTM model

# Ensure the model is loaded or created at startup
def load_or_create_model():
    global lstm_model
    if os.path.exists(MODEL_PATH):
        lstm_model = load_model(MODEL_PATH)
        logging.info("Loaded existing LSTM model.")
    else:
        lstm_model = create_lstm_model(input_shape=(SEQUENCE_LENGTH, 1))
        logging.info("Created a new LSTM model.")

# Function to create a new LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))  # Output layer for 10 digits (0-9)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess data from Google Sheets
def fetch_and_preprocess_data():
    data = pd.DataFrame(sheet.get_all_records())
    if 'Last Digit' not in data.columns:
        raise ValueError("The column 'Last Digit' is missing in the Google Sheet.")
    data['Last Digit'] = pd.to_numeric(data['Last Digit'], errors='coerce').fillna(0).astype(int)
    
    # Prepare sequences for LSTM
    sequences = []
    labels = []
    for i in range(len(data) - SEQUENCE_LENGTH):
        sequences.append(data['Last Digit'].iloc[i:i + SEQUENCE_LENGTH].values)
        labels.append(data['Last Digit'].iloc[i + SEQUENCE_LENGTH])
    
    # Normalize sequences
    sequences = scaler.fit_transform(np.array(sequences).reshape(-1, 1)).reshape(-1, SEQUENCE_LENGTH, 1)
    labels = np.array(labels)
    return sequences, labels

# Train or retrain the LSTM model
def train_lstm_model():
    global lstm_model
    sequences, labels = fetch_and_preprocess_data()
    if len(sequences) == 0:
        logging.warning("Not enough data to train the model.")
        return
    lstm_model.fit(sequences, labels, epochs=5, batch_size=32, verbose=2)
    lstm_model.save(MODEL_PATH)
    logging.info("LSTM model trained and saved.")

# Function to fetch block data and log to Google Sheets
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
        sheet.append_row([datetime.now().isoformat(), block_height, last_digit])

        logging.info(f"Block height: {block_height}, Block hash: {block_hash}, Last digit: {last_digit}")

        # Predict the next digit using the LSTM model
        predicted_digit, confidence = predict_next_digit()  # Ensure predict_next_digit returns both digit and confidence

        # Update shared state (thread-safe)
        with state_lock:
            shared_state["current_block_height"] = block_height
            shared_state["current_last_digit"] = last_digit
            shared_state["latest_prediction"] = {
                "digit": predicted_digit,
                "confidence": f"{confidence * 100:.2f}%",
            }

    except Exception as e:
        logging.error(f"Error in fetch_and_log_block_data: {str(e)}")

# Predict the next digit and log the probability
def predict_next_digit():
    data = pd.DataFrame(sheet.get_all_records())
    if len(data) < SEQUENCE_LENGTH:
        logging.warning("Not enough data to make predictions.")
        return
    last_sequence = data['Last Digit'].iloc[-SEQUENCE_LENGTH:].values
    normalized_sequence = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, SEQUENCE_LENGTH, 1)
    probabilities = lstm_model.predict(normalized_sequence)
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[0][predicted_digit]
    logging.info(f"Predicted digit: {predicted_digit} with confidence {confidence:.2%}")

# Function to calculate the target timestamp
def calculate_target_timestamp():
    now = datetime.now(timezone.utc)
    target_time = now.replace(second=54, microsecond=0)
    if now.second >= 54:
        target_time += timedelta(minutes=1)
    return target_time, int(target_time.timestamp() * 1000)

# Function to fetch block height by time
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
    return None, None

# Function to fetch block hash by height
def get_block_hash_by_height(block_height):
    params = {"chainShortName": CHAIN_SHORT_NAME, "height": block_height}
    headers = {"Ok-Access-Key": API_KEY}
    url = f"{API_BASE_URL}/block/block-hash"
    response = requests.get(url, headers=headers, params=params)
    response_data = response.json()

    if response.status_code == 200 and response_data["code"] == "0":
        block_data = response_data.get("data", [])
        if block_data:
            return block_data[0]["hash"]
    return None

# Flask route: Home
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the LSTM prediction API"})

# Start background tasks
def start_cycle():
    schedule.every().minute.at(":54").do(fetch_and_log_block_data)
    schedule.every(10).minutes.do(train_lstm_model)
    while True:
        schedule.run_pending()
        sleep(1)

# Main function
if __name__ == "__main__":
    load_or_create_model()
    background_thread = threading.Thread(target=start_cycle, daemon=True)
    background_thread.start()
    app.run(debug=True)

# Route to fetch block data and prediction
@app.route('/get_block', methods=['GET'])
def get_block():
    # Assuming `latest_probability` stores the latest prediction globally
    try:
        # Respond with current block data and prediction
        response = {
            "block_height": shared_state.get("current_block_height", "N/A"),
            "last_digit": shared_state.get("current_last_digit", "N/A"),
            "predicted_next_digit": shared_state.get("latest_prediction", {}).get("digit", "N/A"),
            "confidence": shared_state.get("latest_prediction", {}).get("confidence", "N/A"),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
                
# Entry point
if __name__ == "__main__":
    main()