# Imports for core functionality
from time import sleep
import os
import json
import requests
from datetime import datetime, timezone, timedelta
import time
import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

import googleapiclient.discovery
import logging
import joblib  # For saving and loading the scaler

logging.getLogger('googleapiclient').setLevel(logging.DEBUG)

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
from googleapiclient.discovery import build

# Flask imports for creating the API
from flask import Flask, jsonify

# Threading lock for shared state
from threading import Lock
state_lock = Lock()

# Shared state dictionary for thread-safe data sharing
shared_state = {}


# Suppress TensorFlow CPU logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Google Sheets setup
try:
    # Load credentials from the JSON file
    credentials_file = "club-code-442100-85744d72b31e.json"  # Path to your JSON file
    credentials = Credentials.from_service_account_file(
        credentials_file,
        scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    )
    client = gspread.authorize(credentials)
except FileNotFoundError:
    raise ValueError("The credentials.json file was not found. Ensure it's in the correct directory.")
except Exception as e:
    raise ValueError(f"Error loading Google Sheets credentials: {str(e)}")

# Set default Google Sheet name
sheet_name = "91club-api"  # Update with your sheet name
sheet = client.open(sheet_name).sheet1

# Define send data to sheet
def send_data_to_google_sheets(data):
    """
    Send data to Google Sheets.

    :param data: A list containing [timestamp (in IST), block_height, last_digit].
    """
    try:
        # Append the data to the Google Sheet
        sheet.append_row(data)
        logging.info(f"Data sent to Google Sheets: {data}")
    except Exception as e:
        logging.error(f"Failed to send data to Google Sheets: {e}")

# OKLink API setup
API_BASE_URL = "https://www.oklink.com/api/v5/explorer"
API_KEY = "c8f46c6a-11f6-4d1a-bb23-cfa0f55dfa73"
CHAIN_SHORT_NAME = "TRON"

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        logging.error("Model file not found. Train the model before making predictions.")

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

    # Validate labels
    if labels.min() < 0 or labels.max() >= 10:
        raise ValueError(f"Invalid label range detected. Min: {labels.min()}, Max: {labels.max()}. Expected: [0, 9].")

    return sequences, labels
    
# Train or retrain the LSTM model
def train_lstm_model():
    global lstm_model, scaler
    try:
        # Fetch and preprocess data
        sequences, labels = fetch_and_preprocess_data()

        # Log label range for debugging
        logging.info(f"Label range before training: Min={labels.min()}, Max={labels.max()}")

        # Check if there is enough data to train
        if len(sequences) == 0 or len(labels) == 0:
            logging.warning("Not enough data to train the model.")
            return

        # Initialize and fit the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        sequences_scaled = scaler.fit_transform(sequences)

        # Save the scaler for consistency
        scaler_file = "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info("Scaler initialized, fitted, and saved.")

        # Recompile the model and recreate the optimizer
        lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        logging.info("Training the LSTM model...")
        lstm_model.fit(sequences_scaled, labels, epochs=5, batch_size=32, verbose=2)

        # Save the updated model
        lstm_model.save(MODEL_PATH)
        logging.info("LSTM model retrained and saved.")
    except Exception as e:
        logging.error(f"Error in train_lstm_model: {str(e)}")

# scaler startup before first prediction 
def initialize_model_and_scaler():
    """
    Initialize the LSTM model and scaler at startup.
    """
    global lstm_model, scaler

    # Load or create the LSTM model
    load_or_create_model()

    # Load the scaler
    scaler_file = "scaler.pkl"
    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        logging.info("Scaler loaded successfully.")
    else:
        logging.warning("Scaler file not found. Model might need retraining.")
        scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize a default scaler

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

        # Log last digit to Google Sheets
        sheet.append_row([datetime.now().isoformat(), block_height, last_digit])

        logging.info(f"Block height: {block_height}, Block hash: {block_hash}, Last digit: {last_digit}")

        # Predict the next digit using the LSTM model
        predicted_digit, confidence = predict_next_digit()

        # Log the prediction
        logging.info(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.2%}")

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


# Background task for periodic updates
def background_task():
    while True:
        try:
            fetch_and_log_block_data()
        except Exception as e:
            logging.error(f"Error in background task: {str(e)}")
        time.sleep(10)  # Adjust interval as needed


# Predict the next digit and log the probability
def predict_next_digit():
    """
    Predict the next digit using the LSTM model and log the probability.
    """
    try:
        # Fetch data from Google Sheets
        data = pd.DataFrame(sheet.get_all_records())

        # Check for sufficient data
        if len(data) < SEQUENCE_LENGTH:
            logging.warning(f"Not enough data to make predictions. Require {SEQUENCE_LENGTH}, but found {len(data)}.")
            return None, None

        # Prepare the last sequence
        last_sequence = data['Last Digit'].iloc[-SEQUENCE_LENGTH:].values
        if len(last_sequence) != SEQUENCE_LENGTH:
            logging.error(f"Expected sequence length: {SEQUENCE_LENGTH}, Got: {len(last_sequence)}")
            return None, None

        # Ensure scaler is properly initialized
        if scaler is None or not hasattr(scaler, 'min_'):
            logging.error("Scaler is not properly initialized or fitted. Retrain the model first.")
            return None, None

        # Normalize the sequence
        try:
            normalized_sequence = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, SEQUENCE_LENGTH, 1)
        except Exception as e:
            logging.error(f"Scaler transformation error: {e}")
            return None, None

        # Ensure the LSTM model is loaded
        if lstm_model is None:
            logging.error("LSTM model is not initialized. Load or train the model before prediction.")
            return None, None

        # Predict using the LSTM model
        probabilities = lstm_model.predict(normalized_sequence, verbose=0)
        if probabilities is None or len(probabilities) == 0:
            logging.error("Prediction failed: Empty or None probabilities.")
            return None, None

        # Extract the predicted digit and confidence
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[0][predicted_digit]

        return predicted_digit, confidence
    except Exception as e:
        logging.error(f"Error in predict_next_digit: {str(e)}")
        return None, None
        
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
    url = f"{API_BASE_URL}/block/block-fills"
    response = requests.get(url, headers=headers, params=params)
    response_data = response.json()

    if response.status_code == 200 and response_data["code"] == "0":
        block_data = response_data.get("data", [])
        if block_data:
            return block_data[0]["hash"]
    return None

# extract last digit 
def extract_last_numerical_digit(block_hash):
    """
    Extract the last numerical digit from the block hash.
    """
    for char in reversed(block_hash):
        if char.isdigit():
            return int(char)  # Convert to an integer
    return None

# Flask route: Home
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the LSTM prediction API"})

# Route to fetch block data and prediction
@app.route('/get_block', methods=['GET'])
def get_block():
    try:
        # Log access to the endpoint
        logging.info(f"/get_block accessed. Current shared state: {shared_state}")

        # Respond with current block data and prediction
        response = {
            "block_height": shared_state.get("current_block_height", "N/A"),
            "last_digit": shared_state.get("current_last_digit", "N/A"),
            "predicted_next_digit": shared_state.get("latest_prediction", {}).get("digit", "N/A"),
            "confidence": shared_state.get("latest_prediction", {}).get("confidence", "N/A"),
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in /get_block: {str(e)}")
        return jsonify({"error": str(e)}), 500
                          
# Ensure `periodic_task` is defined before adding it to the scheduler
def periodic_task():
    try:
        # Calculate the target timestamp
        target_time, target_timestamp_ms = calculate_target_timestamp()
        logging.info(f"Target Timestamp (54th second): {target_time}, {target_timestamp_ms} ms")

        # Wait until the 54th second
        while datetime.now(timezone.utc) < target_time:
            time.sleep(0.1)  # Check every 100ms to minimize CPU usage

        logging.info("54th second reached. Introducing 8-second delay...")
        time.sleep(8)  # Ensure the block is created and indexed

        # Fetch block height for the target timestamp
        block_height, block_time_utc = get_block_height_by_time(target_timestamp_ms)
        if block_height and block_time_utc:
            # Convert block time from UTC to IST
            block_time_ist = block_time_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))

            if block_time_ist.second == 54:  # Ensure it's the 54th second
                logging.info(f"Block Height at {block_time_ist} (IST): {block_height}")

                # Fetch block hash
                block_hash = get_block_hash_by_height(block_height)
                if block_hash:
                    logging.info(f"Block Hash for Height {block_height}: {block_hash}")

                    # Extract last numerical digit
                    last_digit = extract_last_numerical_digit(block_hash)
                    if last_digit is not None:
                        logging.info(f"Last Numerical Digit in Block Hash: {last_digit}")

                        # Prepare data to send to Google Sheets
                        data_to_send = [str(block_time_ist), block_height, last_digit]

                        # Send data to Google Sheets
                        try:
                            send_data_to_google_sheets(data_to_send)
                            logging.info("Data sent to Google Sheets successfully.")
                        except Exception as e:
                            logging.error(f"Failed to send data to Google Sheets: {e}")

                        # Predict next digit
                        try:
                            logging.info("Predicting next digit...")
                            predicted_digit, confidence = predict_next_digit()
                            if predicted_digit is not None and confidence is not None:
                                logging.info(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2%}")
                            else:
                                logging.error("Prediction failed: No valid predicted digit or confidence.")
                        except Exception as e:
                            logging.error(f"Error predicting next digit: {e}")
    
                        # Retrain the model periodically
                        try:
                            logging.info("Retraining LSTM model with updated data...")
                            train_lstm_model()
                        except Exception as e:
                            logging.error(f"Error retraining LSTM model: {e}")
                    else:
                        logging.error("No numerical digit found in block hash.")
                else:
                    logging.error("Failed to fetch block hash.")
            else:
                logging.error(f"Block timestamp mismatch. Expected: 54s, Found: {block_time_ist.second}s")
        else:
            logging.error("Failed to fetch block height.")
    except Exception as e:
        logging.error(f"Error during periodic task: {e}")
        
# Entry point
if __name__ == "__main__":
    initialize_model_and_scaler()

    # Initialize the scheduler
    scheduler = BackgroundScheduler()

    # Schedule the periodic task
    scheduler.add_job(periodic_task, 'cron', second=54)  # Run every 54th second
    scheduler.add_job(train_lstm_model, 'interval', minutes=10)  # Train model every 10 minutes

    # Import event listener from apscheduler
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

    # Define the job listener
    def job_listener(event):
        if event.exception:
            logging.error(f"Job {event.job_id} failed with exception: {event.exception}")
        else:
            logging.info(f"Job {event.job_id} executed successfully.")

    # Add the job listener to the scheduler
    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    # Start the scheduler
    scheduler.start()

    # Run the Flask app (this will continue running while the scheduler works in the background)
    try:
        app.run(debug=True, use_reloader=False)  # Prevent Flask from reloading the scheduler
    except KeyboardInterrupt:
        logging.info("Shutting down the Flask app and scheduler.")
        scheduler.shutdown()
   