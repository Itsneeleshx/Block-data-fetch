# Imports for core functionality
from time import sleep
import pickle
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

import logging
import joblib  # For saving and loading the scaler

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
    Send data to Google Sheets without checking for duplicates.

    :param data: A list containing [timestamp (in IST), block_height, last_digit, predicted_digit, confidence, predicted_pattern].
    """
    try:
        # Send the data (including Timestamp and Last Digit)
        response = sheet.append_row(data)
        if response.status_code == 200:
            logging.info(f"Data sent to Google Sheets successfully: {data}")
        else:
            logging.error(f"Failed to send data to Google Sheets with status code: {response.status_code}")

    except Exception as e:
        logging.error(f"Failed to send data to Google Sheets: {str(e)}")

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

# Define multi-task LSTM model
def create_multitask_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer for digit prediction
    digit_output = Dense(10, activation='softmax', name='digit_output')(model.output)
    
    # Output layer for Big/Small classification
    big_small_output = Dense(2, activation='softmax', name='big_small_output')(model.output)
    
    # Combine the model
    multitask_model = Model(inputs=model.input, outputs=[digit_output, big_small_output])
    multitask_model.compile(
        optimizer='adam',
        loss={'digit_output': 'sparse_categorical_crossentropy', 'big_small_output': 'categorical_crossentropy'},
        metrics={'digit_output': 'accuracy', 'big_small_output': 'accuracy'}
    )
    
    return multitask_model

# Function to preprocess data from Google Sheets
def fetch_and_preprocess_data():
    try:
        # Load data into a DataFrame
        data = pd.DataFrame(sheet.get_all_records())

        # Validate columns
        if 'Timestamp' not in data.columns or 'Block Height' not in data.columns or 'Last Digit' not in data.columns:
            raise ValueError("The required columns ['Timestamp', 'Block Height', 'Last Digit'] are missing in the Google Sheet.")

        # Drop duplicates based on 'Timestamp' and 'Block Height'
        data = data.drop_duplicates(subset=['Timestamp', 'Block Height'], keep='first')
        logging.info(f"Data after removing duplicates: {len(data)} rows")

        # Ensure 'Last Digit' is numeric
        data['Last Digit'] = pd.to_numeric(data['Last Digit'], errors='coerce').fillna(0).astype(int)

        # Check if there is enough data
        if len(data) < SEQUENCE_LENGTH:
            raise ValueError(f"Not enough data to create sequences. Required: {SEQUENCE_LENGTH}, Found: {len(data)}")

        # Prepare sequences and labels for LSTM
        sequences = []
        labels = []
        for i in range(len(data) - SEQUENCE_LENGTH):
            sequences.append(data['Last Digit'].iloc[i:i + SEQUENCE_LENGTH].values)
            labels.append(data['Last Digit'].iloc[i + SEQUENCE_LENGTH])

        # Normalize sequences
        sequences = scaler.fit_transform(np.array(sequences).reshape(-1, 1)).reshape(-1, SEQUENCE_LENGTH, 1)
        labels = np.array(labels)

        # Validate label range
        if labels.max() >= 10:
            raise ValueError(f"Invalid label range detected. Max: {labels.max()}. Expected: [0, 9].")

        return sequences, labels

    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise
    
# Train or retrain the LSTM model
def train_lstm_model():
    global lstm_model, scaler
    try:
        # Fetch and preprocess data
        sequences, labels = fetch_and_preprocess_data()
        
        sequences = []
        digit_labels = []
        big_small_labels = []
        
        # Generate sequences and labels for multitask learning
        for i in range(len(data) - SEQUENCE_LENGTH):
            sequences.append(data['Last Digit'].iloc[i:i + SEQUENCE_LENGTH].values)
            digit_labels.append(data['Last Digit'].iloc[i + SEQUENCE_LENGTH])
            big_small_labels.append(data['Big/Small'].iloc[i + SEQUENCE_LENGTH])

        # Normalize the sequences
        sequences = scaler.fit_transform(np.array(sequences).reshape(-1, 1)).reshape(-1, SEQUENCE_LENGTH, 1)
        digit_labels = np.array(digit_labels)
        big_small_labels = np.array(big_small_labels)

        # Log label range for debugging
        logging.info(f"Label ranges: Digit Labels Min={digit_labels.min()}, Max={digit_labels.max()}")

        # Check if there is enough data to train
        if len(sequences) == 0 or len(digit_labels) == 0 or len(big_small_labels) == 0:
            logging.warning("Not enough data to train the model.")
            return

        # Create or load the multitask model
        if lstm_model is None:
            lstm_model = create_multitask_lstm_model((SEQUENCE_LENGTH, 1))

        # Train the multitask model
        logging.info("Training the multitask LSTM model...")
        lstm_model.fit(
            sequences,
            {'digit_output': digit_labels, 'big_small_output': big_small_labels},
            epochs=5,
            batch_size=32,
            verbose=2
        )

        # Save the updated model
        lstm_model.save(MODEL_PATH)
        logging.info("Multitask LSTM model retrained and saved.")

        # Save the scaler for consistency
        scaler_file = "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info("Scaler saved for consistency.")
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

# Function to fetch block data, make predictions, and log to Google Sheets
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

        # Extract the last digit from the block hash
        last_digit = extract_last_numerical_digit(block_hash)
        if last_digit is None:
            logging.error("Failed to extract last digit from block hash.")
            return

        # Fetch data from Google Sheets to prepare input for prediction
        data = pd.DataFrame(sheet.get_all_records())
        if len(data) < SEQUENCE_LENGTH:
            logging.warning("Not enough data to make predictions.")
            return

        # Prepare the input data (last sequence of digits)
        input_data = np.array(data['Last Digit'].iloc[-SEQUENCE_LENGTH:].values)

        # Get prediction and pattern
        predicted_digit, confidence, predicted_pattern = predict_next_digit_with_pattern(input_data)

        if predicted_digit is not None:
            # Prepare the data row to append
            timestamp = datetime.now().isoformat()
            data_to_send = [
                timestamp, 
                block_height, 
                last_digit, 
                predicted_digit, 
                f"{confidence:.2%}", 
                predicted_pattern
            ]

            # Append prediction to Google Sheets
            sheet.append_row(data_to_send)

            logging.info(f"Appended prediction and pattern to Google Sheets: {data_to_send}")

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

# Prepare input data 
def prepare_input_data():
    # Fetch data from Google Sheets
    data = pd.DataFrame(sheet.get_all_records())

    # Check for sufficient data
    if len(data) < SEQUENCE_LENGTH:
        logging.warning("Not enough data to prepare input sequence.")
        return None

    # Extract the last sequence
    last_sequence = data['Last Digit'].iloc[-SEQUENCE_LENGTH:].values

    # Normalize the sequence using the scaler
    normalized_sequence = scaler.transform(last_sequence.reshape(-1, 1))

    # Reshape for LSTM input
    return normalized_sequence.reshape(1, SEQUENCE_LENGTH, 1)

# Predict the next digit and Big/Small pattern
def predict_next_digit_with_pattern(input_data):
    """
    Predict the next digit using the LSTM model and analyze patterns (Big/Small).

    :param input_data: A NumPy array of shape (SEQUENCE_LENGTH,) containing the last sequence of digits.
    :return: The predicted digit, confidence level, and Big/Small pattern.
    """
    try:
        # Ensure scaler is properly initialized
        if scaler is None or not hasattr(scaler, 'min_'):
            logging.error("Scaler is not properly initialized or fitted. Retrain the model first.")
            return None, None, None

        # Normalize the input sequence
        normalized_sequence = scaler.transform(input_data.reshape(-1, 1))
        normalized_sequence = normalized_sequence.reshape(1, SEQUENCE_LENGTH, 1)

        # Ensure the LSTM model is loaded
        if lstm_model is None:
            logging.error("LSTM model is not initialized. Load or train the model before prediction.")
            return None, None, None

        # Predict using the LSTM model
        probabilities = lstm_model.predict(normalized_sequence, verbose=0)
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[0][predicted_digit]

        # Analyze the Big/Small pattern for the input sequence
        big_small_pattern = ["Big" if digit >= 5 else "Small" for digit in input_data]
        predicted_pattern = "Big" if predicted_digit >= 5 else "Small"

        logging.info(f"Input Sequence Big/Small Pattern: {big_small_pattern}")
        logging.info(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}, Predicted Pattern: {predicted_pattern}")
        
        return predicted_digit, confidence, predicted_pattern

    except Exception as e:
        logging.error(f"Error in predict_next_digit_with_pattern: {str(e)}")
        return None, None, None
        
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

                        # Prepare input data for prediction
                        input_data = prepare_input_data()
                        if input_data is not None:
                            try:
                                logging.info("Predicting next digit...")
                                predicted_digit, confidence, predicted_pattern = predict_next_digit_with_pattern(input_data)
                                if predicted_digit is not None and confidence is not None and predicted_pattern is not None:
                                    logging.info(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2%}, Predicted Pattern: {predicted_pattern}")

                                    # Add predictions to data_to_send
                                    data_to_send.extend([predicted_digit, f"{confidence:.2%}", predicted_pattern])

                                    # Send data to Google Sheets
                                    try:
                                        send_data_to_google_sheets(data_to_send)
                                        logging.info("Data sent to Google Sheets successfully, including predictions.")
                                    except Exception as e:
                                        logging.error(f"Failed to send data to Google Sheets: {e}")
                                else:
                                    logging.error("Prediction failed: Missing one or more prediction values.")
                            except Exception as e:
                                logging.error(f"Error predicting next digit: {e}")
                        else:
                            logging.warning("Insufficient data to prepare input for prediction.")

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
    scheduler.add_job(periodic_task, 'interval', seconds=30, max_instances=1)
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
   