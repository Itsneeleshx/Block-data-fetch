# Imports for core functionality
import os
import time
import logging
import json
import requests
from datetime import datetime, timezone, timedelta
import threading

# Google Sheets imports
import gspread
from google.oauth2.service_account import Credentials

# Machine learning imports
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Validate environment variables
required_vars = ["GOOGLE_CREDENTIALS", "OKLINK_API_KEY", "GOOGLE_SHEET_NAME"]
for var in required_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Environment variable '{var}' is not set.")

# Google Sheets setup
credentials_info = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
credentials = Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
)
client = gspread.authorize(credentials)
sheet = client.open(os.getenv("GOOGLE_SHEET_NAME")).sheet1

# OKLink API setup
API_BASE_URL = "https://www.oklink.com/api/v5/explorer"
API_KEY = os.getenv("OKLINK_API_KEY")
CHAIN_SHORT_NAME = "TRON"

# Global variables
SEQUENCE_LENGTH = 30
lstm_model = None
sequence = []
latest_prediction = None


# Function: Load or create the LSTM model
def initialize_lstm_model(input_shape=(SEQUENCE_LENGTH, 1)):
    global lstm_model
    if os.path.exists("model.h5"):
        lstm_model = load_model("model.h5")
        logging.info("LSTM model loaded from disk.")
    else:
        lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        lstm_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logging.info("New LSTM model created.")


# Function: Log data to Google Sheets
def log_to_google_sheet(block_height, block_hash, last_digit):
    try:
        timestamp = datetime.utcnow().isoformat()
        sheet.append_row([timestamp, block_height, block_hash, last_digit])
        logging.info(f"Logged data to Google Sheets: {block_height}, {block_hash}, {last_digit}")
    except Exception as e:
        logging.error(f"Failed to log data to Google Sheets: {str(e)}")


# Function: Fetch data from OKLink API
def fetch_block_data():
    target_time, target_timestamp_ms = calculate_target_timestamp()
    params = {"chainShortName": CHAIN_SHORT_NAME, "time": target_timestamp_ms}
    headers = {"Ok-Access-Key": API_KEY}

    # Fetch block height
    block_height_response = requests.get(f"{API_BASE_URL}/block/block-height-by-time", headers=headers, params=params)
    if block_height_response.status_code != 200:
        logging.error(f"Failed to fetch block height: {block_height_response.text}")
        return None, None, None

    block_height_data = block_height_response.json().get("data", [])
    if not block_height_data:
        logging.error("No block height data available.")
        return None, None, None

    block_height = block_height_data[0]["height"]

    # Fetch block hash
    block_hash_response = requests.get(f"{API_BASE_URL}/block/block-fills", headers=headers, params={"height": block_height})
    if block_hash_response.status_code != 200:
        logging.error(f"Failed to fetch block hash: {block_hash_response.text}")
        return None, None, None

    block_hash_data = block_hash_response.json().get("data", [])
    if not block_hash_data:
        logging.error("No block hash data available.")
        return None, None, None

    block_hash = block_hash_data[0]["hash"]
    last_digit = int(block_hash[-1], 16) if block_hash[-1].isdigit() else 0

    return block_height, block_hash, last_digit


# Function: Preprocess data for LSTM
def preprocess_data():
    records = sheet.get_all_records()
    if not records:
        return None, None

    data = [int(row["Last Digit"]) for row in records if "Last Digit" in row]
    if len(data) < SEQUENCE_LENGTH:
        return None, None

    sequences = []
    labels = []
    for i in range(len(data) - SEQUENCE_LENGTH):
        sequences.append(data[i:i + SEQUENCE_LENGTH])
        labels.append(data[i + SEQUENCE_LENGTH])

    sequences = np.array(sequences).reshape(-1, SEQUENCE_LENGTH, 1)
    labels = np.array(labels)

    return sequences, labels


# Function: Train LSTM model
def train_lstm_model():
    sequences, labels = preprocess_data()
    if sequences is None or labels is None:
        logging.warning("Not enough data for training.")
        return

    lstm_model.fit(sequences, labels, epochs=1, batch_size=32, verbose=1)
    lstm_model.save("model.h5")
    logging.info("LSTM model trained and saved.")


# Function: Predict the next digit
def predict_next_digit():
    global latest_prediction
    sequences, _ = preprocess_data()
    if sequences is None:
        logging.warning("Not enough data for prediction.")
        return

    input_sequence = sequences[-1].reshape(1, SEQUENCE_LENGTH, 1)
    probabilities = lstm_model.predict(input_sequence)
    predicted_digit = np.argmax(probabilities)
    prediction_percentage = probabilities[0][predicted_digit] * 100

    latest_prediction = {"digit": predicted_digit, "probability": prediction_percentage}
    logging.info(f"Predicted next digit: {predicted_digit} ({prediction_percentage:.2f}%)")


# Function: Calculate target timestamp
def calculate_target_timestamp():
    now = datetime.now(timezone.utc)
    target_time = now.replace(second=54, microsecond=0)
    if now.second >= 54:
        target_time += timedelta(minutes=1)
    return target_time, int(target_time.timestamp() * 1000)


# Main function
def main():
    initialize_lstm_model()

    def log_and_train():
        """Handles data fetching, logging to Google Sheets, and training the LSTM model."""
        while True:
            try:
                # Fetch block data at the current time
                block_height, block_hash, last_digit = fetch_block_data()
                if block_height and block_hash and last_digit is not None:
                    # Log data to Google Sheets
                    log_to_google_sheet(block_height, block_hash, last_digit)

                    # Train the LSTM model
                    train_lstm_model()

                # Sleep for the remaining time of the current minute
                current_time = datetime.now()
                next_execution = current_time.replace(second=54, microsecond=0) + timedelta(minutes=1)
                sleep_time = (next_execution - current_time).total_seconds()
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Error in log_and_train: {str(e)}")
    
    def prediction_loop():
        """Continuously predicts the next digit before the 54th second of each minute."""
        while True:
            try:
                # Predict next digit
                predict_next_digit()

                # Wait until just before the 54th second of the next minute
                current_time = datetime.now()
                next_execution = current_time.replace(second=53, microsecond=500000) + timedelta(minutes=1)
                sleep_time = (next_execution - current_time).total_seconds()
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Error in prediction_loop: {str(e)}")
    
    # Launch both threads simultaneously
    try:
        logging.info("Starting the script...")
        log_train_thread = threading.Thread(target=log_and_train, daemon=True)
        prediction_thread = threading.Thread(target=prediction_loop, daemon=True)

        log_train_thread.start()
        prediction_thread.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Script terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")


if __name__ == "__main__":
    main()