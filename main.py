import os
import json
import requests
from datetime import datetime, timezone, timedelta
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment variables from a .env file in local development
load_dotenv()

# OKLink API details
API_BASE_URL = "https://www.oklink.com/api/v5/explorer"
API_KEY = os.getenv("OKLINK_API_KEY")
CHAIN_SHORT_NAME = "TRON"


# Use environment variable for Google credentials
credentials = Credentials.from_service_account_info(
    json.loads(os.getenv("GOOGLE_CREDENTIALS")),
    scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
)

client = gspread.authorize(credentials)
sheet_name = os.getenv("GOOGLE_SHEET_NAME", "91club-api")  # Default value is optional.
sheet = client.open(sheet_name).sheet1

# Function to calculate the target timestamp
def calculate_target_timestamp():
    now = datetime.now(timezone.utc)
    target_time = now.replace(second=54, microsecond=0)
    if now.second >= 54:
        target_time += timedelta(minutes=1)
    return target_time, int(target_time.timestamp() * 1000)

# Fetch block height from OKLink
def get_block_height_by_time(target_timestamp_ms):
    delay = 8
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

# Fetch block hash by height
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

if __name__ == "__main__":
    main()