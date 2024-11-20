from flask import Flask, request, jsonify
import os
import requests
import gspread
from google.oauth2.service_account import Credentials
from apscheduler.schedulers.background import BackgroundScheduler
from collections import defaultdict
import pandas as pd
from datetime import datetime
import traceback
from dateutil import parser

app = Flask(__name__)

# Constants (from environment variables)
API_URL = os.getenv("API_URL", "https://www.oklink.com/api/v5/explorer/block/block-fills")
API_KEY = os.getenv("API_KEY")
CHAIN_SHORT_NAME = os.getenv("CHAIN_SHORT_NAME", "TRON")
SHEET_NAME = os.getenv("SHEET_NAME", "91club-api")
JSON_URL = os.getenv("JSON_URL")
JSON_FILE_NAME = "credentials.json"

# Download credentials file
response = requests.get(JSON_URL)
if response.status_code == 200:
    with open(JSON_FILE_NAME, "wb") as file:
        file.write(response.content)

# Authenticate Google Sheets
credentials = Credentials.from_service_account_file(
    JSON_FILE_NAME,
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
client = gspread.authorize(credentials)
sheet = client.open(SHEET_NAME).sheet1

# Global transition matrix
transitions = defaultdict(lambda: defaultdict(int))

# Fetch block data
def fetch_block_data(block_height):
    headers = {"Ok-Access-Key": API_KEY}
    params = {"chainShortName": CHAIN_SHORT_NAME, "height": block_height}
    response = requests.get(API_URL, headers=headers, params=params)
    data = response.json()
    if data.get("code") == "0" and data.get("data"):
        return data["data"][0]
    return None

# Get last numerical digit from block hash
def get_last_numerical_digit(block_hash):
    for char in reversed(block_hash):
        if char.isdigit():
            return int(char)
    return None

# Fetch and write data
def fetch_and_write_data():
    now = datetime.utcnow()
    block_height = int(now.timestamp() // 60) * 20
    block_data = fetch_block_data(block_height)
    if block_data:
        block_hash = block_data.get("hash")
        last_digit = get_last_numerical_digit(block_hash)
        row = [block_height, last_digit]
        sheet.append_row(row)

# Update transitions
def update_transitions():
    data = pd.DataFrame(sheet.get_all_records())
    if 'Last Numerical Digit' not in data.columns:
        return
    data['Last Numerical Digit'] = pd.to_numeric(data['Last Numerical Digit'], errors='coerce').fillna(0).astype(int)
    data['Category'] = data['Last Numerical Digit'].apply(lambda x: 'Small' if x <= 4 else 'Big')
    for i in range(len(data) - 1):
        current_category = data['Category'].iloc[i]
        next_category = data['Category'].iloc[i + 1]
        transitions[current_category][next_category] += 1

# Normalize transitions
def normalize_transitions():
    transition_matrix = {}
    for current, next_states in transitions.items():
        total = sum(next_states.values())
        transition_matrix[current] = {state: (count / total) * 100 for state, count in next_states.items()}
    return transition_matrix

# Predict next digit
def predict_next_digit():
    data = pd.DataFrame(sheet.get_all_records())
    if 'Last Numerical Digit' not in data.columns or data.empty:
        app.logger.error("Data missing or empty")
        return {"error": "Data missing or empty"}
    
    last_digit = int(data['Last Numerical Digit'].iloc[-1])
    category = 'Small' if last_digit <= 4 else 'Big'
    transition_matrix = normalize_transitions()
    if category not in transition_matrix:
        app.logger.error(f"No data for category '{category}'")
        return {"error": f"No data for category '{category}'"}
    
    return transition_matrix[category], last_digit

# Flask API endpoint
@app.route('/')
def home():
    return "Welcome to the Flask App!"
    
@app.route('/predict', methods=['GET'])
def get_prediction():
    timestamp = request.args.get('timestamp')
    try:
        if timestamp:
            # Attempt to parse the timestamp from the query parameter
            timestamp = parser.parse(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        # Call the prediction function
        predicted, last_digit = predict_next_digit()
        
        return jsonify({
            "timestamp": timestamp,
            "last_digit": last_digit,
            "predictions": predicted
        })
    except Exception as e:
        # Log the exception and stack trace
        app.logger.error(f"Error occurred: {str(e)}")
        traceback.print_exc()  # Prints the stack trace to the logs
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        })

# Scheduler to fetch data every minute at the 54th second
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_and_write_data, 'cron', second=54)
scheduler.start()

# Running the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses the PORT environment variable
    app.run(host="0.0.0.0", port=port)