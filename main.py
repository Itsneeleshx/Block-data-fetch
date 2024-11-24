from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
import gspread
import logging

# Initialize Flask app
app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Google Sheets setup
try:
    credentials_file = "club-code-442100-85744d72b31e.json"  # Update with your JSON file path
    credentials = gspread.service_account(filename=credentials_file)
    sheet_name = "91club-api"  # Replace with your Google Sheet name
    sheet = credentials.open(sheet_name).sheet1
    logging.info("Google Sheets connection established successfully.")
except Exception as e:
    logging.error(f"Failed to connect to Google Sheets: {e}")
    sheet = None

# Function to fetch and log block data (stub example)
def fetch_and_log_block_data():
    try:
        # Sample data to log in the sheet
        data = {
            "timestamp": "2024-11-24 15:30:00",  # Replace with actual data
            "block_height": 12345678,           # Replace with actual data
            "hash": "0xabc123",                # Replace with actual data
        }
        logging.info(f"Fetched block data: {data}")

        if sheet:
            # Write data to the Google Sheet
            sheet.append_row([data["timestamp"], data["block_height"], data["hash"]])
            logging.info("Data appended to Google Sheet successfully.")
        else:
            logging.error("Google Sheet object is None. Data not logged.")
    except Exception as e:
        logging.error(f"Error in fetch_and_log_block_data: {e}")

# Load or create the LSTM model (stub function)
def load_or_create_model():
    logging.info("Model loaded or created successfully.")

# Function to train LSTM model (stub example)
def train_lstm_model():
    logging.info("LSTM model trained successfully.")

# Define job listener for APScheduler
def job_listener(event):
    if event.exception:
        logging.error(f"Job {event.job_id} failed with exception: {event.exception}")
    else:
        logging.info(f"Job {event.job_id} executed successfully.")

if __name__ == "__main__":
    # Load or create the LSTM model
    load_or_create_model()

    # Initialize the scheduler
    scheduler = BackgroundScheduler()

    # Add jobs
    scheduler.add_job(fetch_and_log_block_data, 'cron', second=54)  # Fetch and log data every minute at second 54
    scheduler.add_job(train_lstm_model, 'interval', minutes=10)     # Train model every 10 minutes

    # Add listener to log job status
    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    # Start the scheduler
    scheduler.start()

    # Run the Flask app
    app.run(debug=True)