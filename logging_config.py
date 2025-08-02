# logging_config.py
import logging
import os
from datetime import datetime, timedelta
import re

# --- CONFIGURATION ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log filename: logs/transcription-YYYY-MM-DD.log
today_str = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"transcription-{today_str}.log")

# Log retention period (in days)
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "7"))

# --- CLEANUP FUNCTION ---
def clean_old_log_files(log_dir="logs", days=7, prefix="transcription-"):
    """
    Deletes log files older than 'days' days from 'log_dir'.
    Only files with the given prefix and format 'prefix-YYYY-MM-DD.log' are considered.
    """
    now = datetime.now()
    threshold_date = now - timedelta(days=days)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{4}}-\d{{2}}-\d{{2}})\.log$")

    for filename in os.listdir(log_dir):
        match = pattern.match(filename)
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < threshold_date:
                    os.remove(os.path.join(log_dir, filename))
                    print(f"Deleted old log: {filename}")
            except ValueError:
                continue  # Skip malformed date formats


# --- CLEANUP OLD LOGS ON STARTUP ---
try:
    clean_old_log_files(log_dir=LOG_DIR, days=LOG_RETENTION_DAYS)
except Exception as e:
    print(f"Warning: Failed to clean old logs: {e}")


# --- LOGGER SETUP ---
logger = logging.getLogger("transcription_logger")
logger.setLevel(logging.DEBUG) # Set the base level for the logger to DEBUG

if not logger.handlers:
    # Formatter for file logs (no jobId, fileUrl)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO) # Only INFO and higher go to file
    logger.addHandler(file_handler)

    # Formatter for stream logs (no jobId, fileUrl)
    stream_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.DEBUG) # All DEBUG and higher go to stream
    logger.addHandler(stream_handler)
