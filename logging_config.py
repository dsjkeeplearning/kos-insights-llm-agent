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


# --- LOGGER SETUP ---
logger = logging.getLogger("transcription_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        '%(asctime)s - jobId=%(jobId)s - fileUrl=%(fileUrl)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
