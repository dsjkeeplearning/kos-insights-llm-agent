# logging_config.py
import logging
import os
from datetime import datetime, timedelta
import re

# --- CONFIGURATION ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log filenames
today_str = datetime.now().strftime("%Y-%m-%d")
TRANSCRIPTION_LOG_FILE = os.path.join(LOG_DIR, f"transcription-{today_str}.log")
SCORE_LOG_FILE = os.path.join(LOG_DIR, f"score-{today_str}.log")
INACTIVE_DECAY_LOG_FILE = os.path.join(LOG_DIR, f"inactive-decay-{today_str}.log")

# Log retention period (in days)
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "7"))

# --- CLEANUP FUNCTION ---
def clean_old_log_files(log_dir="logs", days=7):
    """
    Deletes log files older than 'days' days from 'log_dir'.
    Only files with the format '*-YYYY-MM-DD.log' are considered.
    """
    now = datetime.now()
    threshold_date = now - timedelta(days=days)

    pattern = re.compile(r"^([\w-]+)-(\d{4}-\d{2}-\d{2})\.log$")

    for filename in os.listdir(log_dir):
        match = pattern.match(filename)
        if match:
            date_str = match.group(2)  # second group = YYYY-MM-DD
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < threshold_date:
                    os.remove(os.path.join(log_dir, filename))
                    print(f"Deleted old log: {filename}")
            except ValueError:
                continue  # Skip malformed date formats

# --- CLEANUP OLD LOGS ON STARTUP ---
try:
    # Clean all log files with the pattern *-YYYY-MM-DD.log
    clean_old_log_files(log_dir=LOG_DIR, days=LOG_RETENTION_DAYS)
except Exception as e:
    print(f"Warning: Failed to clean old logs: {e}")


# --- HELPER FUNCTION FOR LOGGER SETUP ---
def setup_logger(name, log_file, level=logging.DEBUG, file_level=logging.INFO):
    """
    Creates a logger with both file and stream handlers.
    - file_level controls what goes into the log file (default = INFO+)
    - level controls what goes to console (default = DEBUG+)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # avoid duplicate handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    return logger

# --- LOGGER SETUP USING HELPER ---
logger = setup_logger("transcription_logger", TRANSCRIPTION_LOG_FILE)
score_logger = setup_logger("score_logger", SCORE_LOG_FILE)
inactive_decay_logger = setup_logger("inactive_decay_logger", INACTIVE_DECAY_LOG_FILE)