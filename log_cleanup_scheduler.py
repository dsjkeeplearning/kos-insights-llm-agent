from apscheduler.schedulers.background import BackgroundScheduler
from logging_config import clean_old_log_files
import atexit

def start_log_cleanup_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        clean_old_log_files,
        trigger="cron",
        hour=0,
        minute=1,
        kwargs={"log_dir": "logs", "days": 7},
        id="daily_log_cleanup",
        replace_existing=True
    )
    scheduler.start()

    # Ensure it shuts down gracefully on app exit
    atexit.register(lambda: scheduler.shutdown())