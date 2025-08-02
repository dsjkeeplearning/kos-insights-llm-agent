# job_queue.py
import threading, queue, requests, redis, time, json
from transcription_gpu import handle_transcription_request
import os
from logging_config import logger
from dotenv import load_dotenv

load_dotenv()

# In-memory job queue with max size
job_queue = queue.Queue(maxsize=100)

# Redis setup
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Track active jobs in Redis
ACTIVE_JOBS_KEY = "active_jobs"

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
WEBHOOK_RETRY_LIST = "pending_webhooks"
SECURITY_TOKEN = os.environ.get("SECURITY_TOKEN")

def worker():
    while True:
        webhook_url, data = None, None
        try:
            webhook_url, data = job_queue.get()
            jobId = data.get("jobId")

            # Track job as active
            try:
                redis_client.sadd(ACTIVE_JOBS_KEY, jobId)
            except Exception as e:
                logger.error(f"[Worker] Failed to mark job {jobId} as active: {e}")

            result = handle_transcription_request(data)

            # Save result to Redis
            redis_key = f"transcription:{jobId}"
            try:
                redis_client.set(redis_key, json.dumps(result), ex=86400)  # 24h TTL
            except Exception as e:
                logger.error(f"[Worker] Failed to save transcription result for {jobId}: {e}")
            logger.debug(f"[Worker] Saved result to Redis for jobId: {jobId}")

            # Try sending to webhook
            headers = {
                "Content-Type": "application/json",
                "X-SECURITY-TOKEN": SECURITY_TOKEN
            }
            response = requests.post(webhook_url, json=result, headers=headers)

            if response.status_code == 200:
                try:
                    redis_client.delete(redis_key)
                except Exception as e:
                    logger.error(f"[Worker] Failed to delete transcription key {redis_key}: {e}")
                logger.info(f"[Worker] Successfully posted to webhook for jobId: {jobId}")
            else:
                schedule_retry(jobId, webhook_url, initial_delay=5)

        except Exception as e:
            logger.exception(f"[Worker] Error processing job: {e}")
            if data and webhook_url:
                try:
                    schedule_retry(data.get("jobId"), webhook_url, initial_delay=5)
                except Exception as ex:
                    logger.error(f"[Worker] Failed to schedule retry: {ex}")
        finally:
            # Remove job from active tracking
            try:
                redis_client.srem(ACTIVE_JOBS_KEY, jobId)
            except Exception as e:
                logger.error(f"[Worker] Failed to remove job {jobId} from active set: {e}")
            job_queue.task_done()

def schedule_retry(job_id, webhook_url, initial_delay):
    payload = {
        "jobId": job_id,
        "webhookUrl": webhook_url,
        "attempts": 1,
        "next_retry": time.time() + initial_delay
    }
    try:
        redis_client.rpush(WEBHOOK_RETRY_LIST, json.dumps(payload))
    except Exception as e:
        logger.error(f"[Retry] Failed to schedule retry for job {job_id}: {e}")

def webhook_retry_worker():
    while True:
        try:
            all_items = redis_client.lrange(WEBHOOK_RETRY_LIST, 0, -1)
        except Exception as e:
            logger.error(f"[RetryWorker] Failed to read retry list: {e}")
            time.sleep(10)
            continue

        for idx, item in enumerate(all_items):
            data = json.loads(item)
            jobId = data["jobId"]
            webhook_url = data["webhookUrl"]
            attempts = data["attempts"]
            next_retry = data["next_retry"]

            if time.time() >= next_retry:
                try:
                    result = redis_client.get(f"transcription:{jobId}")
                except Exception as e:
                    logger.error(f"[RetryWorker] Failed to get transcription for {jobId}: {e}")
                    continue

                if result is None:
                    try:
                        redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
                    except Exception as e:
                        logger.error(f"[RetryWorker] Failed to remove retry item for {jobId}: {e}")
                    continue

                try:
                    headers = {
                        "Content-Type": "application/json",
                        "X-SECURITY-TOKEN": SECURITY_TOKEN
                    }
                    resp = requests.post(webhook_url, json=json.loads(result), headers=headers)
                    if resp.status_code == 200:
                        try:
                            redis_client.delete(f"transcription:{jobId}")
                        except Exception as e:
                            logger.error(f"[RetryWorker] Failed to delete transcription key for {jobId}: {e}")
                        try:
                            redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
                        except Exception as e:
                            logger.error(f"[RetryWorker] Failed to remove retry item for {jobId}: {e}")
                    else:
                        if attempts < 10:  # ~24h if exponential
                            backoff = min(2 ** attempts, 3600)  # cap at 1hr
                            data["attempts"] += 1
                            data["next_retry"] = time.time() + backoff
                            try:
                                redis_client.lset(WEBHOOK_RETRY_LIST, idx, json.dumps(data))
                            except Exception as e:
                                logger.error(f"[RetryWorker] Failed to update retry entry for {jobId}: {e}")
                        else:
                            try:
                                redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
                            except Exception as e:
                                logger.error(f"[RetryWorker] Failed to remove retry item for {jobId}: {e}")
                except Exception as e:
                    logger.error(f"[RetryWorker] Failed to send job: {e}")
        time.sleep(10)

# Helper to get current job status counts
def get_job_status():
    try:
        pending_len = redis_client.llen(WEBHOOK_RETRY_LIST)
    except Exception:
        pending_len = -1
    try:
        active_jobs = redis_client.scard(ACTIVE_JOBS_KEY)
    except Exception:
        active_jobs = -1
    try:
        transcription_keys = len(redis_client.keys("transcription:*"))
    except Exception:
        transcription_keys = -1

    return {
        "in_memory_queue_size": job_queue.qsize(),
        "active_jobs": active_jobs,
        "pending_webhooks": pending_len,
        "transcription_keys": transcription_keys
    }

# Launch transcription workers
for _ in range(NUM_WORKERS):
    threading.Thread(target=worker, daemon=True).start()

# Launch retry worker
threading.Thread(target=webhook_retry_worker, daemon=True).start()
