# job_queue.py
import threading, queue, requests, redis, time, json
from transcription_gpu import handle_transcription_request
import os
from logging_config import logger

# In-memory job queue with max size
job_queue = queue.Queue(maxsize=100)

# Redis setup
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

NUM_WORKERS = 2
WEBHOOK_RETRY_LIST = "pending_webhooks"
SECURITY_TOKEN = os.environ.get("SECURITY_TOKEN")

def worker():
    while True:
        try:
            webhook_url, data = job_queue.get()
            jobId = data.get("jobId")
            result = handle_transcription_request(data)

            # Save result to Redis
            redis_key = f"transcription:{jobId}"
            redis_client.set(redis_key, json.dumps(result), ex=86400)  # 24h TTL
            logger.debug(f"[Worker] Saved result to Redis for jobId")

            # Try sending to webhook
            headers = {
                "Content-Type": "application/json",
                "X-SECURITY-TOKEN": SECURITY_TOKEN
            }
            response = requests.post(webhook_url, json=result, headers=headers)

            if response.status_code == 200:
                redis_client.delete(redis_key)
                logger.info(f"[Worker] Successfully posted to webhook for jobId: {jobId}")
            else:
                schedule_retry(jobId, webhook_url, initial_delay=5)

        except Exception as e:
            logger.exception(f"[Worker] Error processing job: {e}")
            try:
                schedule_retry(data.get("jobId"), webhook_url, initial_delay=5)
            except:
                pass
        finally:
            job_queue.task_done()

def schedule_retry(job_id, webhook_url, initial_delay):
    payload = {
        "jobId": job_id,
        "webhookUrl": webhook_url,
        "attempts": 1,
        "next_retry": time.time() + initial_delay
    }
    redis_client.rpush(WEBHOOK_RETRY_LIST, json.dumps(payload))

def webhook_retry_worker():
    while True:
        all_items = redis_client.lrange(WEBHOOK_RETRY_LIST, 0, -1)
        for item in all_items:
            data = json.loads(item)
            jobId = data["jobId"]
            webhook_url = data["webhookUrl"]
            attempts = data["attempts"]
            next_retry = data["next_retry"]

            if time.time() >= next_retry:
                result = redis_client.get(f"transcription:{jobId}")
                if result is None:
                    redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
                    continue

                try:
                    headers = {
                        "Content-Type": "application/json",
                        "X-SECURITY-TOKEN": SECURITY_TOKEN
                    }
                    resp = requests.post(webhook_url, json=json.loads(result), headers=headers)
                    if resp.status_code == 200:
                        redis_client.delete(f"transcription:{jobId}")
                        redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
                    else:
                        if attempts < 10:  # ~24h if exponential
                            backoff = min(2 ** attempts, 3600)  # cap at 1hr
                            data["attempts"] += 1
                            data["next_retry"] = time.time() + backoff
                            redis_client.lset(WEBHOOK_RETRY_LIST, all_items.index(item), json.dumps(data))
                        else:
                            redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
                except Exception as e:
                    logger.error(f"[RetryWorker] Failed to send job: {e}")
        time.sleep(10)

# Launch transcription workers
for _ in range(NUM_WORKERS):
    threading.Thread(target=worker, daemon=True).start()

# Launch retry worker
threading.Thread(target=webhook_retry_worker, daemon=True).start()
