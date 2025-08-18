import threading, queue, requests, redis, time, json, os
from dotenv import load_dotenv
from logging_config import score_logger as logger
from score import add_all_scores  

load_dotenv()

# In-memory job queue
lead_score_queue = queue.Queue(maxsize=200)

# Redis setup
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Constants for Redis keys
ACTIVE_JOBS_KEY = "lead_score_active_jobs"
WEBHOOK_RETRY_LIST = "lead_score_pending_webhooks"

NUM_WORKERS = int(os.getenv("LEAD_NUM_WORKERS", "2"))
SECURITY_TOKEN = os.environ.get("SECURITY_TOKEN")

def worker():
    while True:
        webhook_url, data = None, None
        try:
            webhook_url, data = lead_score_queue.get()
            lead_id = data.get("lead_id")

            # Track job as active
            redis_client.sadd(ACTIVE_JOBS_KEY, lead_id)

            result = add_all_scores(data)

            # Save result to Redis
            redis_key = f"lead_score:{lead_id}"
            redis_client.set(redis_key, json.dumps(result), ex=86400)

            # Send to webhook
            headers = {
                "Content-Type": "application/json",
                "X-SECURITY-TOKEN": SECURITY_TOKEN
            }
            resp = requests.post(webhook_url, json=result, headers=headers)
            if resp.status_code == 200:
                redis_client.delete(redis_key)
                logger.info(f"[LeadScoreWorker] Posted to webhook for {lead_id}")
            else:
                schedule_retry(lead_id, webhook_url, initial_delay=5)
        except Exception as e:
            logger.exception(f"[LeadScoreWorker] Error: {e}")
            if webhook_url and data:
                schedule_retry(data.get("lead_id"), webhook_url, initial_delay=5)
        finally:
            redis_client.srem(ACTIVE_JOBS_KEY, lead_id)
            lead_score_queue.task_done()

def schedule_retry(job_id, webhook_url, initial_delay):
    payload = {
        "lead_id": job_id,
        "webhookUrl": webhook_url,
        "attempts": 1,
        "next_retry": time.time() + initial_delay
    }
    redis_client.rpush(WEBHOOK_RETRY_LIST, json.dumps(payload))

def webhook_retry_worker():
    while True:
        all_items = redis_client.lrange(WEBHOOK_RETRY_LIST, 0, -1)
        for idx, item in enumerate(all_items):
            data = json.loads(item)
            if time.time() < data["next_retry"]:
                continue
            lead_id = data["lead_id"]
            webhook_url = data["webhookUrl"]
            attempts = data["attempts"]

            result = redis_client.get(f"lead_score:{lead_id}")
            if not result:
                redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
                continue

            headers = {
                "Content-Type": "application/json",
                "X-SECURITY-TOKEN": SECURITY_TOKEN
            }
            resp = requests.post(webhook_url, json=json.loads(result), headers=headers)
            if resp.status_code == 200:
                redis_client.delete(f"lead_score:{lead_id}")
                redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
            else:
                if attempts < 10:
                    backoff = min(2 ** attempts, 3600)
                    data["attempts"] += 1
                    data["next_retry"] = time.time() + backoff
                    redis_client.lset(WEBHOOK_RETRY_LIST, idx, json.dumps(data))
                else:
                    redis_client.lrem(WEBHOOK_RETRY_LIST, 0, item)
        time.sleep(10)

# Helper for status
def get_lead_score_job_status():
    return {
        "in_memory_queue_size": lead_score_queue.qsize(),
        "active_jobs": redis_client.scard(ACTIVE_JOBS_KEY),
        "pending_webhooks": redis_client.llen(WEBHOOK_RETRY_LIST),
        "stored_results": len(redis_client.keys("lead_score:*"))
    }

# Start workers
for _ in range(NUM_WORKERS):
    threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=webhook_retry_worker, daemon=True).start()
