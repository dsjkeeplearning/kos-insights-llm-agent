# app.py
#type:ignore
from flask import Flask, request, Response, jsonify
import json, os, subprocess
import redis
from job_queue import job_queue, get_job_status
# from crew import run_qualification_agent, ValidationError
from version import __version__
from datetime import datetime, timezone
import requests

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from log_cleanup_scheduler import start_log_cleanup_scheduler
from transcription_gpu import get_model_status
from lead_score_job_queue import lead_score_queue, get_lead_score_job_status
from lead_decay_job_queue import lead_decay_queue, get_lead_decay_job_status


def log_gpu_info():
    """Logs GPU presence, driver version, and GPU names using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True
        )
        lines = output.strip().splitlines()
        if not lines:
            logger.warning("⚠️ No GPUs detected by nvidia-smi.")
            return
        for idx, line in enumerate(lines):
            name, driver = [part.strip() for part in line.split(",")]
            logger.info(f"🟢 GPU {idx}: {name}, Driver Version: {driver}")
    except FileNotFoundError:
        logger.error("❌ nvidia-smi not found. NVIDIA driver may not be installed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to query GPU info: {e}")

# Redis client for status checks
app = Flask(__name__)
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def is_gpu_busy():
    """Check if GPU is actively processing (not just loaded)."""
    try:
        # Check actual GPU utilization percentage
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        )
        gpu_util = int(output.strip())
        return gpu_util > 5  # Consider busy if >5% utilization
    except Exception:
        return False
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

SECURITY_HEADER = "X-SECURITY-TOKEN"
SECURITY_TOKEN = os.environ.get("SECURITY_TOKEN")

if SECURITY_TOKEN is None:
    raise RuntimeError("SECURITY_TOKEN environment variable must be set")
logger.info("✅ SECURITY_TOKEN found and validated")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY environment variable must be set")
logger.info("✅ OPENAI_API_KEY found and validated")

def get_openai_request_count():
    """Fetch today's OpenAI API request count."""
    today = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        resp = requests.get(f"https://api.openai.com/v1/usage?date={today}", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        total_requests = sum(item.get("n_requests", 0) for item in data.get("data", []))
        logger.info(f"OpenAI API request count for {today}: {total_requests}")
        return total_requests
    except Exception as e:
        # fallback in case API fails
        return -1

@app.before_request
def require_security_header():
    token = request.headers.get(SECURITY_HEADER)
    if token != SECURITY_TOKEN:
        error_msg = {"error": "Unauthorized: Missing or invalid security token"}
        return Response(json.dumps(error_msg), mimetype='application/json', status=401)


@app.route("/qualify_lead", methods=["POST"])
def qualify_lead():
    try:
        data = request.get_json()
        if not data:
            error_msg = {"error": "No JSON payload provided"}
            return Response(json.dumps(error_msg), mimetype='application/json', status=400)

        try:
            result = run_qualification_agent(data)
        except ValidationError as e:
            error_msg = {"error": str(e)}
            return Response(json.dumps(error_msg), mimetype='application/json', status=400)

        json_string = json.dumps(result, indent=2, sort_keys=False)
        return Response(json_string, mimetype='application/json', status=200)

    except Exception as e:
        print(f"ERROR in qualify_lead route: {e}") # Keep this print for server-side error logging
        error_msg = {"error": str(e)}
        return Response(json.dumps(error_msg), mimetype='application/json', status=500)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        job_id = data.get("jobId")
        reference_id = data.get("referenceId")
        webhook_url = os.getenv("TRANSCRIPTION_WEBHOOK_URL")

        if job_queue.full():
            return jsonify({
                "jobId": job_id,
                "referenceId": reference_id,
                "status": "REJECTED",
                "message": "Server is busy. Try again later."
            }), 429

        job_queue.put((webhook_url, data))

        return jsonify({
            "jobId": job_id,
            "referenceId": reference_id,
            "status": "QUEUED"
        }), 200

    except Exception as e:
        return jsonify({
            "jobId": data.get("jobId", None),
            "referenceId": data.get("referenceId", None),
            "status": "FAILED",
            "error": str(e)
        }), 500

@app.route("/calculate_lead_score", methods=["POST"])
def calculate_lead_score():
    try:
        data = request.get_json()
        lead_id = data.get("lead_id")
        webhook_url = os.getenv("WEBHOOK_URL")

        if lead_score_queue.full():
            return jsonify({
                "lead_id": lead_id,
                "status": "REJECTED",
                "message": "Server is busy. Try again later."
            }), 429

        lead_score_queue.put((webhook_url, data))

        return jsonify({
            "lead_id": lead_id,
            "status": "QUEUED"
        }), 200

    except Exception as e:
        return jsonify({
            "lead_id": data.get("lead_id", None),
            "status": "FAILED",
            "error": str(e)
        }), 500


@app.route("/calculate_decay", methods=["POST"])
def calculate_decay():
    try:
        data = request.get_json()
        lead_id = data.get("lead_id")
        webhook_url = os.getenv("WEBHOOK_URL")

        if lead_decay_queue.full():
            return jsonify({
                "lead_id": lead_id,
                "status": "REJECTED",
                "message": "Server is busy. Try again later."
            }), 429

        lead_decay_queue.put((webhook_url, data))

        return jsonify({
            "lead_id": lead_id,
            "status": "QUEUED"
        }), 200

    except Exception as e:
        return jsonify({
            "lead_id": data.get("lead_id", None),
            "status": "FAILED",
            "error": str(e)
        }), 500


# Endpoint to check system status for safe shutdown
@app.route('/status', methods=['GET'])
def status():
    """
    Request: GET /status (requires X-SECURITY-TOKEN header)
    Response JSON:
    {
      "status": "SAFE" | "BUSY",
      "gpu_busy": bool,
      "redis_job_counts": {
         "active_jobs": int,
         "pending_webhooks": int,
         "transcription_keys": int,
         "in_memory_queue_size": int
      }, //for all 3 redis services
      "rate_limit_safe": bool
    }
    """
    gpu_busy = is_gpu_busy()
    job_counts = get_job_status()
    lead_score_counts = get_lead_score_job_status()
    lead_decay_counts = get_lead_decay_job_status()
    total_requests = get_openai_request_count()
    rate_limit_safe = total_requests < 9500 if total_requests >= 0 else None

    redis_busy = (
        job_counts["pending_webhooks"] > 0 or 
        job_counts["transcription_keys"] > 0 or 
        job_counts["active_jobs"] > 0 or
        lead_score_counts["pending_webhooks"] > 0 or
        lead_score_counts["stored_results"] > 0 or
        lead_score_counts["active_jobs"] > 0 or
        lead_decay_counts["pending_webhooks"] > 0 or
        lead_decay_counts["stored_results"] > 0 or
        lead_decay_counts["active_jobs"] > 0
    )

    queues_busy = (
        job_counts["in_memory_queue_size"] > 0 or
        lead_score_counts["in_memory_queue_size"] > 0 or
        lead_decay_counts["in_memory_queue_size"] > 0
    )

    if gpu_busy or redis_busy or queues_busy:
        overall_status = "BUSY"
    else:
        overall_status = "SAFE"

    return jsonify({
        "version": __version__,
        "status": overall_status,
        "gpu_busy": gpu_busy,
        "redis_job_counts": {
            "active_jobs": job_counts["active_jobs"],
            "pending_webhooks": job_counts["pending_webhooks"],
            "transcription_keys": job_counts["transcription_keys"],
            "in_memory_queue_size": job_counts["in_memory_queue_size"]
        },
        "lead_score_counts": {
            "active_jobs": lead_score_counts["active_jobs"],
            "pending_webhooks": lead_score_counts["pending_webhooks"],
            "stored_results": lead_score_counts["stored_results"],
            "in_memory_queue_size": lead_score_counts["in_memory_queue_size"]
        },
        "lead_decay_counts": {
            "active_jobs": lead_decay_counts["active_jobs"],
            "pending_webhooks": lead_decay_counts["pending_webhooks"],
            "stored_results": lead_decay_counts["stored_results"],
            "in_memory_queue_size": lead_decay_counts["in_memory_queue_size"]
        },
        "rate_limit_safe": rate_limit_safe
    }), 200


def initialize_service():
    """Run startup checks and log system readiness."""
    logger.info("🚀 Starting LLM Agent Service...")
    logger.info("🟢 Initializing log cleanup scheduler")
    start_log_cleanup_scheduler()
    try:
        logger.info("🟢 Redis connection initialized: %s", redis_client.ping())
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
    log_gpu_info()
    model_status = get_model_status()
    if model_status["loaded"]:
        logger.info(f"🟢 WhisperX model '{model_status['model_name']}' loaded successfully on {model_status['device']} (CUDA available: {model_status['cuda_available']})")
    else:
        logger.error(f"❌ Failed to load WhisperX model '{model_status['model_name']}' on {model_status['device']} (CUDA available: {model_status['cuda_available']})")


# Always run initialization (for both Gunicorn and direct execution)
initialize_service()


if __name__ == "__main__":
    logger.info("🟢 Flask app is running on http://0.0.0.0:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
