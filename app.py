# app.py
#type:ignore
from flask import Flask, request, Response, jsonify
import json, os, subprocess
# from crew import run_qualification_agent, ValidationError
import redis
from job_queue import job_queue, get_job_status
# from crew import run_qualification_agent, ValidationError
import uuid
from version import __version__
# from crew_async import process_qualification_async
# import concurrent.futures

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# from crew_async import process_qualification_async
# import concurrent.futures

from log_cleanup_scheduler import start_log_cleanup_scheduler

from transcription_gpu import get_model_status


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
        webhook_url = os.getenv("TRANSCRIPTION_WEBHOOK_URL")

        if job_queue.full():
            return jsonify({
                "jobId": job_id,
                "status": "REJECTED",
                "message": "Server is busy. Try again later."
            }), 429

        job_queue.put((webhook_url, data))

        return jsonify({
            "jobId": job_id,
            "status": "QUEUED"
        }), 200

    except Exception as e:
        return jsonify({
            "jobId": data.get("jobId", None),
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
         "transcription_keys": int
      },
      "in_memory_queue_size": int
    }
    """
    gpu_busy = is_gpu_busy()
    job_counts = get_job_status()
    redis_busy = (job_counts["pending_webhooks"] > 0) or (job_counts["transcription_keys"] > 0) or (job_counts["active_jobs"] > 0)

    if gpu_busy or redis_busy or job_counts["in_memory_queue_size"] > 0:
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
            "transcription_keys": job_counts["transcription_keys"]
        },
        "in_memory_queue_size": job_counts["in_memory_queue_size"]
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
