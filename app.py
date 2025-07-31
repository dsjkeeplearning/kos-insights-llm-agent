# app.py
#type:ignore
from flask import Flask, request, Response, jsonify
import json, os
# from crew import run_qualification_agent, ValidationError
import uuid
import logging
# from crew_async import process_qualification_async
# import concurrent.futures
from job_queue import job_queue
from log_cleanup_scheduler import start_log_cleanup_scheduler

app = Flask(__name__)
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

SECURITY_HEADER = "X-SECURITY-TOKEN"
SECURITY_TOKEN = os.environ.get("SECURITY_TOKEN")

if SECURITY_TOKEN is None:
    raise RuntimeError("SECURITY_TOKEN environment variable must be set")

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


# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     """
#     API endpoint to handle transcription requests asynchronously.
#     Expects a JSON payload with audio or transcription parameters.
#     The webhook URL for results is configured via environment variable.
#     Immediately returns a processing receipt response (HTTP 200 OK).
#     The actual transcription happens in a background thread, and the result
#     is sent to the pre-configured webhook URL upon completion.
#     """
#     try:
#         data = request.get_json()

#         # Get the webhook URL from an environment variable
#         webhook_url = os.getenv("TRANSCRIPTION_WEBHOOK_URL")
#         # logging.info(f"Transcription request received. Submitting to background for webhook: {webhook_url}")

#         # Submit the background transcription task to the thread pool
#         executor.submit(process_transcription_async, webhook_url, data)

#         # Immediately return a 200 OK response
#         return jsonify({
#             "jobId": data.get("jobId"),
#             "status": "QUEUED",
#         }), 200 # HTTP 200 OK

#     except Exception as e:
#         # logging.error(f"Error in transcribe endpoint: {e}", exc_info=True)
#         return jsonify({
#             "jobId": data.get("jobId"),
#             "status": "FAILED",
#         }), 500


# @app.route("/qualify_lead_async", methods=["POST"])
# def qualify_lead_async():
#     """
#     API endpoint to asynchronously process lead qualification.
#     Expects a JSON payload with lead data.
#     Immediately returns a processing receipt response.
#     Background task executor submits the qualification processing function,
#     which will post results to a specified webhook URL upon completion.
#     """
#     try:
#         data = request.get_json()
#         if not data:
#             # Return 400 Bad Request if no JSON payload was provided
#             return jsonify({"error": "No JSON payload provided"}), 400

#         webhook_url = "{BASE URL + URL}"  # Replace with actual webhook URL from env

#         # Submit the qualification processing to a background thread/executor
#         executor.submit(process_qualification_async, webhook_url, data)

#         # Immediately return a response indicating processing has started
#         return jsonify({
#             "status": "processing",
#             "message": "Request received and processing started",
#         }), 200

#     except Exception as e:
#         # Return 500 Internal Server Error with exception message on failure
#         return jsonify({"error": str(e)}), 500


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


if __name__ == "__main__":
    start_log_cleanup_scheduler()
    app.run(debug=True, port=5000, host="0.0.0.0")
