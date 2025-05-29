# app.py
#type:ignore
from flask import Flask, request, Response, jsonify
import json
from crew import run_qualification_agent

import uuid
import logging
from transcription import process_transcription, handle_transcription_request
from crew_async import process_qualification_async
import concurrent.futures

app = Flask(__name__)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

@app.route("/qualify_lead", methods=["POST"])
def qualify_lead():
    try:
        data = request.get_json()
        if not data:
            error_msg = {"error": "No JSON payload provided"}
            return Response(json.dumps(error_msg), mimetype='application/json', status=400)

        result = run_qualification_agent(data)

        json_string = json.dumps(result, indent=2, sort_keys=False)
        return Response(json_string, mimetype='application/json', status=200)

    except Exception as e:
        print(f"ERROR in qualify_lead route: {e}") # Keep this print for server-side error logging
        error_msg = {"error": str(e)}
        return Response(json.dumps(error_msg), mimetype='application/json', status=500)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    API endpoint to handle transcription requests.
    Expects a JSON payload with audio or transcription parameters.
    Calls handle_transcription_request() to process the request synchronously.
    Returns the transcription result in JSON with HTTP 200 on success,
    or HTTP 500 if an error occurs.
    """
    try:
        data = request.get_json()
        response = handle_transcription_request(data)
        # Return 200 if successful, else 500 for failure in processing
        return jsonify(response), 200 if response["status"] == "success" else 500
    except Exception as e:
        # Return 500 and error message on unexpected exceptions
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500


@app.route("/qualify_lead_async", methods=["POST"])
def qualify_lead_async():
    """
    API endpoint to asynchronously process lead qualification.
    Expects a JSON payload with lead data.
    Immediately returns a processing receipt response.
    Background task executor submits the qualification processing function,
    which will post results to a specified webhook URL upon completion.
    """
    try:
        data = request.get_json()
        if not data:
            # Return 400 Bad Request if no JSON payload was provided
            return jsonify({"error": "No JSON payload provided"}), 400

        webhook_url = "{BASE URL + URL}"  # Replace with actual webhook URL from env

        # Submit the qualification processing to a background thread/executor
        executor.submit(process_qualification_async, webhook_url, data)

        # Immediately return a response indicating processing has started
        return jsonify({
            "status": "processing",
            "message": "Request received and processing started",
        }), 200

    except Exception as e:
        # Return 500 Internal Server Error with exception message on failure
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")