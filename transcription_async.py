# transcription_async.py
import requests
import logging
from transcription import handle_transcription_request # Ensure this import is correct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_transcription_async(webhook_url, data):
    """
    Background function to process transcription asynchronously.

    - Calls the synchronous `handle_transcription_request` function with input data.
    - Sends the result of transcription to the specified webhook URL via HTTP POST.
    - In case of exceptions, sends an error message to the webhook URL.
    """
    logging.info(f"Background task: Starting transcription for data with fixed callback: {webhook_url}")
    try:
        result = handle_transcription_request(data)
        # result = json.dumps(result, indent=2)
        logging.info(f"Background task: Transcription finished. Attempting to post result to webhook: {webhook_url}")
        requests.post(webhook_url, json=result, headers={"Content-Type": "application/json"})
        logging.info(f"Background task: Successfully posted transcription result to webhook")

    except requests.exceptions.RequestException as req_e:
        logging.error(f"Background task: Error posting transcription result to webhook {webhook_url}: {req_e}")
        error_payload = {
            "jobId": data.get("jobId"),
            "status": "FAILED"
        }
        try:
            requests.post(webhook_url, json=error_payload, headers={"Content-Type": "application/json"})
        except Exception as inner_e:
            logging.error(f"Background task: An unexpected error occurred while trying to post webhook delivery error: {inner_e}")
    except Exception as e:
        logging.error(f"Background task: An error occurred during transcription processing for data: {data}. Error: {e}", exc_info=True)
        error_payload = {
            "jobId": data.get("jobId"),
            "status": "FAILED"
        }
        try:
            requests.post(webhook_url, json=error_payload, headers={"Content-Type": "application/json"})
            logging.info(f"Background task: Posted processing error to webhook: {webhook_url}")
        except Exception as webhook_error:
            logging.error(f"Background task: Failed to post processing error to webhook {webhook_url} after initial failure: {webhook_error}")
