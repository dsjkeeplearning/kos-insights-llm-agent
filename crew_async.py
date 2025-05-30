import requests
from crew import run_qualification_agent

def process_qualification_async(webhook_url, data):
    """
    Background function to process lead qualification asynchronously.

    - Calls the synchronous `run_qualification_agent` function with input data.
    - Sends the result of qualification to the specified webhook URL via HTTP POST.
    - In case of exceptions, sends an error message to the webhook URL.
    """
    try:
        # Run the lead qualification logic synchronously
        result = run_qualification_agent(data)

        # Post the successful result back to the webhook URL
        requests.post(webhook_url, json=result, headers={"Content-Type": "application/json"})

    except Exception as e:
        # On error, prepare an error response payload
        error_response = {
            "error": str(e),
            "status": "error"
        }
        # Post the error response to the webhook URL
        requests.post(webhook_url, json=error_response, headers={"Content-Type": "application/json"})
