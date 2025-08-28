import os
import json
from langchain_openai import ChatOpenAI
from datetime import datetime
from dotenv import load_dotenv
import warnings
import re
from logging_config import score_logger as logger
import time

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    logger.error("OPENAI_API_KEY not set. LLM features may not work.")
    openai_key = ""
else:
    openai_key = openai_key.strip()

llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0.3,
   api_key=openai_key
)

def safe_llm_invoke(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            wait = 4 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception("LLM call failed after maximum retries")

def extract_json_from_response(response_text):
    """
    Extracts a JSON object or array from a string that may contain extra text.
    """
    match = re.search(
        r'```(?:json)?\s*(\[.*\]|\{.*\})\s*```|(\[.*\]|\{.*\})',
        response_text.strip(),
        re.DOTALL
    )
    if match:
        json_str = match.group(1) or match.group(2)
        if json_str:
            return json_str

    first_brace = response_text.find('[')
    if first_brace == -1:
        first_brace = response_text.find('{')

    last_brace = response_text.rfind(']')
    if last_brace == -1:
        last_brace = response_text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return response_text[first_brace:last_brace + 1]

    raise ValueError("No valid JSON structure found in the response.")

def analyze_communication_log(communication_log):
    """
    Analyzes the communication log to generate day-wise summaries for all communications and a concise summary for today's interactions.
    """
    # Sort by timestamp (descending) and get today's communications
    valid_communications = [entry for entry in communication_log if validate_timestamp(entry.get('timestamp', ''))]
    if not valid_communications:
        logger.error("No valid timestamps found")
        return {"all_summary": [], "day_wise_summary": "", "latest_day": None}
        
    valid_communications.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    latest_timestamp = valid_communications[0].get('timestamp', '')
    latest_day = latest_timestamp.split('T')[0]
    
    todays_communications = [
        entry for entry in valid_communications 
        if entry.get('timestamp', '').startswith(latest_day)
    ]
    
    if not todays_communications:
        logger.warning(f"No communications found for {latest_day}")
        return {}, latest_day

    system_prompt = f"""You are a CRM conversation summarizer.
    TASK 1:
    Analyze today's communication log.

    Instructions:
    - Focus only on the student's or lead's intent, interest, and engagement (INBOUND COMMUNICATIONS)
    - DO NOT CONSIDER OUTBOUND COMMUNICATIONS FOR THE REASONING.
    - Provide a short reasoning (20 words max) that justifies this.
    - Do not repeat the communication verbatim. Instead, interpret the signals.
    
    OUTPUT FORMAT IS MENTIONED AT THE END.

    TASK 2:
    Given a chronological communication log, group the entries by date and produce one JSON per date with two main sections:

    1. active_conversations — only direct WhatsApp or Email replies (ignore link clicks, email opens and emails delivered). NO OUTBOUND MESSAGES ARE CONSIDERED.
    2. passive_signals — sub_activities like LINK_CLICK, EMAIL_OPEN, BOUNCED, etc. These are important.

    For each section, generate:
    - a concise summary of behavior
    - a strength/engagement label
    - the list of channels involved (for active only)
    - response time for active (Immediate / Delayed / No response)

    For OUTBOUND messages, there is no need to generate response time and engagement level. Generate only summary and list of channels.

    If there are no active conversations for a particular day, the active_conversations section should be empty.
    If there are no passive signals for a particular day, the passive_signals section should be empty.

    Return only a JSON array where each element represents a day's summary.
    Even if there's only one item, return it inside an array.
    Do not wrap it in an object.

    OUTPUT FORMAT FOR TASK 1 AND 2:
    
    ```json
    {{
    "all_summary":[{{
                "date": "YYYY-MM-DD",
                "active_conversations": {{
                    "channels": ["WHATSAPP", "EMAIL"], //or any other if present
                    "summary": "Detailed summary of the conversation behavior",
                    "engagement_level": "High" | "Medium" | "Low",
                    "response_time": "Immediate" | "Delayed" | "No response"
                    }},
                "passive_signals": {{
                    "summary": "Detailed summary of passive signals or actions",
                    "signal_strength": "High" | "Medium" | "Low"
                }}
    }}],
    "day_wise_summary": "reasoning here"}}
    ```    
    """
    user_prompt = f"""
    Here is today's communication log for TASK 1:
    {todays_communications}

    Here is the chronological communication log for TASK 2:
    {communication_log}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = safe_llm_invoke(messages)
        json_str = extract_json_from_response(response.content)
        json_str = json.loads(json_str)
        return json_str, latest_day
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to analyze communication log: {e}")
        return {"all_summary": [], "day_wise_summary": "", "latest_day": latest_day}

def extract_signals_from_input(daily_activities: list):
    """
    Extracts and separates active conversations and passive signals from the raw input,
    associating each with its date and preserving all original keys.
    """

    try:
        active_conversations = []
        passive_signals = []

        # Process the list of daily activities
        for activity in daily_activities:
            activity_date = activity.get("date")
            if not activity_date:
                continue

            if activity.get("active_conversations"):
                active_conversations_data = activity["active_conversations"]
                active_conversations_data["date"] = activity_date
                active_conversations.append(active_conversations_data)

            if activity.get("passive_signals"):
                passive_signals_data = activity["passive_signals"]
                passive_signals_data["date"] = activity_date
                passive_signals.append(passive_signals_data)
        
        return {
            "active_conversations": active_conversations,
            "passive_signals": passive_signals
        }
    except Exception:
        return {"active_conversations": [], "passive_signals": []}

def split_and_reduce_calls(communication_log: list) -> tuple:
    """
    Splits communication log into:
    1. Reduced CALL entries (only timestamp & message)
    2. Other entries for further processing by LLM
    """

    try:
        call_entries = []
        other_entries = []

        for entry in communication_log:
            channel = entry.get("channel", "").upper()
            
            if channel == "CALL":
                call_entries.append({
                    "timestamp": entry.get("timestamp"),
                    "message": entry.get("call_summary", "")
                })
            else:
                other_entries.append(entry)

        return call_entries, other_entries
    except Exception:
        return [], []

def validate_timestamp(timestamp: str) -> bool:
    """
    Validates if a timestamp is in proper ISO 8601 format.
    
    Args:
        timestamp (str): The timestamp string to validate.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if not timestamp:
        return False
    try:
        datetime.fromisoformat(timestamp)
        return True
    except ValueError:
        return False