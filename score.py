import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import warnings
from preprocess import analyze_communication_log, extract_signals_from_input, split_and_reduce_calls, extract_json_from_response, safe_llm_invoke, summarize_todays_communication
from logging_config import score_logger as logger

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


def get_passive_score(passive_signals, today_date):
    """
    Calls GPT-4o mini via LangChain to provide a qualitative passive engagement score.

    Args:
        passive_signals (list): A list of dictionaries for each passive signal event.
        
    Returns:
        dict: A dictionary with passive_score and passive_summary.
    """
    if not passive_signals:
        return {"passive_score": 0, "passive_summary": "No passive signals provided."}

    prompt = f"""
    You are a lead engagement evaluator assessing passive signals like link clicks, email opens, and page visits.

    ### Input
    Today's date: {today_date}
    Passive signals (list of event dictionaries):
    {passive_signals}

    ### Instructions
    Use the following **Passive Engagement Evaluation Rubric** to assign a score between 0-10.
    Award points for each category, then sum them (max 10 points).
    Apply special rules before finalizing.
    After scoring, map the final score to the **Qualitative Summary Mapping Table** exactly.
    #### Passive Engagement Evaluation Rubric
    **1. Recency of Engagement (0-3 points)**
    - 3: Majority of signals occurred within the last 4 days; includes at least one high-intent action (application form click).
    - 2: Majority of signals occurred within the last 7 days; includes at least one relevant link click.
    - 1: All signals occurred within the last 14 days, but none within the last 4 days.
    - 0: All signals are older than 14 days; no recent activity.
    **2. Relevance of Actions (0-3 points)**
    - 3: Engagement with multiple high-value assets (application form, curriculum page, fee details).
    - 2: Engagement with at least one high-value asset but other actions are low-intent.
    - 1: Engagement limited to low-value actions (generic browsing, social links).
    - 0: No relevant or high-intent actions detected.
    **3. Diversity of Engagement (0-2 points)**
    - 2: Interacted with 3+ unique link types or channels (brochure, curriculum, email click, etc.).
    - 1: Interacted with 2 distinct link types.
    - 0: Only one link type engaged with.
    **4. Intent Strength (0-2 points)**
    - 2: At least one high-intent action (application form click, enquiry form, program fee page) in last 4 days.
    - 1: At least one moderate-intent action (brochure download, curriculum view) within last 7 days.
    - 0: No high/moderate-intent actions; activity is casual or non-committal.

    #### Special Rules
    - Repeat clicks on same link within 7 days: Max +2 points total.
    - Excessive clicks (>3) on one link: Count as single strong signal.
    - Signals older than 14 days: Reduce their scoring weight by half.
    - Email opens: Only score if within 7 days.
    - Email clicks: +2 points only if relevant and recent.
    - Page views: Only score if tied to high-value content (max +1 point).

    #### Qualitative Summary Mapping Table
    - 9-10: "Highly engaged, strong intent — likely to convert."
    - 6-8: "Moderately engaged, potential interest — worth nurturing."
    - 3-5: "Low engagement, weak intent — needs targeted outreach."
    - 0-2: "No meaningful engagement — minimal follow-up priority."

    ### Summary:
    Explanation of why the passive score was assigned.

    ### Output (JSON only)
    Return a score and a brief reasoning, using this format:
    {{
    "passive_score": <integer between 0-10>,
    "passive_summary": [
        "<point 1>",
        "<point 2>",
        "... as many as needed"
    ]
    }}

    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only returns JSON."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = safe_llm_invoke(messages)
        json_str = extract_json_from_response(response.content)
        data = json.loads(json_str)

        # Force passive_score to integer
        if "passive_score" in data:
            try:
                data["passive_score"] = int(round(float(data["passive_score"])))
            except (ValueError, TypeError):
                data["passive_score"] = 0
        logger.debug("Passive score generated successfully")
        return data

    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Error parsing passive score JSON: {e}")
        return {
            "passive_score": 0,
            "passive_summary": "Error parsing response from LLM. Passive signals could not be evaluated."
        }


def get_conversion_score(summary_object, active_conversations, today_date):

    """
    Calls GPT-4o mini via LangChain to determine conversion intent from both the summary object
    and a list of active conversation events.
    
    Args:
        summary_object (dict): The dictionary containing the overall lead summary.
        active_conversations (list): A list of dictionaries for each conversation event.

    Returns:
        dict: A dictionary with conversion_score, and conversion_summary.
    """
    
    prompt = f"""
    You are a lead qualification assistant. Your job is to find the **single strongest signal** that a lead is ready to submit an application.

    Date: {today_date}

    ### Data:
    **Call Summaries:**
    {summary_object}
    **WhatsApp & Email Summaries:**
    {active_conversations}

    ---
    ### How to Score (choose the HIGHEST applicable; do NOT add):
    1. **Form submitted / completed**
    Examples: "I have filled the form", "Form submitted", "Completed application", "Admission granted" → **60**
    2. **Will submit very soon (today/now)**
    Examples: "Will fill tonight", "Filling now", "Will do today", "Ready to fill", "Sent my documents" → **55**
    3. **Requests application link/form**
    Examples: "Send me the form", "Where is the link?" → **45**
    4. **Asks how to apply / next step**
    Examples: "How do I apply?", "What's next to apply?" → **40**
    5. **Soft future intent** (only if profile fit & past intent shown)
    Examples: "I'll apply soon", "Planning to apply", "Will confirm soon" → **20-25**
    6. **General interest only** → **0**
    7. **Objections/irrelevant talk** → **0**

    ---
    ### Rules:
    - Use only ONE phrase/action — the strongest — from all inputs.
    - GENERALIZATION: In case there are other phrases with a similar meaning, understand the context and assign the score accordingly.
    - Cap at **60**.
    - If no valid signal, score = 0.

    ### Summary:
    Explanation of why the conversion score was assigned.

    ---
    ### Output (JSON only, no extra text):
    {{
    "conversion_score": 0-60,
    "conversion_summary": [
        "<point 1>",
        "<point 2>",
        "... as many as needed"
    ]
    }}
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only returns JSON."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = safe_llm_invoke(messages)
        json_str = extract_json_from_response(response.content)
        data = json.loads(json_str)

        # Force conversion_score to integer
        if "conversion_score" in data:
            try:
                data["conversion_score"] = int(round(float(data["conversion_score"])))
            except (ValueError, TypeError):
                data["conversion_score"] = 0

        logger.debug("Conversion score generated successfully")
        return data
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Error parsing conversion score JSON: {e}")
        return {
            "conversion_score": 0, 
            "conversion_summary": "Error parsing response from LLM. Conversion signals could not be evaluated."
        }


def get_active_score(summary_object, active_conversations, today_date):
    """
    Scores active conversations by combining summaries and calculating recency.
    
    Args:
        summary_object (dict): The dictionary containing the overall lead summary.
        active_conversations (list): A list of dictionaries for each conversation event.

    Returns:
        dict: A dictionary with active_score and active_summary.
    """
    if not active_conversations and not summary_object:
        return {"active_score": 0, "active_summary": "No active conversation found."}

    prompt = f"""
    You are an EdTech CRM lead conversation quality evaluator.
    Today's date: {today_date}

    ## Input Data
    - **WhatsApp & Email Logs Summary**:
    {active_conversations}
    - **Call Logs Summary**:
    {summary_object}
    ---

    ## Scoring Instructions (Strict Rubric)
    ### Step 1 — Quality Score (0-10)
    
    Follow this rubric exactly. Always pick the **lowest score** in the matching range and output with **one decimal place**.
    #### 0.0-1.0 → No Meaningful Engagement
    - Irrelevant, spam, or no response.
    - No program-related engagement at all.
    - Examples: "Ok", blank, emoji only.
    #### 1.1-3.0 → Very Low Engagement
    - Vague or generic replies.
    - No direct interest or intent.
    - Examples: "Will see later", "I'm busy" with no reschedule.
    #### 3.1-5.0 → Low-Moderate Engagement
    - Understandable but limited detail.
    - Acknowledges program but no specific questions.
    - Examples: "Yes, I saw the brochure", "Not sure if I'm eligible" without follow-up.
    #### 5.1-7.0 → Moderate Engagement
    - Clear communication, polite.
    - Asks general program questions (duration, basic fees, timetable).
    - Examples: "What's the total course fee?", "Can you share eligibility details?".
    #### 7.1-8.5 → High Engagement
    - Clear, professional, decision-focused tone.
    - Multiple specific program-related questions (fee breakdown, syllabus, exams, scholarships).
    - Urgency is present.
    - Examples: "I want to join for the January batch — can you tell me the payment schedule?"
    #### 8.6-10.0 → Exceptional Engagement
    - Very clear, concise, and proactive; lead is driving conversation.
    - Highly targeted, decision-stage questions.
    - Strong urgency or immediate intent.
    - Examples: "I have the fee ready, I just need the final step to register."

    **Important**:
    - Ignore explicit conversion actions (e.g., “Where do I apply?”) when scoring.
    - If no program-related engagement → score ≤ 3.0.
    ---
    ### Step 2 — Decay Factor (0-1.0)
    Base decay factor on recency:
    - 0-7 days old → 1.0
    - 8-14 days old → 0.85
    - 15+ days old → 0.60
    Adjustments:
    - If quality score ≥ 7.0 and date > 14 days old → upgrade decay to 0.85.
    - If quality score < 3.0 → lower decay to 0.60 regardless of date.
    ---
    ### Step 3 — Active Summary
    Explain why the score was assigned in less than 50 words. Include the date for significant points.
    DO NOT mention the quality score or decay factor.

    --- Output format (JSON) ---
     Return ONLY this JSON:
    {{
      "day_scores": {{
        "2025-08-07": {{"quality_score": 9.5, "decay_factor": 1.0}},
        "2025-08-04": {{"quality_score": 7.0, "decay_factor": 0.85}},
        ...
      }},
      "active_summary": [
        "<point 1>",
        "<point 2>",
        "... as many as needed"
      ]
    }}
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only returns JSON."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = safe_llm_invoke(messages)
        json_str = extract_json_from_response(response.content)
        data = json.loads(json_str)
        logger.debug("Active score generated successfully")
        return data
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Error parsing active score JSON: {e}")
        return {
            "active_score": 0, 
            "active_summary": "Error parsing response from LLM. Active signals could not be evaluated."
            }


def compute_final_active_score(day_scores: dict, max_total_score: float = 30.0) -> dict:
    """
    Compute the final normalized active score (max 30), based on daily quality score and decay.
    """
    # Calculate raw score per day
    day_scores = {
        day: {
            **data,
            "actual_score": round(data["quality_score"] * data["decay_factor"], 2)
        }
        for day, data in day_scores.items()
    }

    # Sum raw total
    raw_total = sum(d["actual_score"] for d in day_scores.values())

    # Normalize if needed
    normalization_factor = min(1.0, max_total_score / raw_total) if raw_total > 0 else 0

    for data in day_scores.values():
        data["normalized_score"] = round(data["actual_score"] * normalization_factor, 2)

    final_score = int(round(sum(d["normalized_score"] for d in day_scores.values()), 2))

    return {
        "final_active_score": final_score
    }

def get_final_active_score(summary_object, active_conversations, today_date):

    llm_output = get_active_score(summary_object, active_conversations, today_date)

    day_scores = llm_output.get("day_scores", {})

    active_summary = llm_output.get("active_summary", "No summary available.")

    scoring_result = compute_final_active_score(day_scores)

    return {
        "final_active_score": scoring_result["final_active_score"],
        "active_summary": active_summary
    }

def add_all_scores(data):
    """
    Calculates and returns a total lead score by combining various scores,
    including a compatibility score provided as a separate argument.

    Args:
        summary (dict): A dictionary containing summary information.
        communication_log (str): A string representing the communication log.

    Returns:
        dict: A dictionary containing the individual scores and the total lead score.
    """

    communication_log = data.get("communication_log")
    reference_id = data.get("reference_id")
    lead_id = data.get("lead_id")
    
    try:
        # 1. Get the parsed communication log
        call_summary, other_entries = split_and_reduce_calls(communication_log)
        day_summary = analyze_communication_log(other_entries)
        signals = extract_signals_from_input(day_summary)

        # today_date = datetime.now().date().strftime("%Y-%m-%d")
        day_wise, today_date = summarize_todays_communication(communication_log)

        # 2. Get the passive score
        passive_score_output = get_passive_score(signals["passive_signals"], today_date)
        passive_score = passive_score_output.get("passive_score", 0)
        passive_summary = passive_score_output.get("passive_summary", "")

        # 3. Get the conversion score
        conversion_score_output = get_conversion_score(call_summary, signals["active_conversations"], today_date)
        conversion_score = conversion_score_output.get("conversion_score", 0)
        conversion_summary = conversion_score_output.get("conversion_summary", "")

        # 4. Get the final active score
        active_score_output = get_final_active_score(call_summary, signals["active_conversations"], today_date)
        final_active_score = active_score_output.get("final_active_score", 0)
        active_summary = active_score_output.get("active_summary", "")

        # 5. Calculate the total score
        total_lead_score = passive_score + conversion_score + final_active_score
        logger.debug(f"Total lead score generated successfully")

        # 6. Get the day-wise summary
        day_wise_summary = day_wise.get("day_wise_summary", "")

        if total_lead_score >= 70:
            new_stage = "Hot"
        # Transition to Warm
        elif 40 <= total_lead_score < 70:
            new_stage = "Warm"
        # Transition to Cold
        elif total_lead_score < 40:
            new_stage = "Cold"
        else:
            new_stage = "Unknown"

        breakdown = [
            {
                "component": "passive",
                "score": passive_score,
                "summary": passive_summary
            },
            {
                "component": "conversion",
                "score": conversion_score,
                "summary": conversion_summary,
            },
            {
                "component": "active",
                "score": final_active_score,
                "summary": active_summary
            }
        ]

        logger.info(f"Lead score generated successfully for lead_id: {lead_id}")

        return {
            
            "lead_id": lead_id,
            "status": "COMPLETED",
            "reference_id": reference_id,
            "lead_stage": new_stage,
            "lead_score": total_lead_score,
            "daily_score_summary": day_wise_summary,
            "breakdown": breakdown
        }
    except Exception as e:
        logger.error(f"Failed to calculate lead score for lead_id: {lead_id}. Error: {str(e)}")
        return {
            "lead_id": lead_id,
            "status": "FAILED",
            "reason": str(e)
        }
