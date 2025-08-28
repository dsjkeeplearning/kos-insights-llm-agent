import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import warnings
from preprocess import analyze_communication_log, extract_signals_from_input, split_and_reduce_calls, extract_json_from_response, safe_llm_invoke
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

def get_all_score(summary_object, active_conversations, passive_signals, today_date):
    """
    Scores all lead conversations and actions.
    Args:
        summary_object (dict): The dictionary containing the overall lead summary.
        active_conversations (list): A list of dictionaries for each conversation event.
        passive_signals (list): A list of dictionaries for each passive signal event.

    Returns:
        dict: A dictionary with all scores and summaries.
    """
    if not active_conversations and not summary_object and not passive_signals:
        return {
            "conversion_score": 0,
            "conversion_summary": ["No conversion signals found"],
            "passive_score": 0,
            "passive_summary": ["No passive signals found"]
        }

    system_prompt = f"""
    TASK 1:
    (PASSIVE SCORE)
    Input is: passive_signals

    You are a lead engagement evaluator assessing passive signals like link clicks, email opens, and page visits.

    ### Instructions
    Use the following **Passive Engagement Evaluation Rubric** to assign a score between 0-10.
    Award points for each category, then sum them (max 10 points).
    Apply special rules before finalizing.

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

    ### Passive Summary:
    Explanation of why the passive score was assigned.

    OUTPUT FORMAT:
    "passive_score": <integer between 0-10>,
    "passive_summary": [
        "<point 1>",
        "<point 2>",
        "... as many as needed to justify the score"
    ]
    ---

    TASK 2:
    (CONVERSION SCORE)
    Inputs are:
    - **WhatsApp & Email Logs Summary**: active_conversations
    - **Call Logs Summary**: summary_object

    You are a lead qualification assistant. Your job is to find the **single strongest signal** from the above summaries. 
    Go through each summary in detail and find the strongest signal. Also, go through the rules carefully before scoring.

    ### How to Score (choose the HIGHEST applicable; DO NOT add):
    1. **Action Completed (Score 60)**
    **Intent**: The lead has definitively completed a key conversion step. The action is **done**.
    **Examples**: "I have filled the form", "Form submitted", "Completed application", "Admission granted".

    2. **Immediate Intent (Score 55)**
    **Intent**: The lead expresses a strong and immediate commitment to act. The action is **imminent**.
    **Examples**: "Will fill tonight", "Filling now", "Will do today", "Ready to fill", "Sent my documents".

    3. **Readiness to Act (Score 50)**
    **Intent**: The lead is ready to proceed and is asking for the means to do so. The lead is **ready to act**.
    **Examples**: "Send me the form", "Where is the link?", "How do I get the application link?".

    4. **Exploring the Path (Score 45)**
    **Intent**: The lead is actively seeking guidance on the next steps to apply or complete a task. The lead is **exploring the path to action**.
    **Examples**: "How do I apply?", "What's next to apply?", "assist with application process?".

    5. **Soft Future Intent (Score 35-40)**
    **Intent**: The lead shows a general interest in applying at some undefined point in the future.
    **Examples**: "I'll apply soon", "Planning to apply", "Will confirm soon".

    6. **Follow Up with Intent (Score 30)**
    **Intent**: The lead is proactive about continuing the conversation without a clear statement of action.
    **Examples**: "I'll follow up soon", "Planning to follow up", "Will follow up soon", "Said they will follow up".

    7. **General Interest (Score 20)**
    **Intent**: The lead is in an initial information-gathering stage.
    **Examples**: "Requested brochure", "Asked for details", "Can you share the syllabus?".

    8. **Objections/Irrelevant Talk (Score 0)**
    **Intent**: No discernible interest or action related to the program.
    **Examples**: Objections, spam, or off-topic conversation.

    ### Rules:
    - Use only ONE phrase/action — the strongest — from all inputs.
    - **GENERALIZATION**: In case there are other phrases with a similar meaning, understand the **intent** and assign the score accordingly.
    - If the call summary mentions a next step or action item that a **counsellor or agent is to perform on behalf of the lead**, this should be interpreted as a strong signal of the lead's intent. 
    - Regardless of whether communication is written in first-person ("I will apply") or third-person ("Student said she will apply"), interpret the keywords and intent in context and assign the correct score.
    - Cap at **60**.
    - If no valid signal, score = 0.

    ### Conversion Summary:
    Explanation of why the conversion score was assigned.

    OUTPUT FORMAT:
    "conversion_score": 0-60,
    "conversion_summary": [
        "<point 1>",
        "<point 2>",
        "... as many as needed"
    ]

    --- Output format (JSON) COMBINING TASKS 1 AND 2---
    Return ONLY this JSON and NO EXTRA TEXT:
    {{
        "conversion_score": 0-60,
        "conversion_summary": [
            "<point 1>",
            "<point 2>",
            "... as many as needed"
        ],
        "passive_score": <integer between 0-10>,
        "passive_summary": [
            "<point 1>",
            "<point 2>",
            "... as many as needed to justify the score"
        ]
    }}
    """

    user_prompt = f"""
    USE THE BELOW INFORMATION FOR ALL TASKS:

    Today's date: {today_date}

    FOR TASK 1, INPUT DATA:
    passive_signals: {passive_signals}

    FOR TASK 2, INPUT DATA:
    - **WhatsApp & Email Logs Summary**:
    active_conversations: {active_conversations}
    - **Call Logs Summary**:
    summary_object: {summary_object}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = safe_llm_invoke(messages)
        json_str = extract_json_from_response(response.content)
        data = json.loads(json_str)
        logger.debug("Passive and conversion scores generated successfully")
        return data
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Error parsing passive and conversion scores JSON: {e}")
        return {
            "conversion_score": 0,
            "conversion_summary": ["Error parsing response from LLM. Conversion signals could not be evaluated."],
            "passive_score": 0,
            "passive_summary": ["Error parsing response from LLM. Passive signals could not be evaluated."],
            }

def get_active_score(summary_object, active_conversations, today_date):
    """
    Scores active conversations by combining summaries and calculating recency.
    
    Args:
        summary_object (dict): The dictionary containing the overall lead summary.
        active_conversations (list): A list of dictionaries for each conversation event.

    Returns:
        dict: A dictionary with active_summary and day_scores.
    """

    if not active_conversations and not summary_object:
        return {
            "active_summary": ["No active conversations found"],
            "day_scores": {}
        }

    system_prompt = f"""
    ---
    You are an EdTech CRM lead conversation quality evaluator.

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
    
    ### Step 2 — Decay Factor (0-1.0)
    Base decay factor on recency:
    - 0-7 days old → 1.0
    - 8-14 days old → 0.85
    - 15+ days old → 0.60
    Adjustments:
    - If quality score ≥ 7.0 and date > 14 days old → upgrade decay to 0.85.
    - If quality score < 3.0 → lower decay to 0.60 regardless of date.

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

    user_prompt = f"""
    Today's date: {today_date}

    INPUT DATA:
    - **WhatsApp & Email Logs Summary**:
    active_conversations: {active_conversations}
    - **Call Logs Summary**:
    summary_object: {summary_object}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
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
            "day_scores": {},
            "active_summary": ["Error parsing response from LLM. Active signals could not be evaluated."]
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

def get_final_score(summary_object, active_conversations, passive_signals, today_date):
    try:
        llm_output = get_active_score(summary_object, active_conversations, today_date)
        day_scores = llm_output.get("day_scores", {})
        scoring_result = compute_final_active_score(day_scores)

        all_output = get_all_score(summary_object, active_conversations, passive_signals, today_date)

        return {
            "final_active_score": scoring_result["final_active_score"],
            "active_summary": llm_output.get("active_summary", []),
            "conversion_score": all_output.get("conversion_score", 0),
            "conversion_summary": all_output.get("conversion_summary", []),
            "passive_score": all_output.get("passive_score", 0),
            "passive_summary": all_output.get("passive_summary", [])
        }
    except Exception as e:
        logger.error(f"get_final_active_score failed: {e}")
        return {
            "final_active_score": 0,
            "active_summary": ["Scoring failed"],
            "conversion_score": 0,
            "conversion_summary": ["Scoring failed"],
            "passive_score": 0,
            "passive_summary": ["Scoring failed"]
        }

def add_all_scores(data):
    """
    Calculates and returns a total lead score by combining various scores.

    Args:
        data (dict): A dictionary containing the lead data and communication logs.

    Returns:
        dict: A dictionary containing the individual scores and the total lead score.
    """

    reference_id = data.get("reference_id")
    lead_id = data.get("lead_id")
    
    try:
        # Get the parsed communication log
        communication_log = data.get("communication_log", [])
        if not communication_log:
            logger.error("Communication log is empty")
            return {
                "lead_id": lead_id,
                "reference_id": reference_id,
                "status": "FAILED",
                "reason": "Communication log is empty. Please provide valid communication data."
            }

        call_summary, other_entries = split_and_reduce_calls(communication_log)
        full_summary, today_date = analyze_communication_log(other_entries)
        day_summary = full_summary.get("all_summary", "")
        signals = extract_signals_from_input(day_summary)

        # today_date = datetime.now().date().strftime("%Y-%m-%d")
        today_date = today_date or datetime.now().date().strftime("%Y-%m-%d")

        # Get the scores
        active_score_output = get_final_score(
            call_summary,
            signals.get("active_conversations", []),
            signals.get("passive_signals", []),
            today_date
        )

        # Calculate the total score
        total_lead_score = (
            active_score_output["final_active_score"] +
            active_score_output["conversion_score"] +
            active_score_output["passive_score"]
        )

        logger.debug(f"Total lead score calculated successfully")

        if total_lead_score >= 70:
            new_stage = "Hot"
        elif 40 <= total_lead_score < 70:
            new_stage = "Warm"
        elif total_lead_score < 40:
            new_stage = "Cold"
        else:
            new_stage = "Unknown"

        breakdown = [
            {"component": "passive", "score": active_score_output["passive_score"], "summary": active_score_output["passive_summary"]},
            {"component": "conversion", "score": active_score_output["conversion_score"], "summary": active_score_output["conversion_summary"]},
            {"component": "active", "score": active_score_output["final_active_score"], "summary": active_score_output["active_summary"]}
        ]

        logger.info(f"Lead score generated successfully for lead_id: {lead_id}")

        return {
            "lead_id": lead_id,
            "reference_id": reference_id,
            "status": "COMPLETED",
            "lead_stage": new_stage,
            "lead_score": total_lead_score,
            "daily_score_summary": full_summary.get("day_wise_summary", ""),
            "breakdown": breakdown
        }

    except Exception as e:
        logger.error(f"Failed to calculate lead score for lead_id: {lead_id}. Error: {str(e)}")
        return {
            "lead_id": lead_id,
            "reference_id": reference_id,
            "status": "FAILED",
            "reason": str(e)
        }