#inactive_decay.py
import math
from datetime import date
from logging_config import inactive_decay_logger as logger

# --- Configuration ---
# The number of days a lead remains in a stage before its score can drop below the floor
HOT_STAGE_DAYS = 10
WARM_STAGE_DAYS = 20
COLD_STAGE_DAYS = WARM_STAGE_DAYS + 1

# The minimum score a lead can have in each stage during its protection window
FLOOR_SCORES = {
    "Hot": 70,
    "Warm": 40,
    "Cold": 0
}

# The rate at which the lead score decays over time
DECAY_RATES = {
    "Hot": 0.0075,
    "Warm": 0.02,
    "Cold": 0.035
}

class Lead:
    def __init__(self, lead_id: str, reference_id: str, lead_score: int, last_engagement_date: date, lead_stage: str, last_stage_change_date: date):
        self.lead_id = lead_id
        self.reference_id = reference_id
        self.lead_score = lead_score
        self.last_engagement_date = last_engagement_date
        self.lead_stage = lead_stage
        self.last_stage_change_date = last_stage_change_date

def parse_date(value: str) -> date:
    """Helper to safely parse ISO date or default to today."""
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError:
        return date.today()

def update_lead_status(data: dict) -> dict:
    """
    Process lead data and calculate new score and stage based on decay rules.
    
    Args:
        data (dict): Dictionary containing lead information.
        
    Returns:
        dict: Dictionary containing lead_id, new_score, and new_stage.
        
    Raises:
        ValueError: If required fields are missing or invalid.
    """
    try:
        lead_id = data.get("lead_id")
        reference_id = data.get("reference_id")
        lead_score = data.get("lead_score", 0)
        last_engagement_date = parse_date(data.get("last_engagement_date", ""))
        last_stage_change_date = parse_date(data.get("last_stage_change_date", ""))
        
        # Infer stage if not provided
        if not data.get("lead_stage"):
            if lead_score >= 70:
                lead_stage = "Hot"
            elif lead_score >= 40:
                lead_stage = "Warm"
            else:
                lead_stage = "Cold"
        else:
            lead_stage = data.get("lead_stage")


        lead = Lead(
            lead_id=lead_id,
            reference_id=reference_id,
            lead_score=lead_score,
            last_engagement_date=last_engagement_date,
            lead_stage=lead_stage,
            last_stage_change_date=last_stage_change_date
        )
        
        new_score, new_stage = calculate_decay(lead, date.today())
        logger.info(f"Lead decay successfully applied for lead_id: {lead_id}")

        return {
            "lead_id": lead_id,
            "status": "COMPLETED",
            "reference_id": reference_id,
            "lead_score": new_score,
            "lead_stage": new_stage
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate lead decay for lead_id: {lead_id}. Error: {str(e)}")
        return {
            "lead_id": lead_id,
            "status": "FAILED",
            "reason": str(e)
        }

def calculate_decay(lead_data: Lead, today: date) -> tuple[int, str]:
    """Internal function to calculate decay based on lead data."""
    # Calculate the number of days since the last engagement and stage change
    days_since_engagement = (today - lead_data.last_engagement_date).days
    days_in_lead_stage = (today - lead_data.last_stage_change_date).days

    # Get the decay rate based on the lead's current stage
    decay_rate = DECAY_RATES.get(lead_data.lead_stage, DECAY_RATES["Warm"])

    # Apply the exponential decay formula to the initial lead score
    decayed_score = lead_data.lead_score * math.exp(-decay_rate * days_since_engagement)
    adjusted_score = round(decayed_score)

    # Prevent negative or inflated scores
    adjusted_score = max(0, min(adjusted_score, 100))

    # Implement stage-specific floor scores and inactivity windows
    floor_score = FLOOR_SCORES.get(lead_data.lead_stage, 0)
    
    # The floor score is only active for the duration of the stage's protection window
    if (lead_data.lead_stage == "Hot" and days_in_lead_stage <= HOT_STAGE_DAYS) or \
       (lead_data.lead_stage == "Warm" and days_in_lead_stage <= WARM_STAGE_DAYS):
        if adjusted_score < floor_score:
            adjusted_score = floor_score

    # Determine the new stage based on the final adjusted score and inactivity
    new_stage = lead_data.lead_stage
    
    # Transition to Hot
    if adjusted_score >= 70:
        new_stage = "Hot"
    # Transition to Warm
    elif 40 <= adjusted_score < 70 and days_in_lead_stage > HOT_STAGE_DAYS:
        new_stage = "Warm"
    # Transition to Cold
    elif adjusted_score < 40 and days_in_lead_stage > WARM_STAGE_DAYS:
        new_stage = "Cold"
        
    return adjusted_score, new_stage