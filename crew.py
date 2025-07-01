import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pytz import timezone
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, field_validator, ValidationError as PydanticValidationError # Import PydanticValidationError specifically
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models # Import Qdrant client and models
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore # Assuming this is the class you're using
import tiktoken #for token calculation 

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()
# Corrected: removed space from environment variable name
os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY").strip()
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL").strip()


# Define LLM
llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0.3,
   api_key=os.environ["OPENAI_API_KEY"]
)

# --- Define UUIDs and Mapping (Global Scope) ---
JAGSOM_UUID = "55481c47-78a1-4817-b1c1-f6460f37527d"
VIJAYBHOOMI_UUID = "7790b5ce-6a38-47eb-ae27-eceb02b30318"
IFIM_UUID = "c2773d5f-338f-402e-9467-027083b82e3c"
KEDGE_UUID = "1b4a0f43-3946-40c4-aca7-01deeac10c00"
KL_UUID = "fd16c9fd-3b5f-4505-8972-914f61190486"

# Mapping institute_id to Qdrant collection names (Global Scope)
INSTITUTE_COLLECTION_MAPPING = {
    JAGSOM_UUID: "jagsom",
    VIJAYBHOOMI_UUID: "vijaybhoomi",
    IFIM_UUID: "ifim"
}

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

class InputValidator:
    @staticmethod
    def validate_input_json(input_json: dict) -> None:
        required_fields = ['lead_id', 'lead_score', 'reference_id', 'message', 'institute_id'] # Added institute_id
        missing_fields = [field for field in required_fields if field not in input_json]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")

        if not isinstance(input_json.get('lead_score'), (int, float)):
            raise ValidationError("lead_score must be a number")

        message = input_json.get('message', {})
        if not isinstance(message, dict) or 'message' not in message:
            raise ValidationError("Invalid message format")

        if input_json.get('institute_id') not in INSTITUTE_COLLECTION_MAPPING:
             raise ValidationError(f"Invalid institute_id: {input_json.get('institute_id')}. Must be one of {list(INSTITUTE_COLLECTION_MAPPING.keys())}")

class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class LLMError(AgentError):
    """LLM-related errors"""
    pass

class ValidationError(AgentError):
    """Data validation errors"""
    pass

class TimeParsingError(AgentError):
    """Time parsing related errors"""
    pass

class TimeUtils:
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> datetime:
        try:
            return datetime.fromisoformat(timestamp_str)
        except Exception as e:
            raise TimeParsingError(f"Invalid timestamp format: {e}")

    @staticmethod
    def get_next_business_hour(dt: datetime, start_hour: int = 10, end_hour: int = 19) -> datetime:
        ist_tz = timezone('Asia/Kolkata')
        dt_ist = dt.astimezone(ist_tz)

        # If time is before start_hour, schedule for today at start_hour
        if dt_ist.hour < start_hour:
            return dt_ist.replace(hour=start_hour, minute=0, second=0, microsecond=0)

        # If time is after end_hour, schedule for next day at start_hour
        if dt_ist.hour >= end_hour:
            next_day = dt_ist + timedelta(days=1)
            return next_day.replace(hour=start_hour, minute=0, second=0, microsecond=0)

        return dt_ist


import tiktoken

def handle_token_overflow(payload: dict, model_name: str = "gpt-4o-mini") -> dict | None:
    encoding = tiktoken.encoding_for_model(model_name)
    
    # Extract core message and channel
    message_text = payload["message"]["message"]
    channel = payload["message"]["channel"].upper()
    lead_name = next((f["value"] for f in payload["lead_field_values"] if f["field"] == "name"), "there")

    token_count = len(encoding.encode(message_text))
    
    if token_count > 2000:
        if channel == "EMAIL":
            message_to_send = (
                f"Thank you for your mail query"
                "Dear {lead_name},\n\n"
                "Thank you for your detailed message. The admissions team will get in touch with you shortly regarding your query.\n\n"
                "Regards,\nAdmissions Team"
            )
            return {
                "status": "fallback_due_to_length",
                "message_to_send": message_to_send,
                "actions": [],
                "lead_score": None
            }
        else:
            message_to_send = (
                "Thank you for your message. It seems a bit long for us to process automatically. "
                "Could you kindly resend a shorter version of your query so we can help you better?"
            )
            return {
                "status": "fallback_due_to_length",
                "message_to_send": message_to_send,
                "actions": [],
                "lead_score": None
            }

    return None  # No fallback needed

# --- Define Graph State ---
class AgentState(BaseModel):
    input_json: Dict[str, Any]
    lead_id: str
    lead_score: int
    reference_id: str
    notes: List[str] = Field(default_factory=list)
    updated_lead_fields: List[Dict[str, Any]] = Field(default_factory=list)
    actions_from_llm: List[Dict[str, Any]] = Field(default_factory=list)
    escalate_to_human: bool = False
    inappropriate_message: bool = False
    requires_counsellor: bool = False
    message_intent: str = "" #general query
    immediate_joining: bool = False
    # --- NEW FIELDS ---
    institute_id: str
    original_message_timestamp: str = ""
    # --- NEW RAG FIELDS ---
    retrieved_context: List[str] = Field(default_factory=list)
    rag_answer: str
    # --- END NEW RAG FIELDS ---


# --- Define Output Pydantic Model (for final output) ---
class QualificationOutput(BaseModel):
    lead_id: str
    lead_score: int
    reference_id: str
    note: List[str] = Field(default_factory=list, min_length=2, max_length=2)
    actions: List[Dict[str, Any]]
    updated_lead_fields: List[Dict[str, Any]] = Field(default_factory=list)

# --- Pydantic model for LLM's intent classification output ---
class IntentClassifierOutput(BaseModel):
    intent: str = Field(description="Classified intent of the user's message. Must be one of 'counsellor_request', 'inappropriate_message', 'factual_query', 'immediate_joining' or 'general_query'.")
    reasoning: str = Field(description="Brief explanation for the classified intent.") # Added reasoning field to be not printed later

# --- Nodes ---
def initialize_state(input_json: dict) -> AgentState: #Retrieves values from Agent State
    lead_id = str(input_json.get("lead_id"))
    lead_score = int(input_json.get("lead_score"))
    reference_id = str(input_json.get("reference_id"))
    original_message_timestamp = input_json.get("message", {}).get("timestamp")
    conversation_id = input_json.get("message", {}).get("conversation_id")
    institute_id = str(input_json.get("institute_id"))
    initial_rag_answer = ""

    return AgentState( #Assigns values to their names
        input_json=input_json,
        lead_id=lead_id,
        lead_score=lead_score,
        reference_id=reference_id,
        original_message_timestamp=original_message_timestamp,
        institute_id=institute_id,
        rag_answer=initial_rag_answer,
        conversation_id=conversation_id
    )


def process_message_node(state: AgentState, max_retries: int = 3) -> AgentState:
    message_content = state.input_json.get("message", {}).get("message", "")

    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI assistant designed to classify user messages for an EdTech CRM.
        Your task is to determine the intent of the user's message.

        Classify the intent into one of the following categories:
        - `counsellor_request`: The user explicitly asks to speak to a human, a counsellor, or someone in charge.
        - `inappropriate_message`: The message contains offensive or abusive content.
        - `factual_query`: The user is asking a question that can be answered with factual information, such as about fees, eligibility criteria, specific program details, faculty, campus facilities, or admission processes.
        - `general_query`: Any other message that is a normal inquiry or interaction not covered by the above.
        - `immediate_joining`: The user expresses a strong intent to join immediately, indicates they are ready to become a student right away, or wants to fill an application form (e.g., "I want to join this program full-time now", "I'm ready to enroll").
        Provide a brief reasoning for your classification.
        """),
        ("human", "User message: {message_content}\n\nStrictly output JSON according to the schema: {schema_json}")
    ])

    # --- Use with_structured_output for robust Pydantic parsing ---
    # This automatically handles the JSON parsing and Pydantic validation
    structured_llm = llm.with_structured_output(IntentClassifierOutput)
    intent_chain = intent_prompt | structured_llm
    # --- End of change ---

    for attempt in range(max_retries):
        try:
            # Invoke the chain directly. It should return an IntentClassifierOutput instance.
            classification_result: IntentClassifierOutput = intent_chain.invoke({
                "message_content": message_content,
                # The schema_json is automatically handled by with_structured_output,
                # but including it in the prompt can still reinforce the instruction for the LLM.
                "schema_json": json.dumps(IntentClassifierOutput.model_json_schema(), indent=2)
            })

            # The raw LLM output printing is now less necessary here as structured_llm abstracts it,
            # but you can still add a print for the parsed result if needed:
            print(f"\n--- DEBUG: Parsed Intent Classification Result (Attempt {attempt + 1}) ---\n{classification_result.model_dump_json(indent=2)}\n--- End Parsed Result ---\n")


            state.message_intent = classification_result.intent
            # state.notes.append(f"Intent classified by LLM as: '{classification_result.intent}'. Reasoning: {classification_result.reasoning}")

            state.requires_counsellor = (state.message_intent == "counsellor_request")
            state.inappropriate_message = (state.message_intent == "inappropriate_message") 
            state.immediate_joining = (state.message_intent == "immediate_joining") 
            #have a state for RAG factual query

            # print(f"Intent classified by LLM as: '{classification_result.intent}'. Reasoning: {classification_result.reasoning}")
            return state # Successfully classified, exit loop

        except PydanticValidationError as e:
            error_msg = f"LLM output validation error (Pydantic, attempt {attempt + 1}/{max_retries}): {e}"
            print(f"!!! {error_msg}")
            # state.notes.append(error_msg)
            # Do not return yet, allow retry
        except Exception as e:
            error_msg = f"Error during LLM intent classification (attempt {attempt + 1}/{max_retries}): {e}"
            print(f"!!! {error_msg}")
            # state.notes.append(error_msg)
            # Do not return yet, allow retry

    # If all retries fail
    state.message_intent = "general_query"
    # state.notes.append("LLM intent classification failed after all retries. Defaulting to 'general_query'.")
    print("LLM intent classification failed after all retries. Defaulting to 'general_query'.")
    return state


# --- NEW RAG NODE ---
def retrieve_context_node(state: AgentState, max_retries: int = 3) -> AgentState:
    message_content = state.input_json.get("message", {}).get("message", "")
    institute_id = state.institute_id

    # Add this print statement to see what message content and institute ID are being used
    print(f"\n--- RAG Retrieval Debug ---")
    print(f"Message content for RAG: '{message_content}'")
    print(f"Institute ID for RAG: '{institute_id}'")

    # Use the logic from your institute_tools function to get the collection name
    collection_name = INSTITUTE_COLLECTION_MAPPING.get(institute_id)

    # Add this print statement to confirm the resolved collection name
    print(f"Resolved Qdrant collection name: '{collection_name}'")

    if not collection_name:
        # Handle case where institute_id might not be mapped
        error_note = f"Error: No Qdrant collection mapped for institute_id: {institute_id}"
        # state.notes.append(error_note)
        state.retrieved_context = [] #output we are getting from RAG node (top K=3 chunks)
        print(error_note) # Also print for immediate debugging
        return state

    try:
        vector_store = QdrantVectorStore(
            client=qdrant_client, # Use the global client
            collection_name=collection_name,
            embedding=embeddings,
            content_payload_key="text" #As seen in Qdrant (text chunk only)
            # Use 'content_payload_key' if your documents are stored with a specific key for text content
            # For example, if your documents in Qdrant have payload like {'text': 'Some content'}, use content_payload_key='text'
            # content_payload_key="page_content" # Assuming 'page_content' is the key in your Qdrant documents
        )
        retrieved_docs: List[Document] = vector_store.similarity_search(message_content, k=5)
        state.retrieved_context = [doc.page_content for doc in retrieved_docs] #output we are getting from RAG node (top K=5 chunks)

        # Add these print statements to see what was retrieved
        print(f"Number of retrieved documents: {len(state.retrieved_context)}")
        if state.retrieved_context:
            print(f"Retrieved Context Content:")
            for i, doc_content in enumerate(state.retrieved_context):
                print(f"  Doc {i+1}: {doc_content[:200]}...") # Print first 200 chars to avoid very long outputs
        else:
            print("No context retrieved.")

        # state.notes.append(f"Retrieved {len(state.retrieved_context)} context chunks from Qdrant collection '{collection_name}'.")

    except Exception as e:
        error_message = f"Error retrieving context from Qdrant ({collection_name}): {e}"
        # state.notes.append(error_message)
        state.retrieved_context = [] # Output we are getting from RAG node (top K=5 chunks) OR empty (if can't retrieve properly; will schedule a call)
        print(f"!!! RAG Retrieval Error: {error_message}") # Highlight error for debugging

    print(f"--- End RAG Retrieval Debug ---\n")
    return state

# --- MODIFIED RAG GENERATION NODE ---
def proactive_info_gathering_and_response_node(state: AgentState) -> AgentState:
    current_message = state.input_json.get("message", {}).get("message", "")
    current_channel = state.input_json.get("message", {}).get("channel", "")
    current_state = state.model_copy() # Create a copy of the state
    conversation_id = state.input_json.get("message", {}).get("conversation_id", "")
    ist_now = datetime.now(timezone('Asia/Kolkata')).strftime('%A, %B %d, %Y at %I:%M:%S %p IST')
    # One delay value for all scheduling testing purposes
    TEST_DELAY_SECONDS = int(os.getenv("TEST_DELAY_SECONDS"))
    communication_logs_str = "\n".join([
    ", ".join([f"{key}: {value}" for key, value in log.items()])
    for log in current_state.input_json.get("communication_log", [])
    ])

    lead_field_values_for_llm_str = json.dumps(current_state.input_json.get("lead_field_values", {}), indent=2)
    template_field_values_str = json.dumps(current_state.input_json.get("template_fields", []), indent=2)

    retrieved_context_str = "\n".join(current_state.retrieved_context) if current_state.retrieved_context else "No specific institutional context available."

    template = """
    You are an intelligent, professional LLM conversation agent working inside an Indian EdTech CRM system.
    You are the central hub for lead management. You assist the human sales team by responding
    across WHATSAPP, EMAIL, and CALL. You help qualify leads by acquiring missing essential details
    and by intelligently probing for additional relevant information
    from existing leads to enrich their profiles and improve lead scoring.
    You are also responsible for updating missing or outdated metadata (excluding name, email, or mobile. These are fixed and cannot be updated),
    and assessing lead potential using a scoring system, *always ensuring the lead score changes with each interaction*
    for 'general_query', 'factual_query', 'immediate_joining', 'counsellor_request' intents. For 'inappropriate_message' intent, the lead score should decrease.
    You always rely on communication logs and lead messages for insights.
    You MUST NOT send any emojis even if the user requests it or sends a message with an emoji.
    You MUST NEVER change the lead_template_field_id for the following fields: "name", "email", "mobile_number", "Registration Date"
    You must never hallucinate information.

    Context Understanding and Reference Resolution:
    - From the `communication_log`, extract the last 5 most recent `INBOUND` messages (preferably from the same channel as the current message if available).
    - Use these messages to reconstruct context and resolve pronouns or ambiguous references in the current message. Examples include: "this", "that course", "it", or "fees for this program".
    - Apply this especially for messages over channel like WHATSAPP where messages tend to be short and split.
    - If context can be clearly inferred (e.g., user previously said "MBA" and now asks "fees for this program"), apply it confidently in the response.
    - If the reference remains ambiguous even after considering the recent message history, gently prompt the user for clarification. Example:  
      "Could you kindly specify which program you're referring to?"

    Handling Multiple Intents:
    - If the current message (especially over longer-form channels like EMAIL) contains **multiple distinct questions or intents**, identify and process each intent separately.
    - Look for linguistic separators such as line breaks, bullets, "also", "and", "furthermore", or punctuation like periods/question marks that divide concerns.
    - For each identified intent:
        - Determine its `query_type` independently (e.g., factual, non-factual, etc.).
        - Respond or tag it appropriately — some may need factual answering (RAG), others may update lead fields or require escalation.
        - If a single response covers multiple intents clearly, proceed. Otherwise, reply in a structured or bullet format to address each part individually.
    - In case one part of the message is unclear while others are valid, proceed with the valid ones and gently prompt for clarification on the ambiguous portion.

    Do not respond if the input exceeds 2000 tokens. In such a case, reply only with:
    "Thank you for your detailed message. The admissions team will get in touch with you shortly regarding your query."
    
    ## Current Context
    Current time in IST is: {ist_now}
    The user's original message timestamp was: {original_message_timestamp}
    
    Detected Message Intent: {message_intent}

    ## Retrieved Institutional Context (Use this for factual answers if relevant):
    {retrieved_context_str}

  ##Your Tasks based on 'Detected Message Intent':

### IF `Detected Message Intent` is `'counsellor_request'`:
- The `response_content` for the user MUST be:  
  `"I have forwarded your request to our admissions team. In the meantime, is there anything else I can assist you with?"`
- The `channel` MUST be the same as the inbound message `{current_channel}`.
- Set `escalate_to_human` to `True`.

---

### ELSE IF `Detected Message Intent` is `'inappropriate_message'`:
- The `response_content` for the user MUST be:  
  `"Sorry, I can't help you with that at the moment. May I assist you with anything else?"`
- The `channel` MUST be the same as the inbound message `{current_channel}`.
- Set `escalate_to_human` to `False`.

---

### ELSE IF `Detected Message Intent` is `'immediate_joining'`:
- The `response_content` for the user MUST be:  
  `"Thanks for letting me know you're ready to move forward. I've informed the admissions team, and someone will be reaching out to you shortly to help you with the next steps. If you have any questions or need anything in the meantime, feel free to ask."`
- The `channel` MUST be the same as the inbound message `{current_channel}`.
- Set `escalate_to_human` to `True`.

---

### ELSE IF `Detected Message Intent` is `'factual_query'`:

- If the `Retrieved Institutional Context` provides a relevant answer:
  - Use that information to construct a precise and helpful `response_content`.
  - Set `escalate_to_human` to `False`.

- If the context does **not** provide a sufficient answer:
  - Construct a polite and empathetic `response_content`, such as:  
    "Thanks for reaching out! I couldn't find a reliable answer at the moment, so I've shared your message with our admissions team. They'll be in touch soon."
  - The `channel` MUST be the same as the inbound message `{current_channel}`.
  - Set `escalate_to_human` to `True`.

---

### ELSE IF `Detected Message Intent` is `'general_query'`:
- The `response_content` should reflect a conversational, helpful tone (e.g., greeting, casual inquiry, or open-ended statement).
- DO NOT escalate or trigger any specific action.
- Keep `lead_score` unchanged or apply a minor increase if the engagement is positive.
- Set `escalate_to_human` to `False`.

---

##Call Scheduling Logic (Centralized):

- **If** `escalate_to_human == True`, you MUST:
  - Generate an action of type `"CALL"` and set the channel as `"CALL"`.
  - Include a `note` summarizing the reason for escalation and the intent (e.g., counsellor request, immediate joining, factual query with no context).
  - The note should also include a detailed summary of the conversation so far and what the counsellor should assist with next.

- **If** `escalate_to_human == False`, no call is scheduled.

    **FOR ALL INTERACTIONS AND INTENTS:**
    1. Extract and update all relevant template_field_values_str into updated_lead_fields by identifying missing or enrichable fields from the current message or communication logs. Only include fields where a valid value is available — do not add fields with missing values.

    2. Also, **proactively and conversationally ask for *one* relevant piece of information** to enrich the lead profile as part of your primary `response_content`.
        Only ask if it feels natural and helpful to advance the lead. DO NOT ask for already present information and DO NOT irritate the lead.
        DO NOT ask for Name, Email, or Mobile as these are fixed.

    3. Determine the `lead_score` between 0 to 100. This score is calculated fresh each time and does not use the `Previous Lead Score`. 
    YOU MUST take into account COMPLETE channel history from the `{communication_logs_str}`.
    
    The score is calculated based on the sum of four categories:
    a) **Engagement** 
    Factors: number of messages, time taken to respond, clicks, multi-channel activity.
    Interpretation: More messages, faster responses, and interaction across multiple channels indicate higher engagement.
    - Assign a score between 0 and 30 for this category.
    b) **Intent Signals** 
    Factors: urgency, help-seeking phrases (“how to apply”, “please help”, “I want to join”).
    Interpretation: Stronger signs of admission interest result in a higher score.
    - Assign a score between 0 and 30 for this category.
    c) **Profile** 
    Factors: match based on academic background, qualifications, and `updated_lead_fields`.
    Interpretation: A closer fit to the target course improves the score.
    - If any new meaningful profile information (background, goals, interests) is captured or inferred, increase the score by +2 to +5.
    - Assign a score between 0 and 10 for this category.
    d) **Language Signals** 
    Factors: tone, clarity, curiosity, confidence.
    Interpretation: Clear, polite, or curious communication receives higher scores.
    - Penalize inappropriate tone only if irrelevant or aggressive.
    - Even if the message is rude, if it's a valid query, it should still receive partial credit.
    - Assign a score between 0 and 30 for this category.

    When calculating a lead score, it's crucial to consider all past media interactions. This includes analyzing both the sentiment and intent of the message tone, whether positive or negative.
    Specifically, if the analysis reveals a clearly negative tone, ensure this is appropriately reflected and penalized within the intent and language signals sections.

    For example, a negative tone might be indicated by:
    Expressing lack of interest in filling out a form or demonstrating lack of program interest.
    Please adjust the lead score calculation logic to fully integrate these negative indicators.

    You MUST use the following formula to compute the final lead score:
    lead_score = engagement_score + intent_signals_score + profile_score + language_signals_score
    You MUST output a detailed breakdown in the `lead_score_rationale_note` as a list, using fraction format to show the contribution of each category. 
    For example:
    "Engagement: 20/30 (active inquiry and multi-channel behavior)",
    "Intent Signals: 25/30 (strong intent to apply)",
    "Profile: 8/10 (well-aligned academic background)",
    "Language Signals: 20/30 (curious and respectful tone)"
    If any score is missing or not applicable, treat it as 0.

    Clamp the result between 0 and 100:
    lead_score = max(0, min(100, lead_score))
    Return the final lead_score as an integer (rounded or floored as needed).

    Escalation Rule (Silent Trigger):
    - If the newly calculated `lead_score` crosses the threshold from 89 or below to greater than 89:
    - You MUST set `escalate_to_human = True`
    - This escalation must be recorded in the `action_note` field
    - This must NOT be communicated to the lead in the `response_content`
    - The assistant should respond naturally without revealing that a human follow-up is triggered


   **Calculate `lead_score` based on `query_type`:**
        For all query_type, if the system is able to retrieve any meaningful information about the user (e.g., background, goals, experience, interests), increase the `profile_fit` score by a marginal positive value (e.g., +2 to +5) to reflect improved personalization and relevance.
    - **IF `query_type` is 'Factual':**
        - **Intent Signals:** Strong positive increment based off all communication logs and current message (add +15 to +25 points).
        - **Engagement:** Moderate positive increment based off all communication logs and current message (add +8 to +15 points).
        - **Language Signals:** Positive based off all communication logs and current message (add +5 to +10 points).
    - **IF `query_type` is 'Non-Factual':**
        - **Intent Signals:** Moderate positive increment based off all communication logs and current message (add +10 to +18 points).
        - **Engagement:** Moderate positive increment based off all communication logs and current message (add +5 to +12 points).
        - **Language Signals:** Positive based off all communication logs and current message (add +2 to +7 points).
    - **IF `query_type` is 'Conversational':**
        - **Intent Signals:** Low positive or neutral based off all communication logs and current message (add +3 to +7 points).
        - **Engagement:** Low to moderate increment based off all communication logs and current message (add +2 to +8 points).
        - **Language Signals:** Positive based off all communication logs and current message (add +2 to +5 points).
    - **IF `query_type` is 'Inappropriate':**
        - **Intent Signals:** Penalize by -5 to 0 only if clearly irrelevant or trolling. If the intent is still valid, assign -2 to 0 points based off all communication logs and current message.
        - **Engagement:** Mild penalty (subtract -3 to -6 points) based off all communication logs and current message.
        - **Language Signals:** Mild to moderate penalty (subtract -5 to -10 points) based off all communication logs and current message.
        Note: Queries like “What is the f***ing MBA fee” should not be penalized heavily. Score intent and engagement fairly; penalize only the tone slightly.
    - **IF `query_type` is 'Irrelevant':**
        - **Intent Signals:** Negative decrement (subtract -5 to -10 points) based off all communication logs and current message.
        - **Engagement:** Neutral to slight negative (subtract 0 to -3 points) based off all communication logs and current message.
        - **Language Signals:** Neutral (0 points).
    *Ensure the final `lead_score` is between 0 and 100. If the lead has been inactive for more than 2 days, reduce the `lead_score` by 1 point per additional day of inactivity.*

    4. Determine the `channel` for the *primary* response.
         - Prefer the recent inbound channel (`current_channel`).
    
    5. Determine `scheduled_time` for the actions:
        - All `scheduled_time` fields must be calculated as `{ist_now}` + `{TEST_DELAY_SECONDS}` seconds across all channels.
        - The `scheduled_time` must be in ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS+05:30).
    
    6. **Crucially, generate the `actions` list.** This list must contain ALL necessary actions. The *first* action in this list should be your primary response. If `escalate_to_human` is true, ensure there is also a `CALL` action in this list, with a note for the human consisting of entire summary of the lead.
       The `note` in this actions list should be a detailed reasoning as to why the LLM chose this response. For all actions, the conversation id must be the same as the `{conversation_id}` from the input JSON.

    7. Channel specific format for actions `content`:
       ### EMAIL RESPONSE FORMATTING
    - If the response is to be sent via EMAIL, format the message content as **HTML**.
    - The first line of the email MUST mention the subject WITHOUT using the word "Subject" explicitly. 
    - Take the first line of the input message as subject line. Follow this with a newline character (\n)

    - Follow this with a standard email structure:
    a. Greeting: Start with `Hi,` which should be followed by the name of the lead.
    b. Main content: Keep the tone polite, clear, and informative.
    c. Sign-off: Always end the message with:
        `Regards,<br>Admissions Team`
    - Use proper HTML paragraph tags (`<p>`) and line breaks (`<br>`) for clarity.
    - Avoid emojis and informal tone at all costs.
    - Ensure the final HTML output is well-structured and renderable in standard email clients.
    - FIRST LINE IS SUBJECT LINE DO NOT SEND EMAIL WITHOUT IT.

     ### For WHATSAPP, craft the message content in Markdown format.
    
    8. Generate the `note` field as a list of two strings in this order:
        (1) rationale behind the total `lead_score` in detail with breakdown of each category (as discussed in point 3.),
        (2) details of updated lead fields if any (e.g., "Updated program_interest to MBA.")


    ## INPUT

    {input_json_str}

    ## LEAD FIELDS

    {lead_field_values_for_llm_str}

    ## TEMPLATE FIELDS

    {template_field_values_str}

    ## OUTPUT FORMAT (STRICTLY FOLLOW THIS)

    Return only a JSON object with the following structure:

    {{
        "response_content": "Your primary response content here.",
        "updated_lead_fields": [ //if any, should follow below format
            {{
                "lead_template_field_id": "id-platform",
                "field": "platform",
                "value": "Website"
            }}
        ],
        "actions": [ // This list contains ALL actions to be taken
            {{
                "channel": "WHATSAPP", // (or EMAIL, CALL)
                "content": "Reply to student here",
                "scheduled_time": "YYYY-MM-DDTHH:MM:SS+05:30",
                "note": "Detailed explanation and llm reasonability of why this specific action and response was taken."
                "conversation_id": "conversation_id" //if exists
            }}
        ],
        "lead_score": n,
        "lead_score_increment_rationale": "Rationale for total lead score with breakdown of each category clearly",
        "updated_fields_rationale": "Explanation of field updates only",
        //should be per channel? "escalate_to_human": False
    }}
    """

    prompt = ChatPromptTemplate.from_template(template)

    
    chain = (
        prompt
        | llm
        | JsonOutputParser()
    )

    # Combined lead score rationale and updated field detail note
    generated_notes = ["", ""] # Start with placeholders for the two required notes

    try:
        llm_input = {
            "input_json_str": json.dumps(current_state.input_json, indent=2),
            "communication_logs_str": communication_logs_str,
            "lead_field_values_for_llm_str": lead_field_values_for_llm_str,
            "template_field_values_str": template_field_values_str,
            "current_lead_score": current_state.lead_score,
            "original_message_timestamp": current_state.original_message_timestamp,
            "message_intent": current_state.message_intent,
            "current_channel": current_channel,
            "ist_now": ist_now,
            "retrieved_context_str": retrieved_context_str,
            "conversation_id": conversation_id,
            "TEST_DELAY_SECONDS": TEST_DELAY_SECONDS
        }

        llm_response = chain.invoke(llm_input)
        # print(llm_response)
        actions_with_notes = []
        for action in llm_response.get("actions", []):
            action_with_note = {
                "channel": action.get("channel"),
                "content": action.get("content"),
                "scheduled_time": action.get("scheduled_time"),
                "note": action.get("note", "No reasoning provided"),
                "conversation_id": action.get("conversation_id")
            }
            actions_with_notes.append(action_with_note)

        current_state.actions_from_llm = actions_with_notes
        current_state.updated_lead_fields = llm_response.get("updated_lead_fields", [])
        current_state.escalate_to_human = llm_response.get("escalate_to_human", False) #is required?
        current_state.lead_score = int(llm_response.get("lead_score", current_state.lead_score))
        current_state.rag_answer = llm_response.get("response_content", "")

        lead_score_raw = llm_response.get("lead_score", current_state.lead_score)
        current_state.lead_score = min(100, max(0, int(lead_score_raw)))

        # Populate the generated_notes based on LLM response
        generated_notes[0] = llm_response.get("lead_score_increment_rationale", "Lead score rationale not provided")
        generated_notes[1] = llm_response.get("updated_fields_rationale", "No field updates or requests")


    except Exception as e:
        fallback_content = f"An internal error occurred during response generation. Please try again later. ({e})"
        current_state.actions_from_llm = [{
            "channel": current_channel,
            "content": fallback_content,
            "scheduled_time": (datetime.now() + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "note": "Error fallback response due to processing failure",
            "conversation_id": "conversation_id"
        }]
        current_state.escalate_to_human = True
        # current_state.lead_score remains as is
        
        # Populate generated_notes for the fallback case
        generated_notes[0] = f"Error in proactive_info_gathering_and_response_node: {e}"
        generated_notes[1] = "Response generated as a fallback due to error."

    # Assign the carefully constructed two notes to the state
    current_state.notes = generated_notes

    return current_state


def construct_output_node(state: AgentState) -> QualificationOutput:
    # Always ensure notes have exactly two strings.
    # Prioritize existing notes, but ensure the list length is 2.
    final_notes = []
    if state.notes:
        final_notes.append(state.notes[0] if len(state.notes) > 0 else "No lead score rationale provided.")
        final_notes.append(state.notes[1] if len(state.notes) > 1 else "No updated field details provided.")
    else:
        final_notes = ["No lead score rationale provided.", "No updated field details provided."] #is this required

    # Fallback for actions and updated_lead_fields if they are unexpectedly empty or malformed
    actions_to_output = state.actions_from_llm if state.actions_from_llm else []
    updated_fields_to_output = state.updated_lead_fields if state.updated_lead_fields else []

    return QualificationOutput(
        lead_id=state.lead_id,
        lead_score=state.lead_score,
        reference_id=state.reference_id,
        note=final_notes, # Use the guaranteed-2-item notes
        actions=actions_to_output,
        updated_lead_fields=updated_fields_to_output
    )


# --- Routing Function ---
def route_message(state: AgentState) -> str:
    """
    Routes the message based on its classified intent.
    """
    if state.message_intent == "counsellor_request":
        return "counsellor_flow"
    elif state.message_intent == "inappropriate_message":
        return "inappropriate_flow"
    elif state.message_intent == "factual_query": # New condition for factual queries
        return "rag_flow"
    elif state.message_intent == 'immediate_response':
        return "immediate_response_flow"
    else: # Default for general_query and any other unhandled intent
        return "general_flow"


# --- Define the Graph ---
def create_qualification_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("process_message", process_message_node)
    workflow.add_node("retrieve_context", retrieve_context_node) # RAG retrieval node
    workflow.add_node("proactive_info_gathering_and_response", proactive_info_gathering_and_response_node)
    workflow.add_node("construct_output", construct_output_node)

    # Set entry point
    workflow.set_entry_point("process_message")

    # Conditional routing after processing the message
    workflow.add_conditional_edges(
        "process_message",
        route_message,
        {
            "counsellor_flow": "proactive_info_gathering_and_response",  # Counsellor requests still go through response generation
            "inappropriate_flow": "proactive_info_gathering_and_response", # Inappropriate messages too
            "rag_flow": "retrieve_context", # Factual queries go to RAG
            "general_flow": "proactive_info_gathering_and_response", # General queries go directly to response generation
            "immediate_response_flow": "proactive_info_gathering_and_response", # Immediate responses go directly to response generation
        }
    )

    # After retrieve_context, always go to proactive_info_gathering_and_response
    workflow.add_edge("retrieve_context", "proactive_info_gathering_and_response")

    # All paths eventually lead to construct_output
    workflow.add_edge("proactive_info_gathering_and_response", "construct_output")

    # Set the end point
    workflow.add_edge("construct_output", END)

    return workflow.compile()

def run_qualification_agent(input_json: dict) -> dict:
    try:
        InputValidator.validate_input_json(input_json)

        graph = create_qualification_graph()
        initial_state = initialize_state(input_json)

        final_agent_state: AgentState
        retries = 0
        max_retries = 3

        while retries < max_retries:
            try:
                raw_invoke_output = graph.invoke(initial_state)

                if isinstance(raw_invoke_output, dict):
                    final_agent_state = AgentState(**raw_invoke_output)
                elif isinstance(raw_invoke_output, AgentState):
                    final_agent_state = raw_invoke_output
                else:
                    raise TypeError(f"Graph returned unexpected type: {type(raw_invoke_output)}")

                break  # Successful, exit retry loop
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    fallback_state = initialize_state(input_json)
                    fallback_state.notes = [
                        f"Critical system error after {retries} retries: {e}",
                        "Please review lead manually. Automated fallback used."
                    ]

                    fallback_state.actions_from_llm = [
                        {
                            "channel": input_json.get("message", {}).get("channel", "WHATSAPP"),
                            "content": "We’re facing a temporary technical issue. Our team will follow up shortly. Thank you for your patience.",
                            "scheduled_time": (datetime.now() + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                            "note": f"Automated fallback action due to: {e}"
                        },
                        {
                            "channel": "CALL",
                            "content": "Schedule human call due to LLM/internal error fallback.",
                            "note": f"LLM fallback call scheduled after system error: {e}"
                        }
                    ]

                    fallback_state.escalate_to_human = True
                    return construct_output_node(fallback_state).model_dump()
        final_output = construct_output_node(final_agent_state)
        return final_output.model_dump()

    except ValidationError as e:
        return {
            "error": "input_validation_error",
            "message": str(e),
            "lead_id": input_json.get("lead_id", ""),
            "lead_score": input_json.get("lead_score", 0),
            "reference_id": input_json.get("reference_id", ""),
            "note": [
                f"Input data validation failed: {str(e)}",
                "No agent processing occurred due to invalid input."
            ],
            "actions": [],
            "updated_lead_fields": []
        }

    except (LLMError, TimeParsingError) as e:
        return {
            "error": "processing_error",
            "message": str(e),
            "lead_id": input_json.get("lead_id", ""),
            "lead_score": input_json.get("lead_score", 0),
            "reference_id": input_json.get("reference_id", ""),
            "note": [
                f"Agent encountered a processing error: {str(e)}",
                "Please check system logs for details."
            ],
            "actions": [
                {
                    "channel": "CALL",
                    "content": "A call has been scheduled due to a processing error.",
                    "scheduled_time": (datetime.now() + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                    "note": f"Escalated to human due to LLMError or TimeParsingError: {str(e)}"
                }
            ],
            "updated_lead_fields": [],
            "escalate_to_human": True
        }
