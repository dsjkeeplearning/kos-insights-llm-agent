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

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()
# Corrected: removed space from environment variable name
os.environ["QDRANT_DIVINA_API_KEY"] = os.getenv("QDRANT_DIVINA_API_KEY").strip()
os.environ["QDRANT_DIVINA_URL"] = os.getenv("QDRANT_DIVINA_URL").strip()


# Define LLM
llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0.3,
   api_key=os.environ["OPENAI_API_KEY"]
)

# --- Define UUIDs and Mapping (Global Scope) ---
JAGSOM_UUID = "55481c47-78a1-4817-b1c1-f6460f37527d"
VIJAYBHOOMI_UUID = "7790b5ce-6a38-47eb-ae27-eceb02b30318"
IFIM_UUID = "00000000-0000-0000-0000-000000000000"

# Mapping institute_id to Qdrant collection names (Global Scope)
INSTITUTE_COLLECTION_MAPPING = {
    JAGSOM_UUID: "jagsom",
    VIJAYBHOOMI_UUID: "vijaybhoomi",
    IFIM_UUID: "ifim"
}

ist_now = datetime.now(timezone('Asia/Kolkata')).strftime('%A, %B %d, %Y at %I:%M:%S %p IST')

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_DIVINA_URL"],
    api_key=os.environ["QDRANT_DIVINA_API_KEY"]
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
    message_intent: str = "general_query"
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
    note: List[str] = Field(default_factory=list, max_length=2)
    actions: List[Dict[str, Any]]
    updated_lead_fields: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator('note')
    def validate_note_length(cls, v):
        if len(v) != 2:
            raise ValueError("note must contain exactly 2 strings")
        return v

# --- Pydantic model for LLM's intent classification output ---
class IntentClassifierOutput(BaseModel):
    intent: str = Field(description="Classified intent of the user's message. Must be one of 'counsellor_request', 'inappropriate_message', 'factual_query', or 'general_query'.")
    reasoning: str = Field(description="Brief explanation for the classified intent.")

    @field_validator('intent')
    def validate_intent(cls, v):
        if v not in ['counsellor_request', 'inappropriate_message', 'factual_query', 'general_query']:
            raise ValueError("Intent must be 'counsellor_request', 'inappropriate_message', 'factual_query', or 'general_query'")
        return v

# --- Nodes ---

def initialize_state(input_json: dict) -> AgentState:
    lead_id = str(input_json.get("lead_id"))
    lead_score = int(input_json.get("lead_score"))
    reference_id = str(input_json.get("reference_id"))
    original_message_timestamp = input_json.get("message", {}).get("timestamp")
    conversation_id = input_json.get("message", {}).get("conversation_id")
    institute_id = str(input_json.get("institute_id"))
    initial_rag_answer = ""

    return AgentState(
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
            state.notes.append(f"Intent classified by LLM as: '{classification_result.intent}'. Reasoning: {classification_result.reasoning}")

            state.requires_counsellor = (state.message_intent == "counsellor_request")
            state.inappropriate_message = (state.message_intent == "inappropriate_message")

            print(f"Intent classified by LLM as: '{classification_result.intent}'. Reasoning: {classification_result.reasoning}")
            return state # Successfully classified, exit loop

        except PydanticValidationError as e:
            error_msg = f"LLM output validation error (Pydantic, attempt {attempt + 1}/{max_retries}): {e}"
            print(f"!!! {error_msg}")
            state.notes.append(error_msg)
            # Do not return yet, allow retry
        except Exception as e:
            error_msg = f"Error during LLM intent classification (attempt {attempt + 1}/{max_retries}): {e}"
            print(f"!!! {error_msg}")
            state.notes.append(error_msg)
            # Do not return yet, allow retry

    # If all retries fail
    state.message_intent = "general_query"
    state.notes.append("LLM intent classification failed after all retries. Defaulting to 'general_query'.")
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
        state.notes.append(error_note)
        state.retrieved_context = []
        print(error_note) # Also print for immediate debugging
        return state

    try:
        vector_store = QdrantVectorStore(
            client=qdrant_client, # Use the global client
            collection_name=collection_name,
            embedding=embeddings,
            content_payload_key="text"
            # Use 'content_payload_key' if your documents are stored with a specific key for text content
            # For example, if your documents in Qdrant have payload like {'text': 'Some content'}, use content_payload_key='text'
            # content_payload_key="page_content" # Assuming 'page_content' is the key in your Qdrant documents
        )
        retrieved_docs: List[Document] = vector_store.similarity_search(message_content, k=3)
        state.retrieved_context = [doc.page_content for doc in retrieved_docs]

        # Add these print statements to see what was retrieved
        print(f"Number of retrieved documents: {len(state.retrieved_context)}")
        if state.retrieved_context:
            print(f"Retrieved Context Content:")
            for i, doc_content in enumerate(state.retrieved_context):
                print(f"  Doc {i+1}: {doc_content[:200]}...") # Print first 200 chars to avoid very long outputs
        else:
            print("No context retrieved.")

        state.notes.append(f"Retrieved {len(state.retrieved_context)} context chunks from Qdrant collection '{collection_name}'.")

    except Exception as e:
        error_message = f"Error retrieving context from Qdrant ({collection_name}): {e}"
        state.notes.append(error_message)
        state.retrieved_context = []
        print(f"!!! RAG Retrieval Error: {error_message}") # Highlight error for debugging

    print(f"--- End RAG Retrieval Debug ---\n")
    return state

# --- MODIFIED RAG GENERATION NODE ---
def proactive_info_gathering_and_response_node(state: AgentState) -> AgentState:
    current_message = state.input_json.get("message", {}).get("message", "")
    current_channel = state.input_json.get("message", {}).get("channel", "WHATSAPP")
    current_state = state.model_copy() # Create a copy of the state
    conversation_id = state.input_json.get("message", {}).get("conversation_id", "")
    communication_logs_str = "\n".join([
        f"Timestamp: {log.get('timestamp')}, Channel: {log.get('channel')}, Direction: {log.get('direction')}, Content: {log.get('message')}"
        for log in current_state.input_json.get("communication_log", [])
    ])

    lead_field_values_for_llm_str = json.dumps(current_state.input_json.get("lead_field_values", {}), indent=2)
    template_field_values_str = json.dumps(current_state.input_json.get("template_fields", []), indent=2)

    retrieved_context_str = "\n".join(current_state.retrieved_context) if current_state.retrieved_context else "No specific institutional context available."

    template = """
    You are an intelligent, professional LLM conversation agent working inside an Indian EdTech CRM system.
    You are the central hub for lead management. You assist the human sales team by responding
    across WHATSAPP, EMAIL, SMS, and CALL. You help qualify leads by acquiring missing essential details
    and by intelligently probing for additional relevant information
    from existing leads to enrich their profiles and improve lead scoring.
    You are also responsible for updating missing or outdated metadata (excluding name, email, or mobile. These are fixed and cannot be updated),
    and assessing lead potential using a scoring system, *always ensuring the lead score increases with each interaction*
    for 'general_query', 'factual_query', 'counsellor_request' intents. For 'inappropriate_message' intent, the lead score should decrease.
    You always rely on communication logs and student messages for insights.
    You must never hallucinate information.

    ## Current Context
    Current time in IST is: {ist_now}
    The user's original message timestamp was: {original_message_timestamp}
    
    Detected Message Intent: {message_intent}

    ## Retrieved Institutional Context (Use this for factual answers if relevant):
    {retrieved_context_str}

    ## Your Tasks based on 'Detected Message Intent':

    **IF 'Detected Message Intent' is 'counsellor_request':**
    - The `response_content` for the user MUST be: "I have forwarded your request to our admissions team. In the meantime, is there anything else I can assist you with?"
    - The `channel` for this MUST be the same as the inbound message `{current_channel}`.
    - You MUST also generate an action with "channel": "CALL".
    - The `lead_score` MUST be increased, reflecting high intent.
    - Set `escalate_to_human` to `true`.
    - The `note` should reflect the counsellor request and call scheduling, along with total summary of the conversation.

    **ELSE IF 'Detected Message Intent' is 'inappropriate_message':**
    - The `response_content` for the user MUST be: "Sorry, I can't help you with that at the moment. May I assist you with anything else?"
    - The `channel` for the primary response MUST be the same as the inbound message `{current_channel}`.
    - The `lead_score` MUST be decreased.
    - Set `escalate_to_human` to `false`.
    - The `note` should reflect the inappropriate message and score adjustment.

    **ELSE (IF 'Detected Message Intent' is 'general_query' or any other non-specific intent):**
    1. Determine the `response_content` for the primary response to the user.
        - **Leverage {retrieved_context_str} for factual queries.** If the user's message contains a factual query (e.g., about fees, eligibility, specific program details) and the `Retrieved Institutional Context` provides a relevant answer, use that information to construct a precise and helpful `response_content`. DO NOT SCHEDULE A CALL IN THIS CASE.
        - If the user's message contains a factual query that you *cannot* answer with the `Retrieved Institutional Context` (or if no context was retrieved):
            -   The `response_content` for the user MUST be: "I have forwarded your request to our admissions team. In the meantime, is there anything else I can assist you with?"
            -   The `channel` for this MUST be the same as the inbound message `{current_channel}`.
            -   You MUST also generate an action with "channel": "CALL".
                -   The `note` should reflect the counsellor request and call scheduling, along with reasoning of the conversation and response.
            -   Set `escalate_to_human` to `true`.
    
        - Otherwise, if the user's message is a general statement or non factual query, respond appropriately in a conversation style.

    **FOR ALL INTERACTIONS AND INTENTS:**
    1. Identify if any `template_field_values_str` are missing or can be enriched from the current interaction or communication logs. Update these in the `updated_lead_fields` list. Only update if `value` is present. 

        Also, **proactively and conversationally ask for *one* relevant piece of information** to enrich the lead profile as part of your primary `response_content`.
        Only ask if it feels natural and helpful to advance the lead. DO NOT ask for already present information and DO NOT irritate the lead.
        DO NOT ask for Name, Email, or Mobile as these are fixed.

    2. Determine the `lead_score` between 0 to 100. Follow the breakdown below: (consider current message and communication logs for lead score calculation)
                Engagement - 25%  - To include No. of messages, time taken to respond, clicks, multi-channel activity
                Intent Signals - 30% - Urgency, help questions (How to, Apply, please)
                Profile - 15% - Right profile and qualification fit
                Language Signals - 30% - Tone, confidence, clarity, curiosity, positivity
            Calculate the `lead_score` based on these factors. 

    3. Extract and update all relevant `template_field_values_str` from the current message or communication logs into `updated_lead_fields` (fill the `value` field compulsorily). If there is no `value` to fill, do not add it to the list.
    
    4. Determine the `channel` for the *primary* response.
        - Prefer the recent inbound channel (`current_channel`).
    
    5. Determine `scheduled_time` for the actions:
        - All `scheduled_time` fields for actions WHATSAPP should be a minimum of 3 minutes after the input JSON `message.timestamp`.
        - For CALL actions, `scheduled_time` must adhere to IST business hours (10:00-19:00 IST). (make sure to check current time before scheduling)
        - For other actions like SMS and EMAIL, smart timing can be considered (e.g., avoid late night,) (make sure to check current time before scheduling). It should be a minimum of 3 minutes after the input JSON `message.timestamp`
    
    6. **Crucially, generate the `actions` list.** This list must contain ALL necessary actions. The *first* action in this list should be your primary response. If `escalate_to_human` is true, ensure there is also a `CALL` action in this list, with a note for the human consisting of summary of the lead.
       The `note` in this actions list should be a detailed reasoning as to why the LLM chose this response.
    
    7. Channel specific format for `content`:
       For EMAIL, craft the message content in html format. The first line should be the subject line content only followed by a newline. Then the actual email content should follow, in html style and regular email format. The closing greeting should be from Admissions Team for every email.
       For SMS and whatsapp, craft the message content in Markdown format.
    
    8. Generate the `note` field as a list of two strings in this order:
        (1) rationale behind the total `lead_score` in detail with breakdown of each category,
        (2) details of updated lead fields if any (e.g., "Updated program_interest to MBA.")

    9. For all actions, the conversation id must be the same as the `{conversation_id}` from the input JSON.

    ## INPUT

    {input_json_str}

    ## COMMUNICATION LOGS

    {communication_logs_str}

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
                "channel": "WHATSAPP", // (or EMAIL, SMS, CALL)
                "content": "Reply to student here",
                "scheduled_time": "YYYY-MM-DDTHH:MM:SS+05:30",
                "note": "Detailed explanation and llm reasonability of why this specific action and response was taken."
                "conversation_id": "conversation_id"
            }}
        ],
        "lead_score": n,
        "lead_score_increment_rationale": "Rationale for total lead score with breakdown of each category clearly",
        "updated_fields_rationale": "Explanation of field updates only",
        "escalate_to_human": false
    }}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        prompt
        | llm
        | JsonOutputParser()
    )

    # Initialize notes for this node's output.
    # Any notes from previous nodes are part of the 'state' input and are handled by that node.
    # This node is responsible for generating *its own* two notes.
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
        }

        llm_response = chain.invoke(llm_input)

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
        current_state.escalate_to_human = llm_response.get("escalate_to_human", False)
        current_state.lead_score = int(llm_response.get("lead_score", current_state.lead_score))
        current_state.rag_answer = llm_response.get("response_content", "")

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
            "conversation_id": "null"
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
        final_notes.append(state.notes[0] if len(state.notes) > 0 else "No primary note provided.")
        final_notes.append(state.notes[1] if len(state.notes) > 1 else "No secondary note provided.")
    else:
        final_notes = ["No specific notes provided by agent.", "Review lead for manual action."]

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
        }
    )

    # After retrieve_context, always go to proactive_info_gathering_and_response
    workflow.add_edge("retrieve_context", "proactive_info_gathering_and_response")

    # All paths eventually lead to construct_output
    workflow.add_edge("proactive_info_gathering_and_response", "construct_output")

    # Set the end point
    workflow.add_edge("construct_output", END)

    return workflow.compile()


# --- Main function to run the agent ---
def run_qualification_agent(input_json: dict) -> dict:
    try:
        InputValidator.validate_input_json(input_json)

        graph = create_qualification_graph()
        initial_state = initialize_state(input_json)

        final_agent_state: AgentState # Declare type hint

        try:
            # LangGraph's invoke returns the final state of the graph.
            # If an error happens *within* the graph, LangGraph might still
            # return an incomplete state or raise its own exception.
            # We must explicitly cast or handle what it returns.
            raw_invoke_output = graph.invoke(initial_state)

            # Try to convert whatever LangGraph returns into an AgentState.
            # If it's already AgentState, great. If it's a dict (like AddableValuesDict),
            # try to instantiate AgentState from it. This will validate against AgentState schema.
            if isinstance(raw_invoke_output, dict):
                final_agent_state = AgentState(**raw_invoke_output)
            elif isinstance(raw_invoke_output, AgentState):
                final_agent_state = raw_invoke_output
            else:
                # If it's neither, it's an unexpected type, raise error to fall into outer except
                raise TypeError(f"Graph invoked returned unexpected type: {type(raw_invoke_output)}. Expected AgentState or dict.")

        except Exception as e:
            # This is the critical fallback for any errors during graph execution.
            print(f"Error during graph execution (caught in run_qualification_agent_langgraph main try block): {e}")

            # Create a simple, guaranteed-valid fallback state from the initial input
            fallback_state = initialize_state(input_json) # Re-initialize for cleanliness

            # Ensure notes are compliant for QualificationOutput
            fallback_state.notes = [
                f"Critical system error during processing: {e}",
                "Please review lead manually. Automated response generated as fallback."
            ]

            # Populate fallback actions
            fallback_state.actions_from_llm = [{
                "channel": input_json.get("message", {}).get("channel", "WHATSAPP"),
                "content": "We are experiencing an internal system error. Please try again later, or contact support if the issue persists.",
                "scheduled_time": (datetime.now() + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                "note": f"Automated fallback due to system error: {e}"
            }]
            fallback_state.escalate_to_human = True
            
            # Construct and return the output from this guaranteed-valid fallback state
            final_output = construct_output_node(fallback_state)
            return final_output.model_dump()

        # If we successfully made it here, final_agent_state should be a valid AgentState.
        final_output = construct_output_node(final_agent_state)
        return final_output.model_dump()

    except ValidationError as e:
        # Input validation error (your custom ValidationError)
        return {
            "error": "input_validation_error",
            "message": str(e),
            "lead_id": input_json.get("lead_id", ""),
            "lead_score": input_json.get("lead_score", 0),
            "reference_id": input_json.get("reference_id", ""),
            "note": [f"Input data validation failed: {str(e)}", "No agent processing occurred due to invalid input."],
            "actions": [],
            "updated_lead_fields": []
        }
    except (LLMError, TimeParsingError) as e:
        # Specific anticipated processing errors not caught by the graph's internal handlers
        return {
            "error": "processing_error",
            "message": str(e),
            "lead_id": input_json.get("lead_id", ""),
            "lead_score": input_json.get("lead_score", 0),
            "reference_id": input_json.get("reference_id", ""),
            "note": [f"Agent encountered a processing error: {str(e)}", "Please check system logs for details."],
            "actions": [],
            "updated_lead_fields": []
        }