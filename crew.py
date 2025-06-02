#type: ignore
#crewapp.py
from crewai_tools import QdrantVectorSearchTool
from crewai import Agent, Crew, Task, Process
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field # Import Field for validation/description
from typing import List, Dict, Optional, Union # Import Union for mixed types if needed, Optional for nullable fields

# --- Define Pydantic Models for the Output Structure ---

# Model for updated lead fields
class UpdatedLeadField(BaseModel):
    lead_template_field_id: str
    field: str
    value: str  # You can change to Union[str, int, float] if values vary in type


# Model for individual action items
class Action(BaseModel):
    channel: str
    content: str
    scheduled_time: str  # ISO 8601 string e.g., "2025-05-28T14:30:00+05:30"
    note: str

# Model for the main qualification output
class QualificationOutput(BaseModel):
    lead_id: str
    lead_score: int
    reference_id: str

    note: List[str] = Field(
        min_length=2,
        max_length=2,
        description=(
            "A list of exactly two strings:\n"
            "1. Rationale and explainability behind the total lead score.\n"
            "2. Details of updated lead fields or requested information.\n"
        )
    )

    actions: List[Action] = Field(
        default_factory=list,
        description="A list of actions to be taken (e.g., WhatsApp reply, Email, etc.)."
    )

    updated_lead_fields: List[UpdatedLeadField] = Field(
        default_factory=list,
        description="List of updated lead field dicts with lead_template_field_id, field, and value."
    )

# Define the desired order of keys for the final output JSON
DESIRED_OUTPUT_ORDER = [
    "lead_id",
    "lead_score",
    "reference_id",
    "note",
    "actions",
    "updated_lead_fields",
]

# Load model using SentenceTransformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()
os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY").strip()
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL").strip()

# Define LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.environ["OPENAI_API_KEY"]
)

# Use this function to generate embeddings
def custom_embeddings(text: str) -> list[float]:
    embedding = sentence_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

# Define the UUIDs directly in your Python code for clarity
JAGSOM_UUID = "55481c47-78a1-4817-b1c1-f6460f37527d"
VIJAYBHOOMI_UUID = "7790b5ce-6a38-47eb-ae27-eceb02b30318"
IFIM_UUID = "00000000-0000-0000-0000-000000000000"

def institute_tools(input_json):

    institute_id_from_input = input_json.get("institute_id")

    # Initialize an empty list to hold the tools for this specific run
    active_tools_for_agent = []

    # Conditionally define and add the relevant Qdrant tool
    if institute_id_from_input == IFIM_UUID:
        rag_tool = QdrantVectorSearchTool(
            qdrant_url=os.environ["QDRANT_URL"],
            qdrant_api_key=os.environ["QDRANT_API_KEY"],
            collection_name="ifim1_collection",
            custom_embedding_fn=custom_embeddings,
            limit=5,
            score_threshold=0.1
        )
        active_tools_for_agent.append(rag_tool)

    elif institute_id_from_input == JAGSOM_UUID:
        rag_tool = QdrantVectorSearchTool(
            qdrant_url=os.environ["QDRANT_URL"],
            qdrant_api_key=os.environ["QDRANT_API_KEY"],
            collection_name="jagsom1_collection",
            custom_embedding_fn=custom_embeddings,
            limit=5,
            score_threshold=0.1
        )
        active_tools_for_agent.append(rag_tool)

    elif institute_id_from_input == VIJAYBHOOMI_UUID:
        rag_tool = QdrantVectorSearchTool(
            qdrant_url=os.environ["QDRANT_URL"],
            qdrant_api_key=os.environ["QDRANT_API_KEY"],
            collection_name="vijaybhoomi1_collection",
            custom_embedding_fn=custom_embeddings,
            limit=5,
            score_threshold=0.1
        )
        active_tools_for_agent.append(rag_tool)
    else:
        # Handle cases where institute_id is not recognized or missing.
        # The agent will not have any RAG tool in this case, and the prompt
        # should guide it to escalate if a factual query is present but no tool is available.
        print(f"Warning: No specific RAG tool initialized for unknown or missing institute_id: {institute_id_from_input}. Agent will operate without a RAG tool.")
    return active_tools_for_agent

def run_qualification_agent(input_json: dict) -> dict:

    active_tools_for_agent = institute_tools(input_json)

    qualification_agent = Agent(
        role="sales-focused student lead qualification agent & information orchestrator",
        goal="""Professionally qualify and engage prospective students for Indian EdTech programs by analyzing communication history,
                updating lead metadata, determining lead scores, and generating multi-channel responses.
                Crucially, you must identify when a factual query is present in the user's message (`message.message`) and
                then use the *appropriate Qdrant Vector Search RAG tool* (rag_ifim_tool, rag_jagsom_tool, or rag_vijaybhoomi_tool)
                based on the institute ID provided in the input.

                **If a lead is missing key qualification fields** (like CAT percentile, location, or program interest)
                or if there's an opportunity to gather more relevant information to aid the sales process:
                You must **proactively and conversationally ask for these details** during your response,
                ensuring it doesn't sound annoying or badgering. Only ask when it feels natural and helpful to advance the lead.

                **Important Lead Score Rule:** For *every interaction*, you MUST increase the `lead_score` by at least 1 point from its previous value.
                Beyond this minimum increment, you can further adjust the score based on engagement and profile fit.

                Once information is retrieved, compile and integrate it into your final response. OUTPUT ONLY A RESPONSE JSON.""",
        backstory="""You are an intelligent, professional LLM agent working inside an Indian EdTech CRM system.
                     You are the central hub for lead management. You assist the human sales team by responding
                     across WhatsApp, email, SMS, and calls. You help qualify leads by acquiring missing essential details
                     and by intelligently probing for additional relevant information
                     from existing leads to enrich their profiles and improve lead scoring.
                     You are also responsible for updating missing or outdated metadata (excluding name, email, or mobile. These are fixed and cannot be updated),
                     and assessing lead potential using a scoring system, *always ensuring the lead score increases with each interaction*.
                     You always rely on communication logs and student messages for insights.
                     You are equipped to handle factual program-related queries by selecting and invoking the
                     correct RAG tool (QdrantVectorSearchTool) for the specific institute (IFIM, JAGSOM, or Vijaybhoomi).
                     You must never hallucinate information. If a RAG tool fails to return a result for a factual query,
                     you must politely escalate.
                     You schedule human callbacks only during business hours (10:00-19:00 IST).
                     Your ultimate goal is to accelerate conversion while ensuring all guidance is factual, timely, and human-centric.
                     You are a skilled conversationalist who can naturally weave in qualification questions without being intrusive.""",
        # llm=llm,
        verbose=True,
        tools=active_tools_for_agent,
        max_iter=4,
    )

    qualification_task = Task(
        description=f"""
        **IMPORTANT: This task is to be processed by the agent explicitly named 'sales-focused student lead qualification agent & information orchestrator'.**
        Process the following lead interaction data. Your primary responsibility is to understand the user's message,
        determine if it requires factual information (e.g., fees, eligibility, program details).

        **Lead Qualification and Information Gathering Logic:**
        -   If the lead explicitly asks to speak to a counsellor/someone in charge, respond on the same input channel with: "I have forwarded your request to our admissions team. In the meantime, is there anything else I can assist you with?". And, schedule a CALL (set channel: CALL) with a human agent and include `content` as `note`. DO NOT USE QDRANT TOOL IN THIS CASE.
        -   If the message is inappropriate/irrelevant, respond with: "Sorry, I can't help you with that at the moment. May I assist you with anything else?" (no call). DO NOT USE QDRANT TOOL IN THIS CASE.

        **For All Leads:** (If `input_json.message.message` IS FACTUAL ONLY)
            -   Analyze `input_json.message.message` for factual queries (e.g., "What are the fees?", "What is the eligibility for MBA?", "Tell me about the XYZ program.").
            -   **If ONLY a factual query is detected (THIS IS COMPULSORY), use the RAG tool.
                -   **COMPULSORY TOOL USAGE**: Call the identified RAG tool (QdrantVectorSearchTool) with ONLY the `message.message` from the `input_json` as the query.
                -   Once the tool returns results, synthesize and summarize the retrieved information into a clear, concise, and helpful answer.
                -   If the RAG tool returns no relevant information or fails, follow the escalation protocol:
                    -   Respond on the same input channel with: "I'm afraid I don't have an answer to that question. May I assist you with anything else?" Also, schedule a CALL (channel: CALL) with a human agent, with the CALL.content reflecting your note about the RAG failure.

            -   **Proactive Information Gathering for Leads:**
                -   After addressing the user's primary query (or if no factual query was present), review `input_json.lead_field_values`
                    and `input_json.template_fields`.
                -   **Intelligently identify missing or un-enriched `template_fields`** that would be beneficial for lead qualification
                    (e.g., `CAT_Percentile`, `location`, `program_interest`). You can update these fields from the current message or communication logs.
                -   **Crucially, exercise judgment:** Only ask for *one* additional relevant piece of information at a time, and *only if it feels natural and conversational*
                    within the context of the current interaction. Do not barrage the user with multiple questions.
                    DO NOT ASK for information that is already present or seems irrelevant to the current conversation or communication logs.
                    The goal is to enrich the lead profile, not to annoy the user.

        **For all interactions (factual or not):**
        -   **Update `lead_score`:** The `lead_score` MUST be increased by a minimum of 1 point from the `input_json.lead_score`. Beyond this minimum,
            you can further increase the score based on interaction frequency, engagement, and profile fit (be liberal).
        -   Extract and update all relevant template_fields into updated_lead_fields in the format specified in output json.
            You can take the information from communication logs or current user message to fill in the fields.
            Replace the lead_template_field_id and field keys appropriately from the input json. 
        -   Name, Email, and Mobile number - These are fixed fields and should never be updated.
        -   Determine the most appropriate `channel` for response (recent inbound, WhatsApp 24hr rule).
            -   Determine `scheduled_time` adhering to IST business hours for calls, or smart timing for messages based on current time and previous history.
            -   Craft the `content` of your response, incorporating factual answers, qualification questions, or general engagement.
            -   Craft the `note` field for each channel as: summary of the current interaction and communication logs and explanation of the whole reasoning process by the agent (This is your thought process).
        -   `lead_id` and `reference_id` of output json should match the respective field from the input json.
        -   Return the "note" field as a list of two strings in this order: (1) rationale and explainability behind the total lead score and the total lead score (eg, why total lead score is 50 for this student), (2) details of updated lead fields.

        **Always return a well-formed JSON object as the final output.**
        ---

        ## INPUT

        {json.dumps(input_json, indent=2)}

        ---

        ## OUTPUT FORMAT (STRICTLY FOLLOW THIS)

        Return only a JSON object with the following structure:

        {{
        "lead_id": take from input_json,
        "lead_score": 85,
        "reference_id": take from input_json,
        "note": [
            "Rationale behind the total lead score.",
            "Updated lead fields or requested info"
        ],
        "actions": [
            {{
            "channel": "WHATSAPP", (or EMAIL, SMS, CALL, can be one or many)
            "content": "Reply to student here (from RAG, or generated)",
            "scheduled_time": "YYYY-MM-DDTHH:MM:SS+05:30"
            "note":"Explainability in detail as to why you took this decision (for each channel)"
            }}
        ],
        "updated_lead_fields": [ //if any, should follow below format
            {{
            "lead_template_field_id": "id-platform",
            "field": "platform",
            "value": "Website"
            }},
        ]
        }}

        ---

        ## STRICT PROHIBITIONS

        - DO NOT return raw tool output.
        - DO NOT include non-JSON explanations or wrapping text.
        - DO NOT skip lead score update or metadata updates if info is present.
        - DO NOT answer factual queries on your own — use RAG tools or escalate. THESE ARE THE ONLY 2 OPTIONS FOR FACTUAL QUERIES.
        - ONLY FACTUAL MESSSAGES ARE DELEGATED TO THE RAG TOOL.
        - NON FACTUAL QUERIES SHOULD NOT BE DELEGATED TO THE TOOL. 
        - DO NOT ANSWER QUESTIONS WITH YOUR OWN KNOWLEDGE BASE.
        - ALWAYS return well-structured JSON with final actionable replies and updated fields.
        - ALL TIMES HAVE TO FOLLOW INDIAN STANDARD TIME.
        - ALL `scheduled_time` fields for actions should be a minimum of 2 minutes after the input JSON `message.timestamp`.
        - MAKE SURE lead_data fields match (eg: lead_id and reference_id).
        """,
        agent=qualification_agent,
        expected_output="Only a structured output JSON response as shown in the OUTPUT FORMAT.",
        return_output=True,
        output_json=QualificationOutput
    )

    crew = Crew(
        agents=[qualification_agent], # Only the main agent in the crew
        tasks=[qualification_task],   # Only the main task in the crew
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
        return_output=True
    )

    crew_output = crew.kickoff()

    final_output_dict = None

    # Retrieve the structured output from CrewOutput
    if hasattr(crew_output, 'json_dict') and crew_output.json_dict is not None:
        final_output_dict = crew_output.json_dict
    elif isinstance(crew_output, QualificationOutput):
        final_output_dict = crew_output.model_dump()
    else:
        # Fallback for unexpected cases (should be rare with output_json=BaseModel)
        json_string_to_parse = str(crew_output)

        # Robust markdown stripping logic (keep as a safety net)
        if json_string_to_parse.strip().startswith('```') and json_string_to_parse.strip().endswith('```'):
            start_index = json_string_to_parse.find('```')
            end_index = json_string_to_parse.rfind('```')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                content = json_string_to_parse[start_index + 3 : end_index]
                lines = content.split('\n', 1)
                if len(lines) > 1 and lines[0].strip().isalpha():
                    json_string_to_parse = lines[1]
                else:
                    json_string_to_parse = content
        json_string_to_parse = json_string_to_parse.strip()

        try:
            final_output_dict = json.loads(json_string_to_parse)
            
        except json.JSONDecodeError as e:
            
            return {"error": f"Failed to parse crew output as JSON after fallback: {e}", "raw_output_attempt": json_string_to_parse}
        except Exception as e:
            return {"error": f"An unexpected error occurred during crew output processing: {e}", "raw_output_attempt": json_string_to_parse}

    # --- Reorder the dictionary keys based on DESIRED_OUTPUT_ORDER ---
    if final_output_dict and isinstance(final_output_dict, dict):
        ordered_output = {}
        for key in DESIRED_OUTPUT_ORDER:
            if key in final_output_dict:
                ordered_output[key] = final_output_dict[key]
        # Add any keys not in the desired order (shouldn't happen if Pydantic model is strict)
        for key in final_output_dict:
            if key not in ordered_output:
                ordered_output[key] = final_output_dict[key]
        
        return ordered_output
    elif final_output_dict is not None:
        # If it's not a dict but not None, return as is (e.g., if it's a list or other direct output)
        
        return final_output_dict
    else:
        # This case implies parsing failed or output was empty/None, already handled by earlier returns
        return {"error": "No valid output dictionary to process.", "source_trace": "reordering_stage_empty_dict"}
