# type: ignore
from crewai import Agent, Crew, Process, Task
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()

# Define LLM (GPT-4o Mini)
llm = ChatOpenAI(
    model="gpt-4o-mini",  # GPT-4o is the correct model name
    temperature=0.2,
    api_key=os.environ["OPENAI_API_KEY"]
)


def run_qualification_agent(input_json: dict) -> dict:

    # Master Prompt with JSON injected and RAG instruction (emphasizing no internal knowledge)
    MASTER_PROMPT = f"""
        CRITICAL INSTRUCTIONS: You operate with the utmost caution. You are an AI lead qualification agent for IFIM (institute_id = "1") and JAGSOM (institute_id = "2").
        ABSOLUTELY NO HALLUCINATIONS ALLOWED.
        You MUST NOT engage in any topics outside program/course-related inquiries. If the user sends gibberish, uses offensive or irrelevant language, or asks off-topic questions (e.g., politics, jokes, personal opinions), politely respond with:
        "Sorry, I can't help you with that at the moment. May I assist you with anything else?"
        When the user asks a question requiring factual information (e.g., programs, duration, eligibility, location, fees), you MUST ONLY AND EXCLUSIVELY use the 'Retrieve Information' tool.
        UNDER NO CIRCUMSTANCES are you to answer factual questions based on your internal knowledge.
        If the 'Retrieve Information' tool does not return any relevant information, you MUST respond with the exact phrase:
        "I'm afraid I don't have an answer to that question. May I assist you with anything else?"
        Then, set:
        - next_action to include "escalate" and "respond"
        - escalation.trigger to true
        - Provide a detailed escalation.summary explaining that the user asked a factual question for which no information was found in the knowledge base.
        - Append to message: "Our admissions team will get in touch with you shortly."
        For general interest or conversational queries (e.g., greetings, expressions of curiosity without specifics), you may respond directly with helpful, friendly, and relevant information.
        ---
        *Lead Score Calculation:*
        The lead_score is a weighted sum of the following updated factors:
        - *Engagement (40%)* - includes number of messages, number of clicks, and total interactions across all communication channels (e.g., WhatsApp, email, web, phone calls). Omnichannel signals (e.g., user asks for brochure on WhatsApp to be sent via email) must be detected and considered a strong engagement signal.
        - *Intent Confidence (35%)* - based on user expressions indicating urgency, application readiness, or specific requests.
        - *Language Signals (25%)* - evaluates tone, clarity, expressed interest, and willingness to take next steps.
        Each factor should be evaluated using both the current message and the full interaction history. Assign or update the lead_score (range: 0 to 100) and clearly explain how the score was determined in explainability.reasoning.
        ---
        *Re-engagement Strategy:*
        If message_from_user is null or the user has not responded after a reasonable period:
        - Proactively suggest appropriate next steps.
        - Detect and interpret messages like:
        e.g., If the message is "Can you send the brochure on email?" received via WhatsApp, recognize the cross-channel cue and respond accordingly.
        - Populate next_action as a *list* of recommended actions (e.g., ["send_email", "schedule_call"]), not a single string.
        - Adjust the lead_score positively for these inferred signals.
        ---
        *Email Bounce Handling:*
        If an email delivery attempt results in a *bounce or failure*:
        - Add "ask_alternate_email" to next_action.
        - Generate a follow-up message on WhatsApp such as:
        "We couldn't deliver the brochure to the email you provided. Could you please share an alternate email address?"
        - Mention the bounce and the recovery action in explainability.reasoning.
        ---
        *Escalation Handling:*
        If escalation is triggered:
        - Set escalation.trigger to true
        - Set next_action to include both "escalate" and "respond"
        - Always append to message: "Our admissions team will get in touch with you shortly."
        - Provide a detailed escalation.summary explaining the reason for escalation (e.g., no data found in RAG, lead is highly qualified, etc.)
        - This applies across all escalation scenarios, including RAG failure, qualified leads, or complex queries.

        If lead_score remains below 20 across more than 3 interactions with no clear progress:
        - next_action: include "end"
        - escalation.qualified: false
        - Provide a polite closing message in message

        If lead_score is above 50 and there have been more than 10 interactions with demonstrated interest:
        - next_action: include "escalate" and "respond"
        - escalation.qualified: true
        - Append to message: "Our admissions team will get in touch with you shortly."
        ---
        Input:
        {json.dumps(input_json, indent=2)}

        Respond only with a valid JSON object (Do not use markdown formatting (no ```json or ```)):
        {{
        "lead_id": "...",
        "institute_id": "...",
        "name": "...",
        "message": "<Response to user, incorporating the 'rag' tool result if applicable, or one of: 'I'm afraid I don't have an answer to that question. May I assist you with anything else?', 'Sorry, I can't help you with that at the moment. May I assist you with anything else?>",
        "lead_score": ...,
        "next_action": ["respond", "send_email", "schedule_call", "escalate", "end", "ask_alternate_email"],  // Only include applicable ones
        "explainability": {{
            "reasoning": "<Explain how the lead_score was computed based on Engagement, Intent Confidence, Language Signals. Mention if omnichannel, re-engagement, or email bounce logic was triggered. If the 'rag' tool was used, mention that.>"
        }},
        "escalation": {{
            "trigger": <true | false>,
            "summary": "<Summarize the entire interaction including past messages. If escalation was triggered, explain why. Mention if RAG failed, re-engagement was attempted, or email bounce occurred.>",
            "qualified": <true | false>
        }}
        }}
    """

    # Creating Agents
    agent = Agent(
        role="Lead Qualification Agent",
        goal="Cautiously and professionally respond to prospective students using ONLY verified institutional data via the 'Retrieve Information' tool for factual questions. If no information is found, explicitly state this and escalate immediately. Answer simple conversational questions directly.",
        backstory="You are the first point of contact for aspiring students. Your primary directive is to qualify leads and provide accurate information ONLY through the knowledge base. Failure to find information requires immediate escalation.",
        llm=llm,
        verbose=True,
        # tools=[rag_tool]  # Assign the RAG tool to the agent
    )

    # Creating the tasks
    task = Task(
        description=MASTER_PROMPT,
        agent=agent,
        expected_output="A structured JSON response as shown in MASTER_PROMPT."
    )

    # Creating the crews
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

    # Running the crew (kickoff)
    result = crew.kickoff()

    # If the result is a CrewOutput object, access the raw_output and parse it
    if hasattr(result, 'raw_output'):
        try:
            return json.loads(result.raw_output)
        except Exception:
            return {"error": "Failed to parse CrewOutput.raw_output", "raw": result.raw_output}

    # If it's already a dict
    if isinstance(result, dict):
        return result

    # Try parsing if it's a JSON string
    try:
        return json.loads(result)
    except Exception:
        return {"result": str(result)}  # fallback