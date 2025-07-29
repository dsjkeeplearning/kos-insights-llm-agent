import whisperx
import os
import uuid
import requests
import logging
import warnings
from dotenv import load_dotenv
import re
import json
from langchain_openai import ChatOpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()
llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0.3,
   api_key=os.environ["OPENAI_API_KEY"]
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to get Hugging Face auth token from environment variable
hf_token = os.getenv("HF_AUTH_TOKEN")

def handle_transcription_request(data):
    """
    Parses input JSON and processes the transcription request.

    Args:
        data (dict): Request payload

    Returns:
        dict: Response with job_id, transcription results, and summary.
    """
    try:
        # Extract required and optional fields from the 'data' dictionary
        jobId = data.get("jobId")
        fileUrl = data.get("fileUrl") 

        # Generate a unique filename
        local_filename = f"audio_{uuid.uuid4().hex}.mp3" #Uncomment this when we get link

        # Transcribe and diarize the audio
        result = process_transcription(fileUrl, local_filename)

        # Assign speaker roles and clean using LLM
        cleaned = assign_speaker_roles(result["conversation"])
        
        summary = summarize_transcript(cleaned)

        return {
            "jobId": jobId,
            "status": "COMPLETED",
            "conversation": cleaned,  # Updated with role names
            "summary": summary
        }

    except Exception as e:
        logger.exception(f"Failed to handle transcription request. Error: {str(e)}")
        return {
            "jobId": jobId,
            "status": "FAILED"
        }

def download_audio(audio_url, local_filename):
    """
    Downloads an audio file from a public URL.

    Args:
        audio_url (str): URL to the audio file
        local_filename (str): Destination path to save the file

    Raises:
        Exception: If the download fails
    """
    try:
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded audio to {local_filename}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading audio: {str(e)}")
        raise Exception(f"Failed to download audio file: {str(e)}")
    except IOError as e:
        logger.error(f"Error saving audio file: {str(e)}")
        raise Exception(f"Failed to save audio file: {str(e)}")

def process_transcription(fileUrl, local_filename): #Uncomment this when we get link
# def process_transcription(fileUrl):
    """
    Main function to handle audio transcription and speaker diarization.

    Args:
        fileUrl (str): URL to download the audio
        local_filename (str): Temporary filename to save audio

    Returns:
        dict: Transcription result containing speaker blocks and conversation
    """
    try:

        download_audio(fileUrl, local_filename) #Uncomment this when we get link
        audio = whisperx.load_audio(local_filename) #Uncomment this when we get link
        # audio = whisperx.load_audio(fileUrl)

        model = whisperx.load_model("small", device="cpu", compute_type="float32")
        # Transcribe the audio
        result = model.transcribe(audio, language="en")
        
        # Align segments
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device="cpu",
                                return_char_alignments=False)

        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token,
            device="cpu"
        )

        # Perform speaker diarization
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Convert to chronological conversation
        segments = sorted(result["segments"], key=lambda x: x["start"])
        conversation = [
            {"speaker": segment.get("speaker", "Unknown"), "text": segment.get("text", "").strip()}
            for segment in segments if segment.get("text", "").strip()
        ]

        return {
            "conversation": conversation,
            "speaker_blocks": format_speaker_blocks(segments)
        }

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise Exception(f"Transcription processing failed: {str(e)}")
        
    finally:  #Uncomment whole block this when we get link
        try:
            if os.path.exists(local_filename):
                os.remove(local_filename)
                logger.info(f"Cleaned up file {local_filename}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {local_filename}: {str(e)}")

def format_speaker_blocks(segments):
    """
    Groups transcript segments by speaker.

    Args:
        segments (list): WhisperX segments with speaker labels

    Returns:
        dict: Mapping of speaker to their spoken text blocks
    """
    content = {}
    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        content.setdefault(speaker, []).append(segment["text"])

    return {speaker: " ".join(lines) for speaker, lines in content.items()}

def assign_speaker_roles(conversation):
    """
    Automatically determine Student/Counsellor roles and also clean transcript using LLM.
    
    Args:
        conversation (list): List of conversation turns with speaker and text.
        
    Returns:
        list: List of conversation turns with speaker and text.
    """

    system_prompt = ("""
        You are an expert transcriber and editor. Your task is to take a raw sales call transcript between an EdTech **Counsellor** and a **Student**, clean it up, and format it for clarity and readability.
        ## Your Specific Instructions:

        ### Punctuation
        Add all necessary punctuation (periods, commas, question marks, etc.) to make sentences grammatically correct and easy to read.
        ### Grammar
        Fix any grammatical errors, awkward phrasing, or incomplete sentences while preserving the original meaning.
        **Important:** Do **not** make drastic changes or rewrite the content beyond necessary corrections.
        ### Filler Words
        Remove common filler words such as:
        - *um*, *uh*, *like*, *you know*, *basically*, *actually*, *so* (when used as a filler), *right* (when used as a filler), *I mean*
        ### Speaker Labels
        Each line must be clearly labeled as either **"Counsellor"** or **"Student"**. However, the input transcript may have **incorrect or missing speaker assignments**, so you must:
        - **Carefully analyze each line** to correctly determine the speaker based on content and intent.
        - **Reassign roles if they are incorrect**, using the following detailed context:

        ## Role Identification Guidelines
        ### Counsellor
        - Represents the **EdTech company or University/College**.
        - Asks sales-oriented or qualification questions like:
          - “What's your graduation year?”
          - “Are you looking for a full-time or part-time course?”
        - Shares information about:
          - Course offerings, fees, placements, payment options, program structures, deadlines.
        - Tries to **guide or persuade** the student to take admission.
        - Often **initiates the conversation** and drives it forward.
        ### Student
        - A **prospective learner** (or occasionally their parent/guardian).
        - Typically asks **academic or admissions-related doubts** such as:
          - “Is this course available online?”
          - “What are the job opportunities?”
          - “What is the fee structure?”
        - Responds to questions about their background, interests, and preferences.
        - May express hesitation or need more clarity before making a decision.
        
        ## Important Notes on Input Quality
        - **Expect speaker labels to be missing or inaccurate**. Your role includes **correcting them** wherever needed.
        - Some parts may be **jumbled, misattributed, or fragmented**. Use contextual clues to make sensible assignments without assuming details not present in the transcript.
        ## Output Format
        Output only a **JSON list** (i.e., an array of objects). Each object must contain:
        - `"speaker"`: One of `"Counsellor"` or `"Student"`
        - `"text"`: The cleaned, properly punctuated transcript line.

        ### Example Output
        ```json
        [
            {
                "speaker": "Counsellor",
                "text": "Hello, I'm calling from ABC University regarding your course inquiry."
            },
            {
                "speaker": "Student",
                "text": "Hi, yes. I wanted to know more about the online MBA program."
            }
        ]
        """)

    user_prompt = f"""
        CONVERSATION:
        {conversation}

        OUTPUT:
        """

    try:
        # Create messages list for ChatOpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate content using ChatOpenAI (already initialized as 'llm' globally)
        response = llm.invoke(messages)
        
        response_text = response.content.strip()
        logger.info(f"Raw response: {response_text}")
        
        # Clean the response
        json_str = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.IGNORECASE)
        json_str = re.sub(r'^```\s*|\s*```$', '', json_str, flags=re.IGNORECASE)
        
        # Extract JSON object from text
        match = re.search(r'\[.*\]|\{.*\}', json_str, re.DOTALL)
        if match:
            json_str = match.group()
        json_str = json.loads(json_str)
        return json_str
        
    except Exception as e:
        logger.error(f"Cleaning and role assignment failed: {str(e)}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
        return {}

def summarize_transcript(conversation):
    """
    Summarize the conversation using LLM.
    
    Args:
        conversation (list): List of conversation turns with speaker and text.
        
    Returns:
        dict: Summary of the conversation with different keys.
    """

    system_prompt = ("""
      You are an expert sales call analyst. Your task is to analyze the following EdTech sales call transcript between a 'Counsellor' and a 'Student' and produce a structured summary. Your summary must strictly cover the following four aspects:

      1. **Interest Level:** Assess the student's level of interest in the EdTech offering (choose from: 'highly interested', 'moderately interested', 'undecided', 'low interest', or 'exploring options'). Include a brief justification based on the conversation.

      2. **Questions Asked:** List the key questions or clarifications the student asked the counsellor—these may relate to the courses, curriculum, fees, admission process, career prospects, etc.

      3. **Objections:** Identify any concerns, hesitations, or objections raised by the student (e.g., cost concerns, lack of time, doubts about course fit, technical challenges).

      4. **Next Steps:** Clearly outline any next steps discussed or agreed upon between the counsellor and student (e.g., follow-up call, sending brochure, reviewing course content, booking a demo, confirming enrollment).

      5. **Output Format:** Return the result **only** as a valid JSON object in the following format (no explanations or extra text):

      ```json
      {
          "interest_level": "string",
          "questions_asked": "string",
          "objections": "string",
          "next_steps": "string"
      }
        Be concise but complete. Base your judgment strictly on the content of the conversation."""
    )

    user_prompt = f"""
        CLEANED CONVERSATION:
        {conversation}

        OUTPUT: 
        """

    try:
        # Create messages list for ChatOpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate content using ChatOpenAI (already initialized as 'llm' globally)
        response = llm.invoke(messages)
        
        response_text = response.content.strip()
        logger.info(f"Raw response: {response_text}")
        
        # Clean the response
        json_str = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.IGNORECASE)
        json_str = re.sub(r'^```\s*|\s*```$', '', json_str, flags=re.IGNORECASE)
        
        # Extract JSON object from text
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            json_str = match.group()
        json_str = json.loads(json_str)
        return json_str
        
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
        return {}
