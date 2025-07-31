import whisperx
import os
import uuid
import requests
import warnings
from dotenv import load_dotenv
import re
import json
from langchain_openai import ChatOpenAI
from logging_config import logger
import torch

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()
llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0.3,
   api_key=os.environ["OPENAI_API_KEY"]
)

warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to get Hugging Face auth token from environment variable
hf_token = os.getenv("HF_AUTH_TOKEN")

def get_log_context(jobId=None, fileUrl=None):
    return {
        "jobId": jobId or "UNKNOWN",
        "fileUrl": fileUrl or "UNKNOWN"
    }

def handle_transcription_request(data):
    """
    Parses input JSON and processes the transcription request.

    Args:
        data (dict): Request payload

    Returns:
        dict: Response with job_id, transcription results, and summary.
    """
    try:
        jobId = data.get("jobId")
        fileUrl = data.get("fileUrl") 
        local_filename = f"audio_{uuid.uuid4().hex}.mp3"

        logger.info("Transcription job received", extra=get_log_context(jobId, fileUrl))

        result = process_transcription(fileUrl, local_filename, jobId)

        try:
            # MODIFIED: Check if the returned value is the "Unusable" string
            cleaned_or_unusable_string = assign_speaker_roles(result["conversation"])
            if isinstance(cleaned_or_unusable_string, str) and cleaned_or_unusable_string.startswith("Unusable:"):
                logger.info(f"Transcript rejected", extra=get_log_context(jobId, fileUrl))
                return {
                    "jobId": jobId,
                    "status": "REJECTED",
                    "reason": cleaned_or_unusable_string
                }
            cleaned = cleaned_or_unusable_string # Assign to 'cleaned' if not rejected
            logger.info("Cleaning and speaker roles assigned successfully", extra=get_log_context(jobId, fileUrl))
        except Exception as e:
            logger.error("Error assigning speaker roles and cleaning", extra=get_log_context(jobId, fileUrl))
            raise
        
        try:
            summary = summarize_transcript(cleaned)
            logger.info("Summary generated successfully", extra=get_log_context(jobId, fileUrl))
        except Exception as e:
            logger.error("Error during transcript summarization", extra=get_log_context(jobId, fileUrl))
            raise

        logger.info("Transcription job completed", extra=get_log_context(jobId, fileUrl))

        return {
            "jobId": jobId,
            "status": "COMPLETED",
            "conversation": cleaned,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Failed to handle transcription request. Error: {str(e)}", extra=get_log_context(jobId, fileUrl))
        return {
            "jobId": jobId,
            "status": "FAILED",
            "reason": str(e)
        }
    
    finally: # ADDED: Ensure local_filename is cleaned up here
        if local_filename and os.path.exists(local_filename):
            try:
                os.remove(local_filename)
                logger.info(f"Cleaned up file {local_filename}", extra=get_log_context(jobId, fileUrl))
            except Exception as e:
                logger.warning(f"Failed to delete temp file {local_filename}: {str(e)}", extra=get_log_context(jobId, fileUrl))
        torch.cuda.empty_cache()

def download_audio(audio_url, local_filename, jobId=None):
    try:
        with requests.get(audio_url, stream=True, timeout=20, allow_redirects=True) as r:
            r.raise_for_status()
            
            # Validate content-type
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("audio/"):
                raise Exception(f"Invalid Content-Type: {content_type}")
            
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info(f"Downloaded audio to {local_filename}", extra=get_log_context(jobId, audio_url))

    except requests.exceptions.Timeout as e:
        raise Exception(f"Download failed: Timeout occurred: {str(e)}")
    except requests.exceptions.TooManyRedirects as e:
        raise Exception(f"Download failed: Too many redirects: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Download failed: {str(e)}")
    except IOError as e:
        raise Exception(f"Failed to save audio file: {str(e)}")


def process_transcription(fileUrl, local_filename, jobId=None): 
    try:
        download_audio(fileUrl, local_filename, jobId)
        audio = whisperx.load_audio(local_filename)
        model = whisperx.load_model("small", device="cuda", compute_type="default")
        result = model.transcribe(audio, language="en")

        logger.info("Whisper Transcription Stage 1 completed", extra=get_log_context(jobId, fileUrl))

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda", return_char_alignments=False)

        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device="cuda")
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = sorted(result["segments"], key=lambda x: x["start"])
        conversation = [
            {"speaker": segment.get("speaker", "Unknown"), "text": segment.get("text", "").strip()}
            for segment in segments if segment.get("text", "").strip()
        ]

        logger.info("Speaker Diarization completed", extra=get_log_context(jobId, fileUrl))

        return {
            "conversation": conversation,
            "speaker_blocks": format_speaker_blocks(segments)
        }

    except Exception as e:
        raise Exception(f"Transcription processing failed: {str(e)}")

def format_speaker_blocks(segments):
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
        You are an expert transcriber and editor. Your task is to take a raw sales call transcript generated by **Whisper transcription** between an EdTech **Counsellor** and a **Student**, clean it up, and format it for clarity and readability.

        ---

        ## STEP 1: Transcript Quality Check

        First, assess whether the transcript is meaningful and processable.

        If **any** of the following are true, consider the transcript **Unusable**:

        - The content is **too noisy**, **gibberish**, or **completely broken**.
        - The call contains **only a few disjointed or random words or short phrases** (e.g., "hello", "yeah", "okay", "thank you") with **no coherent context or factual content**.
        - There is **no clear back-and-forth dialogue**, **no questions or answers**, and **no identifiable sales or academic discussion**.
        - It is **not worth storing or analyzing further** due to poor quality or meaningless content.

        In such cases, respond with **ONLY this string** (no JSON output):
        "Unusable: Transcript quality is too poor and incoherent to process."


        If the transcript is **usable**, proceed to STEP 2.

        ---

        ## STEP 2: Clean and Structure the Transcript

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

        ---

        ## Role Identification Guidelines

        ### Counsellor

        - Represents the **EdTech company or University/College**.
        - Asks sales-oriented or qualification questions like:
        - “What's your graduation year?”
        - “Are you looking for a full-time or part-time course?”
        - Shares information about:
        - Course offerings, fees, placements, payment options, program structure, deadlines.
        - Attempts to **guide or persuade** the student to consider or take admission.
        - Often **initiates the conversation** and drives it forward.

        ### Student

        - A **prospective learner** (or sometimes their parent, guardian, family member, or relative).
        - Typically asks **academic or admissions-related questions** such as:
        - “Is this course available online?”
        - “What are the job opportunities?”
        - “What is the fee structure?”
        - Responds to questions about their background, interests, preferences, and availability.
        - May express hesitation, confusion, or interest.

        ---

        ## Important Notes on Input Quality

        - Whisper transcripts may contain **missing speaker labels**, transcription artifacts, or repeated phrases — expect this and correct accordingly.
        - Some parts may be **jumbled, misattributed, or fragmented**. Use contextual clues to make sensible speaker assignments.
        - Do **not** add or assume any new content that wasn't present in the original transcript.

        ---

        ## STEP 3: Output Format (ONLY if the transcript is usable):

        - Output only a **JSON list** (i.e., an array of objects). Each object must contain:
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
        # logger.info(f"Raw response: {response_text}")
        if response_text.startswith("Unusable:"):
            return response_text

        # Clean the response
        json_str = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.IGNORECASE)
        json_str = re.sub(r'^```\s*|\s*```$', '', json_str, flags=re.IGNORECASE)
        
        # Extract JSON object from text
        match = re.search(r'\[.*\]|\{.*\}', json_str, re.DOTALL)
        if match:
            json_str = match.group()
        json_str = json.loads(json_str)
        return json_str
    
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response from LLM: {str(e)}")
    except Exception as e:
        raise Exception(f"assign_speaker_roles and cleaning failed: {str(e)}")

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

      1. **Interest Level:** Assess the student's level of interest in the EdTech offering (choose from: 'highly interested', 'moderately interested', 'undecided', 'low interest', or 'exploring options'). Include a justification based on the conversation.

      2. **Questions Asked:** List the key questions or clarifications ONLY asked by the student to the counsellor—these may relate to the courses, curriculum, fees, admission process, career prospects, etc. DO NOT include questions asked by the counsellor.

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
        # logger.info(f"Raw response: {response_text}")
        
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
        raise Exception(f"summarize_transcript failed: {str(e)}")