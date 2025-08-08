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
import time
import torch

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to get Hugging Face auth token from environment variable
hf_token = os.getenv("HF_AUTH_TOKEN")
if not hf_token:
    logger.warning("HF_AUTH_TOKEN not set. Diarization may fail or be limited.")


# Helper: Return WhisperX model status, model name, device, and CUDA availability
def get_model_status():
    """
    Returns WhisperX model load status with model name, device, and CUDA availability.
    Adds robustness if env vars or models are missing.
    """
    cuda_available = torch.cuda.is_available()
    # Defensive: Use defaults if not set
    model_name = whisper_model_name if 'whisper_model_name' in globals() and whisper_model_name else "unknown"
    device = model_device if model_device else ("cuda" if cuda_available else "cpu")
    return {
        "loaded": all([
            'whisper_model_instance' in globals() and whisper_model_instance,
            'align_model' in globals() and align_model,
            'align_metadata' in globals() and align_metadata,
            'diarization_pipeline' in globals() and diarization_pipeline
        ]),
        "model_name": model_name,
        "device": device,
        "cuda_available": cuda_available
    }

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    logger.error("❌ OPENAI_API_KEY not set. LLM features may not work.")
    openai_key = ""
else:
    openai_key = openai_key.strip()

llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0.3,
   api_key=openai_key
)

# Load WhisperX models globally after env vars are loaded
whisper_model_name = os.getenv("WHISPER_MODEL")
model_device = os.getenv("MODEL_DEVICE")
cuda_available = torch.cuda.is_available()
if model_device and model_device.lower() == "cuda" and not cuda_available:
    logger.warning("MODEL_DEVICE is set to 'cuda' but CUDA is not available. Falling back to 'cpu'.")
    model_device = "cpu"
logger.info(f"Loading WhisperX model '{whisper_model_name}' on device '{model_device}'...")
try:
    whisper_model_instance = whisperx.load_model(whisper_model_name, device=model_device, compute_type="default")
    logger.info("🟢 Whisper model loaded successfully.")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=model_device)
    logger.info("🟢 Alignment model loaded successfully.")
    diarization_pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=model_device)
    logger.info("🟢 Diarization pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Failed to load WhisperX models: {e}")
    whisper_model_instance, align_model, align_metadata, diarization_pipeline = None, None, None, None

def handle_transcription_request(data):
    """
    Parses input JSON and processes the transcription request.

    Args:
        data (dict): Request payload

    Returns:
        dict: Response with job_id, transcription results, and summary.
    """
    jobId = None
    local_filename = None
    try:
        jobId = data.get("jobId")
        fileUrl = data.get("fileUrl")
        student_name = data.get("studentName", "Student") # Get student name, with default
        counsellor_name = data.get("counsellorName", "Counsellor") # Get counsellor name, with default
        institute_name = data.get("instituteName", "Institute") # Get institute name, with default
        local_filename = f"audio_{uuid.uuid4().hex}.mp3"

        logger.info(f"Transcription job received for jobId: {jobId}")
        start_total_time = time.monotonic()
        result, audio_duration_seconds = process_transcription(fileUrl, local_filename)
        
        try:
            # Check if the returned value is the "Unusable" string
            cleaned_or_unusable_string = assign_speaker_roles(result["conversation"], student_name, counsellor_name, institute_name)
            if isinstance(cleaned_or_unusable_string, str) and cleaned_or_unusable_string.startswith("Unusable:"):
                end_total_time = time.monotonic()
                total_processing_time = end_total_time - start_total_time
                logger.info(f"Transcript rejected for jobId: {jobId}. Audio duration: {audio_duration_seconds}s. Total processing time: {total_processing_time:.2f}s")
                return {
                    "jobId": jobId,
                    "status": "REJECTED",
                    "reason": cleaned_or_unusable_string
                }
            cleaned = cleaned_or_unusable_string # Assign to 'cleaned' if not rejected
            logger.debug("Cleaning and speaker roles assigned successfully")
        except Exception as e:
            logger.error("Error assigning speaker roles and cleaning")
            raise

        try:
            # combine consecutive speakers
            combined = combine_consecutive_speakers(cleaned) 
            logger.debug("Speakers combined successfully")
        except Exception as e:
            logger.error("Error during speaker combination")
            raise

        try:
            # Pass the post-processed transcript to the summarizer
            summary = summarize_transcript(combined) 
            logger.debug("Summary generated successfully")
        except Exception as e:
            logger.error("Error during transcript summarization")
            raise
        end_total_time = time.monotonic()
        total_processing_time = end_total_time - start_total_time
        logger.info(f"Transcription job completed for jobId: {jobId}. Audio duration: {audio_duration_seconds}s. Total processing time: {total_processing_time:.2f}s")

        return {
            "jobId": jobId,
            "status": "COMPLETED",
            "conversation": combined, # Return the post-processed version
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Failed to handle transcription request. Error: {str(e)}")
        return {
            "jobId": jobId,
            "status": "FAILED",
            "reason": str(e)
        }

    finally:
        if local_filename and os.path.exists(local_filename):
            try:
                os.remove(local_filename)
                logger.debug(f"Cleaned up local file")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {local_filename}: {str(e)}")
        torch.cuda.empty_cache()

def download_audio(audio_url, local_filename):
    timeout = int(os.getenv("AUDIO_DOWNLOAD_TIMEOUT", "60"))
    try:
        with requests.get(audio_url, stream=True, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()

            # Validate content-type
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("audio/"):
                raise Exception(f"Invalid Content-Type: {content_type}")

            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    except requests.exceptions.Timeout as e:
        raise Exception(f"Download failed: Timeout occurred: {str(e)}")
    except requests.exceptions.TooManyRedirects as e:
        raise Exception(f"Download failed: Too many redirects: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Download failed: {str(e)}")
    except IOError as e:
        raise Exception(f"Failed to save audio file: {str(e)}")


def process_transcription_internal(fileUrl, local_filename):
    download_audio(fileUrl, local_filename)
    audio = whisperx.load_audio(local_filename)
    audio_duration_seconds = len(audio) / 16000

    if not whisper_model_instance:
        raise Exception("WhisperX model not loaded")
    result = whisper_model_instance.transcribe(audio, language="en")
    logger.debug("Whisper Transcription Stage 1 completed")

    if not align_model or not align_metadata:
        raise Exception("Alignment model not loaded")
    result = whisperx.align(result["segments"], align_model, align_metadata, audio, device=model_device, return_char_alignments=False)

    if not diarization_pipeline:
        raise Exception("Diarization pipeline not loaded")
    diarize_segments = diarization_pipeline(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = sorted(result["segments"], key=lambda x: x["start"])
    conversation = [
        {"speaker": segment.get("speaker", "Unknown"), "text": segment.get("text", "").strip()}
        for segment in segments if segment.get("text", "").strip()
    ]
    logger.debug("Speaker Diarization completed")

    return {
        "conversation": conversation,
        "speaker_blocks": format_speaker_blocks(segments)
    }, audio_duration_seconds

def process_transcription(fileUrl, local_filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            return process_transcription_internal(fileUrl, local_filename)
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"process_transcription failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception(f"Transcription process failed after maximum retries")


def format_speaker_blocks(segments):
    content = {}
    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        content.setdefault(speaker, []).append(segment["text"])
    return {speaker: " ".join(lines) for speaker, lines in content.items()}

def safe_llm_invoke(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception("LLM call failed after maximum retries")

def extract_json_from_response(response_text):
    """
    Extracts a JSON object or array from a string that may contain extra text.
    """
    # Use a regex that specifically looks for a JSON array or object
    # The `re.DOTALL` flag is crucial for matching across newlines
    match = re.search(r'```(?:json)?\s*(\[.*\]|\{.*\})\s*```|(\[.*\]|\{.*\})', response_text.strip(), re.DOTALL)
    
    if match:
        # Prioritize the content inside the code block if it exists
        json_str = match.group(1) or match.group(2)
        if json_str:
            return json_str
    
    # If no match, try a more permissive extraction
    first_brace = response_text.find('[')
    if first_brace == -1:
        first_brace = response_text.find('{')
    
    last_brace = response_text.rfind(']')
    if last_brace == -1:
        last_brace = response_text.rfind('}')
        
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return response_text[first_brace : last_brace + 1]
        
    raise ValueError("No valid JSON structure found in the response.")

def assign_speaker_roles(conversation, student_name, counsellor_name, institute_name):
    """
    Automatically determine Student/Counsellor roles and also clean transcript using LLM.

    Args:
        conversation (list): List of conversation turns with speaker and text.
        student_name (str): Name of the student.
        counsellor_name (str): Name of the counsellor.
        institute_name (str): Name of the institute.

    Returns:
        list: List of conversation turns with speaker and text.
    """

    system_prompt = (f"""
        You are an expert transcript editor for EdTech sales calls. Given a raw transcript from Whisper between a Counsellor and a Student, your task is to clean it up, format it for clarity and readability, and assign correct speaker roles.
            Fix any grammatical errors, awkward phrasing, or incomplete sentences while preserving the original meaning. 
            **Important:** Do **not** make drastic changes or rewrite the content beyond necessary corrections.

        ## STEP 1: Transcript Quality Check
        Determine if the transcript is **unusable**.
        Mark it as **unusable** if **all** the following are true.
        - The content is mostly gibberish or broken.
        - There's no coherent dialogue or academic/sales-related conversation.
        - Only contains random greetings or disjointed phrases.
        - Not useful for analysis.
        If so, return exactly:
        "Unusable: Transcript quality is too poor and incoherent to process."
        ---

        ## STEP 2: Clean and Structure the Transcript
        ### Punctuation
        Add periods, commas, question marks, etc., to ensure grammatical correctness.
        ### Grammar
        Fix minor grammatical errors and awkward phrasing. Keep original intent.
        ### Filler Words
        Remove fillers like:
        *um*, *uh*, *like*, *you know*, *basically*, *actually*, *so* (when filler), *right* (when filler), *I mean*
        ---

        ### Proper Noun Correction
        You are provided with the following names:
        - {student_name} : Name of the student
        - {counsellor_name} : Name of the counsellor
        - {institute_name} : Name of the institute/college/university. The counsellor represents this institute.

        If you encounter **misspelled or mis-transcribed variants** of these names in the transcript (e.g., "Shrutha" instead of "Shruti", or "ISM", "IFM" instead of "IFIM"), **correct them using the provided names**.
        ---

        ### Speaker Labels
        Use **"Counsellor"** or **"Student"**.
            Each line must be clearly labeled as either **"Counsellor"** or **"Student"**. The input transcript might have **incorrect or misattributed** speaker assignments**, so you must:
            - **Carefully analyze each line** to correctly determine the speaker based on content and intent.
            - **Reassign roles if they are incorrect**, using the following detailed context:

        ## Role Identification Guidelines

        ## Counsellor
        - Represents the EdTech company or institution.
        - May say:
            > “Are you looking for full-time or part-time?”
            > “Let me explain the course structure...”
        - Shares fees, placements, deadlines, and tries to guide or persuade.
        - Represents the **EdTech company or University/College/Institution**.
        - Introduces themselves as a representative from the particular institution.
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

        ## STEP 3: Output Format (ONLY if the transcript is usable):
        - Output ONLY a **JSON list** (i.e., an array of objects). Each object must contain:
        - "speaker": One of "Counsellor" or "Student"
        - "text": The cleaned, properly punctuated transcript line.

            ### Example Output
            ```json
            [
                {{
                "speaker": "Counsellor",
                "text": "Hello, I'm calling from ABC University regarding your course inquiry."
                }},
                {{
                "speaker": "Student",
                "text": "Hi, yes. I wanted to know more about the online MBA program.
                }}
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
        response = safe_llm_invoke(messages)
        response_text = response.content.strip()

        if response_text.startswith("Unusable:"):
            return response_text

        try:
            json_str = extract_json_from_response(response_text)
            json_data = json.loads(json_str)
            return json_data
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse JSON response from LLM: {str(e)}")
            raise Exception(f"Failed to parse JSON response from LLM: {str(e)}")

    except Exception as e:
        raise Exception(f"assign_speaker_roles and cleaning failed: {str(e)}")


def clean_key_value(data):
    if not isinstance(data, dict):
        # return
        print("Wrong datatype")
    keys_to_remove = []
    for key, value in data.items():
        if isinstance(value, dict):
            clean_key_value(value)
            if not value:
                keys_to_remove.append(key)
        elif value is None or value == "" or value == "None" or value == "None." or value == "none" or value == "none.":
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del data[key]
    return data

def combine_consecutive_speakers(transcript_json):
    if not transcript_json:
        return []
    combined = [transcript_json[0].copy()]  # Start with first entry
    for i in range(1, len(transcript_json)):
        current = transcript_json[i]
        last = combined[-1]
        if current['speaker'] == last['speaker']:
            # Combine text with space separation
            last['text'] += ' ' + current['text']
        else:
            # Add new entry
            combined.append(current.copy())
    return combined

def summarize_transcript(transcript_json):
    """
    Summarizes transcript
    """
    system_prompt = ("""
      You are a sales call analyst for an EdTech company. Your task is to analyze the following transcript between a Counsellor and a Student and provide a structured, two-part summary.
        Summary:
        This is an extremely detailed summary for the whole call transcript between the Counsellor and the Student.
        It will contain all the details and infomation in the transcript.
        **Do not miss any facts or information**
        Key Offerings:
        Extract all the key details and information mentioned in the transcript.
        - LEAD DETAILS - Student's name, general academic background (no specific scores), achievements, and interests. If none, state "None".
        - ACADEMIC DETAILS - Specific scores, percentiles, or qualifications mentioned (e.g., "90% in 12th grade"). If none, state "None".
        - PROGRAM DETAILS - Specific program features, certifications, placements, or curriculum details discussed. If none, state "None".
        - FEE STRUCTURE - Exact costs, payment terms, and scholarship information. If none, state "None".
        - DISCOUNTS - Any specific discounts/scholarships requested or offered. If none, state "None".
        - APPLICATION PROCESS - Steps, requirements, or timelines mentioned for applying. If none, state "None".
        - MISCELLANEOUS - Any other important contextual details, such as location, family background, or unique constraints. If none, state "None".
        Interest Level:
        Choose one: "highly interested", "moderately interested", "undecided", "low interest", or "exploring options".
        Provide a brief, one-sentence reasoning based on their enthusiasm, questions, and expressed intent.
        Questions Asked:
        List the key questions asked by the student only. These are questions that reveal their priorities (e.g., about programs, fees, placement, timing). If none, state "None".
        Objections:
        List any specific concerns or doubts raised by the student (e.g., cost, timing, job relevance). If none, state "None".
        Next Steps:
        List any specific action items discussed or agreed upon. These should be concrete steps (e.g., "Counsellor to share brochure", "Student to submit form", "Schedule follow-up call").  If none, state "None".
        ---
        ### Output Format
        Return ONLY this JSON structure:
        ```json
        {
            "summary": "string",
            "key_offerings" : {
              "lead_details" : "string",
              "academic_details" : "string",
              "program_details" : "string",
              "fee_structure" : "string",
              "discounts" : "string",
              "application_process" : "string",
              "miscellaneous" : "string"
            },
            "interest_level": "string",
            "questions_asked": "string",
            "objections": "string",
            "next_steps": "string"
        }
        ```
        """
    )
    user_prompt = f"""
    Here is the raw transcript to clean:
    {transcript_json}
    output:
    """

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate content using ChatOpenAI (already initialized as 'llm' globally)
        response = safe_llm_invoke(messages)
        response_text = response.content.strip()

       # --- Refactored JSON extraction logic ---
        try:
            json_str = extract_json_from_response(response_text)
            json_data = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse JSON response from LLM: {str(e)}")
            raise Exception(f"Failed to parse JSON response from LLM: {str(e)}")

        # Clean the key value pairs
        json_data = clean_key_value(json_data)
        return json_data
    except Exception as e:
        raise Exception(f"summarize_transcript failed: {str(e)}")