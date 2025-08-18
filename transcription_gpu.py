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
    logger.info("Whisper model loaded successfully.")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=model_device)
    logger.info("Alignment model loaded successfully.")
    diarization_pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=model_device)
    logger.info("Diarization pipeline initialized successfully.")
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
        fileData = data.get("fileData")
        student_name = data.get("studentName", "Student")
        counsellor_name = data.get("counsellorName", "Counsellor")
        institute_name = data.get("instituteName", "Institute")

        logger.info(f"Transcription job received for jobId: {jobId}")
        start_total_time = time.monotonic()

        # Step 1: Get transcription result
        try:
            if fileUrl:
                local_filename = f"audio_{uuid.uuid4().hex}.mp3"
                result, audio_duration_seconds = process_transcription(fileUrl, local_filename)
                conversation_data = result["conversation"]
            elif fileData:
                audio_duration_seconds = None
                conversation_data = fileData
            else:
                raise ValueError("Either 'fileUrl' or 'fileData' must be provided.")
        except Exception as e:
            logger.error("Error during transcription step"); raise

        # Step 2: Assign speaker roles & clean
        try:
            cleaned_or_unusable_string = assign_speaker_roles(
                conversation_data, student_name, counsellor_name, institute_name
            )
            if isinstance(cleaned_or_unusable_string, str) and cleaned_or_unusable_string.startswith("Unusable:"):
                end_total_time = time.monotonic()
                logger.info(f"Transcript rejected for jobId: {jobId}. Audio duration: {audio_duration_seconds}s. Total processing time: {end_total_time - start_total_time:.2f}s")
                return {
                    "jobId": jobId,
                    "status": "REJECTED",
                    "reason": cleaned_or_unusable_string
                }
            cleaned = cleaned_or_unusable_string
            logger.debug("Cleaning and speaker roles assigned successfully")
        except Exception as e:
            logger.error("Error assigning speaker roles and cleaning"); raise

        # Step 3: Combine consecutive speakers
        try:
            combined = combine_consecutive_speakers(cleaned)
            logger.debug("Speakers combined successfully")
        except Exception as e:
            logger.error("Error during speaker combination"); raise

        # Step 4: Summarize
        try:
            summary = summarize_transcript(combined)
            logger.debug("Summary generated successfully")
        except Exception as e:
            logger.error("Error during transcript summarization"); raise

        # Step 5: Call Score
        try:
            score = score_call(combined)
            logger.debug("Score generated successfully")
        except Exception as e:
            logger.error("Error during transcript scoring"); raise

        # Done
        end_total_time = time.monotonic()
        logger.info(f"Transcription job completed for jobId: {jobId}. Audio duration: {audio_duration_seconds}s. Total processing time: {end_total_time - start_total_time:.2f}s")

        return {
            "jobId": jobId,
            "status": "COMPLETED",
            "conversation": combined,
            "summary": summary,
            "score": score
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
            - **Reassign roles IF AND ONLY IF they are incorrect**, using the following detailed context:

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

        Key Details:
        Extract all the key details and information mentioned in the transcript. 
        - LEAD DETAILS - Student's name, general academic background (no specific scores), achievements, and interests. If none, do not return the field.
        - ACADEMIC DETAILS - Specific scores, percentiles, or qualifications mentioned (e.g., "90% in 12th grade"). If none, state "None".
        - PROGRAM DETAILS - Specific program features, certifications, placements, or curriculum details discussed. If none, state "None".
        - FEE STRUCTURE - Exact costs, payment terms, and scholarship information. If none, state "None".
        - DISCOUNTS - Any specific discounts/scholarships requested or offered. If none, state "None".
        - APPLICATION PROCESS - Steps, requirements, or timelines mentioned for applying. If none, state "None".
        - MISCELLANEOUS - Any other important contextual details, such as location, family background, or unique constraints. If none, state "None".

        Interest Level:
        Choose one: "highly interested", "moderately interested", "undecided", "low interest", or "exploring options".
        
        Concerns:
        List any significant questions, doubts, hesitations or objections expressed by the student that are relevant to their decision-making.
        Ignore trivial questions such as greetings, introductions, or queries about the counsellor/institution name.
        Include concerns related to program details, fees, placements, career prospects, timelines, or other decision-impacting factors.
        If none, state "None".

        Next Steps:
        List any specific action items discussed or agreed upon. These should be concrete steps (e.g., "Counsellor to share brochure", "Student to submit form", "Schedule follow-up call").  If none, state "None".
        
        ---
        ### Output Format
        Return ONLY this JSON structure:
        ```json
        {
            "summary": "string",
            "key_details" : {
              "lead_details" : "string",
              "academic_details" : "string",
              "program_details" : "string",
              "fee_structure" : "string",
              "discounts" : "string",
              "application_process" : "string",
              "miscellaneous" : "string"
            },
            "interest_level": "string",
            "concerns": "string",
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

def get_performance_rating(score):
    if score >= 75:
        return "Excellant"
    elif score >= 50:
        return "Good"
    elif score >= 30:
        return "Average"
    elif score >= 10:
        return "Below Average"
    else:
        return "Poor"

def score_call(transcript_json):
    """
    Scores the call based on predefined categories and weights
    """

    # Predefined weights for each category (sum should ideally be 1.0)
    CATEGORY_WEIGHTS = {
        "Opening & Rapport": 0.15,
        "Solution Alignment": 0.20,
        "Objection Handling": 0.20,
        "Closing Technique": 0.15,
        "Talk-to-Listen Ratio": 0.10,
        "Call Duration Appropriateness": 0.05,
        "Follow-up Commitments": 0.15
    }

    system_prompt = """
    You are an evaluator for a university marketing and sales team's call performance.
    Analyze the provided call transcript and score the rep's performance based on the metrics
    and rubrics below. The goal is to evaluate **quality of execution** regardless of whether
    the student intends to enroll.

    For each metric:
    - Give a score from 0-5 (0 = very poor, 5 = excellent) using the rubrics.
    - Provide the reasoning for the score.
    - If a scenario did not occur (e.g., no objections), score based on demonstrated ability/readiness.
    
    Metrics & Rubrics
    =================
    1. Opening & Rapport
    Definition: Quality of greeting, tone, professionalism, and rapport building.
    Rubric:
    0 = No greeting; abrupt or robotic.
    1 = Minimal greeting; poor tone.
    2 = Basic greeting; lacks warmth or personalization.
    3 = Polite and clear greeting; some rapport but minimal personalization.
    4 = Warm, confident, and builds some rapport.
    5 = Highly engaging, personal connection, sets strong positive tone.

    2. Solution Alignment
    Definition: Demonstrates knowledge of programs and connects to prospect needs.
    Rubric:
    0 = No program knowledge; irrelevant details.
    1 = Very limited program knowledge; mostly generic.
    2 = Some program info; weak connection to needs.
    3 = Adequate knowledge; some relevant alignment.
    4 = Strong knowledge; aligns well to stated needs.
    5 = Excellent depth; highly tailored and persuasive.

    3. Objection Handling
    Definition: Listens to, acknowledges, and resolves concerns effectively.
    Rubric:
    0 = Ignores objections entirely.
    1 = Acknowledges but provides weak or irrelevant response.
    2 = Addresses partially; leaves doubt.
    3 = Adequate handling; provides reasonable reassurance.
    4 = Strong handling with relevant examples or solutions.
    5 = Masterful handling; overcomes concern fully and builds confidence.
    Note: If no objections occur, score based on demonstrated readiness or proactive clarification.

    4. Closing Technique
    Definition: Guides conversation toward next step without being pushy.
    Rubric:
    0 = Ends abruptly with no next step.
    1 = Vague closing; no clear action.
    2 = Suggests possible step but not confirmed.
    3 = Clear next step but lacks strong close.
    4 = Confident close; confirms commitment or timeline.
    5 = Strong, natural close; locks in next step and commitment.

    5. Talk-to-Listen Ratio
    Definition: Was the balance between speaking and listening appropriate for the prospect's needs and engagement level? In discovery phases, more listening is expected; during explanation phases, more speaking may be appropriate. The score should reflect whether the counselor adjusted naturally to the flow of the conversation, rather than aiming for a fixed percentage.
    Rubric:
    0 = Very poor adjustment; rep talks excessively or barely at inappropriate times.
    1 = Poor balance with little responsiveness to call context.
    2 = Somewhat appropriate but occasionally mismatched talk/listen times.
    3 = Generally appropriate; minor mismatches.
    4 = Good adjustment; mostly follows prospect needs.
    5 = Excellent adjustment; naturally varies talk/listen to optimize engagement.

    6. Call Duration Appropriateness
    Definition: Whether length matched complexity and stage.
    Rubric:
    0 = Extremely short or unnecessarily long; major impact.
    1 = Poor pacing; misses key topics or overly drawn out.
    2 = Slight mismatch in timing.
    3 = Adequate duration; minor improvement possible.
    4 = Good pacing; all points covered.
    5 = Perfect pacing; efficient and thorough.

    7. Follow-up Commitments
    Definition: Clear outline of next steps and confirmation of follow-up method/timeline.
    Rubric:
    0 = No follow-up mentioned.
    1 = Vague promise without specifics.
    2 = Follow-up suggested but unclear method/timeline.
    3 = Clear follow-up but lacks confirmation.
    4 = Strong follow-up; confirms method/timeline.
    5 = Excellent follow-up; sets expectation and shows accountability.


    After scoring all metrics:
    1. Identify the **top-performing areas** (scores 4-5) and summarize them in a section called **"What Went Well"**.
    2. Identify the **lower-performing areas** (scores 0-2, or any notable weaknesses) and summarize them in a section called **"Areas of Improvement"**.
    3. If scores are mostly **average (around 3)**:
    - In **What Went Well**, highlight consistency and any relatively stronger qualitative behaviors, even if all numeric scores are similar.
    - In **Areas of Improvement**, point out opportunities to refine adequate performance into excellence.
    4. If scores are mostly **high (4-5)**:
    - In **What Went Well**, highlight standout skills and strengths.
    - In **Areas of Improvement**, suggest micro-refinements, missed opportunities, or small optimizations to further improve.
    5. If scores are mostly **low (0-2)**:
    - In **What Went Well**, find and acknowledge any positive elements, even if small (e.g., polite greeting, attempt at explanation).
    - In **Areas of Improvement**, focus on the most critical foundational gaps first.
    6. Always ensure both sections are meaningful and non-empty.
    7. For each of these sections, produce a short summary, which is 1-2 concise sentences (40-50 words) suitable for compact UI display.

    ** The Output should be in this JSON format ONLY. Do not add any extra text.**
    {
    "breakdown": [
        {
            "category": "Opening & Rapport",
            "score": INT,
            "reason": "String"
        },
        {
            "category": "Solution Alignment",
            "score": INT,
            "reason": "String"
        },
        {
            "category": "Objection Handling",
            "score": INT,
            "reason": "String"
        },
        {
            "category": "Closing Technique",
            "score": INT,
            "reason": "String"
        },
        {
            "category": "Talk-to-Listen Ratio",
            "score": INT,
            "reason": "String"
        },
        {
            "category": "Call Duration Appropriateness",
            "score": INT,
            "reason": "String"
        },
        {
            "category": "Follow-up Commitments",
            "score": INT,
            "reason": "String"
        }
        ],
        "what_went_well": "<Summary of strengths>",
        "areas_of_improvement": "<Summary of areas for improvement>"
    }
    """

    user_prompt = f"""
    Here is the raw transcript to evaluate:
    {transcript_json}
    output:
    """

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Get response from LLM
        response = safe_llm_invoke(messages)
        response_text = response.content.strip()

        # Extract JSON
        try:
            json_str = extract_json_from_response(response_text)
            llm_output = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse JSON response from LLM: {str(e)}")
            raise Exception(f"Failed to parse JSON response from LLM: {str(e)}")

        # --- Prepare breakdown list & calculate call_score ---
        breakdown_sorted = []
        weighted_sum = 0

        for item in llm_output.get("breakdown", []):
            category = item.get("category", "").strip()
            score = int(item.get("score", 0))
            reason = item.get("reason", "").strip()

            # Normalize category name for weight matching
            normalized_category = category.lower()
            matched_key = next((k for k in CATEGORY_WEIGHTS.keys() if k.lower() == normalized_category), None)

            weight = CATEGORY_WEIGHTS.get(matched_key, 0)
            weighted_sum += score * weight

            breakdown_sorted.append({
                "category": matched_key if matched_key else category,
                "score": score,
                "reason": reason
            })

        # Sort breakdown by score (descending)
        breakdown_sorted.sort(key=lambda x: x["score"], reverse=True)

        # Calculate final score out of 100
        call_score = int(round(weighted_sum * 20))

        # Get the performance rating
        performance_rating = get_performance_rating(call_score)
        what_went_well = llm_output.get("what_went_well", "")
        areas_of_improvement = llm_output.get("areas_of_improvement", "")

        return {
            "breakdown": breakdown_sorted,
            "call_score": call_score,
            "performance_rating": performance_rating,
            "what_went_well": what_went_well,
            "areas_of_improvement": areas_of_improvement
        }

    except Exception as e:
        raise Exception(f"score_call failed: {str(e)}")