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
        dict: Response with job_id, lead_id, transcription results, etc.
    """
    try:
        # Extract required and optional fields from the 'data' dictionary
        jobId = data.get("jobId")
        fileUrl = data.get("fileUrl") 

        # Generate a unique filename
        local_filename = f"audio_{uuid.uuid4().hex}.mp3" #Uncomment this when we get link

        # Transcribe and diarize the audio
        result = process_transcription(fileUrl, local_filename)

        # Assign speaker roles using LLM
        roles = assign_speaker_roles(result["conversation"])
        
        # Create reverse mapping for role replacement
        role_mapping = {spk_id: role for spk_id, role in roles.items()}
        
        # Update conversation with assigned roles
        updated_conversation = []
        for turn in result["conversation"]:
            speaker = turn["speaker"]
            # Replace speaker ID with assigned role if available
            turn["speaker"] = role_mapping.get(speaker, speaker)
            updated_conversation.append(turn)

        return {
            "jobId": jobId,
            "status": "COMPLETED",
            "conversation": updated_conversation  # Updated with role names
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
    Automatically determine Lead/Agent roles using LLM
    
    Args:
        conversation (list): List of conversation turns with speaker and text
        
    Returns:
        dict: Mapping of original speaker IDs to assigned roles (Lead/Agent)
    """
    # Format conversation for analysis
    convo_lines = []
    for turn in conversation:
        convo_lines.append(f"{turn['speaker']}: {turn['text']}")
    convo_string = "\n".join(convo_lines)

    system_prompt = (
        "You are an expert AI assistant analyzing a sales call transcript between an EdTech agent and a student lead (or their guardian). "
        "Your primary goal is to accurately identify and assign one of two roles — *'Lead'* or *'Agent'* — to all speakers present in the conversation, *even if the diarization tool has incorrectly assigned multiple speaker IDs to a single individual.*\n"
        "Regardless of how many speaker IDs are present (e.g., SPEAKER_00, SPEAKER_01, SPEAKER_02, etc.), you must consolidate them so that the final output assigns roles to *exactly two unique individuals: one 'Lead' and one 'Agent'.*\n\n"
        "Here are the rules for role assignment and output format:\n"
        "1. *'Lead'*: This role belongs to the prospective student or their guardian. They will typically be asking questions about courses, fees, admissions processes, or expressing interest in enrolling.\n"
        "2. *'Agent'*: This role belongs to the EdTech representative. They will be answering questions, explaining course offerings, providing program details, discussing payment plans, and generally guiding or persuading the lead.\n"
        "3. *Crucial Consolidation*: The diarization tool (WhisperX) may sometimes incorrectly split a single person's speech into multiple speaker IDs (e.g., SPEAKER_00 and SPEAKER_02 might both be the 'Lead'). You *must* identify such instances by analyzing the content and continuity of speech. All speaker IDs that belong to the same person must be assigned the *same consolidated role* ('Lead' or 'Agent').\n"
        "4. *Final Role Count*: Your output must reflect *exactly one 'Lead' and one 'Agent'*. All distinct speaker IDs from the input conversation must be mapped to one of these two consolidated roles.\n"
        "5. *Output Format*: Your output MUST be a JSON object where each detected speaker ID from the input is mapped to its assigned role ('Lead' or 'Agent').\n"
        "6. *No Extra Content*: Output ONLY the JSON. Do not include any explanations, comments, or additional formatting.\n"
    )
    user_prompt = f"""
        CONVERSATION:
        {convo_string}
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
        
        # Parse JSON
        roles = json.loads(json_str)
        
        # Validate roles
        valid_roles = {'Lead', 'Agent'}
        if not all(role in valid_roles for role in roles.values()):
            raise ValueError(f"Invalid roles detected: {roles}. Must be 'Lead' or 'Agent'")
            
        logger.info(f"Role assignment successful: {roles}")
        return roles
        
    except Exception as e:
        logger.error(f"Role assignment failed: {str(e)}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
        return {}
