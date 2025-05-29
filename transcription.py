import whisperx
import os
import uuid
import requests
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to get Hugging Face auth token from environment variable
hf_token = os.getenv("HF_AUTH_TOKEN")

# Load WhisperX ASR model once at import time for efficiency
model = whisperx.load_model("small", device="cpu", compute_type="default")

if hf_token:
    hf_token = hf_token.strip()
else:
    # TODO: Remove this static token after testing is complete
    hf_token = "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"

# Initialize diarization pipeline with authentication token and CPU device
diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=hf_token,
    device="cpu"
)

def handle_transcription_request(data):
    """
    Parses input JSON and processes the transcription request.

    Args:
        data (dict): Request payload

    Returns:
        dict: Response with job_id, lead_id, transcription results, etc.
    """
    try:
        # Extract required and optional fields
        job_id = data["job_id"]
        lead_id = data["lead_id"]
        audio_url = data["pre_url"]
        s3_link = data.get("s3_link", "")

        # Generate a unique filename
        local_filename = f"audio_{uuid.uuid4().hex}.mp3"

        # Transcribe and diarize the audio
        result = process_transcription(audio_url, local_filename)

        return {
            "job_id": job_id,
            "lead_id": lead_id,
            "s3_link": s3_link,
            "content": result["speaker_blocks"],
            "conversation": result["conversation"],
            "status": "success",
            "message": "Transcription completed successfully"
        }

    except Exception as e:
        logger.exception("Failed to handle transcription request.")
        return {
            "job_id": data.get("job_id", ""),
            "lead_id": data.get("lead_id", ""),
            "s3_link": data.get("s3_link", ""),
            "status": "error",
            "message": str(e),
            "content": {},
            "conversation": []
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

def process_transcription(audio_url, local_filename):
    """
    Main function to handle audio transcription and speaker diarization.

    Args:
        audio_url (str): URL to download the audio
        local_filename (str): Temporary filename to save audio

    Returns:
        dict: Transcription result containing speaker blocks and conversation
    """
    try:
        download_audio(audio_url, local_filename)
        audio = whisperx.load_audio(local_filename)

        # Transcribe the audio
        result = model.transcribe(audio, language="en")

        # Align segments
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device="cpu",
                                return_char_alignments=False)

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
    finally:
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
