"""
Flask web application for transcribing audio files using OpenAI's Whisper API.
Enhanced with speaker identification through smart prompting.

This version includes:
- Duration limit fix (handles files >23.3 minutes)
- Streaming response fix (proper delta extraction)
- Speaker identification via prompt engineering
"""

from __future__ import annotations

import os
import stat
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI
from pydub import AudioSegment  # type: ignore
import math
import io
import tempfile


app = Flask(__name__)

# Directory used to temporarily store uploaded files
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API key file location
API_KEY_FILE = os.getenv("OPENAI_KEY_PATH", os.path.expanduser("~/.openai_api_key"))


def load_api_key() -> str | None:
    """Read the saved API key from disk or return None if not set."""
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()
            return key if key else None
    except FileNotFoundError:
        return None


def save_api_key(key: str) -> None:
    """Persist the given API key to disk with restricted permissions."""
    os.makedirs(os.path.dirname(API_KEY_FILE), exist_ok=True)
    with open(API_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())
    os.chmod(API_KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)


def build_speaker_aware_prompt(user_prompt: str = "", speaker_names: str = "", 
                               enable_speakers: bool = False) -> str:
    """
    Build an enhanced prompt for speaker identification.
    
    Args:
        user_prompt: User's custom prompt/context
        speaker_names: Comma-separated list of speaker names
        enable_speakers: Whether to enable speaker detection
    
    Returns:
        Enhanced prompt string
    """
    if not enable_speakers:
        return user_prompt
    
    # Base prompt for speaker identification
    speaker_prompt = """Please format the transcript with speaker labels.
When you can identify distinct speakers or speaker changes, format as:

[Speaker A]: Their text here...
[Speaker B]: Their response here...

Look for these cues to identify speakers:
- Self-introductions ("Hi, I'm John")
- When people address each other by name
- Clear voice changes or conversation turn-taking
- Questions and answers pattern

If you can determine actual names from the conversation, use them:
[John]: Hello everyone...
[Sarah]: Thanks John...

Otherwise use Speaker A, Speaker B, etc."""
    
    # Add specific speaker names if provided
    if speaker_names:
        names = [n.strip() for n in speaker_names.split(',')]
        speaker_prompt += f"\n\nExpected speakers in this recording: {', '.join(names)}"
        speaker_prompt += "\nUse these names when you can identify who is speaking."
    
    # Combine with user's custom prompt
    if user_prompt:
        speaker_prompt += f"\n\nAdditional context: {user_prompt}"
    
    return speaker_prompt


@app.route("/")
def index() -> str:
    """Serve the main page with speaker identification options."""
    env_key = os.getenv("OPENAI_API_KEY")
    has_key = bool(env_key or load_api_key())
    return render_template("index_with_speakers.html", has_key=has_key)


@app.route("/set_api_key", methods=["POST"])
def set_api_key() -> tuple[dict[str, str], int]:
    """Accept an API key from the client and save it locally."""
    data = request.get_json(silent=True) or {}
    key = data.get("api_key", "").strip()
    if not key:
        return {"error": "Missing API key"}, 400
    save_api_key(key)
    return {"status": "saved"}, 200


@app.route("/transcribe", methods=["POST"])
def transcribe() -> tuple[dict[str, str], int]:
    """
    Receive an audio file and return its transcript using OpenAI's API.
    Enhanced with speaker identification support.
    """
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY") or load_api_key()
    if not api_key:
        return {"error": "API key not set"}, 400
    
    # Check for audio file
    if "audio" not in request.files:
        return {"error": "No audio file provided"}, 400
    
    audio_file = request.files["audio"]
    if not audio_file.filename:
        return {"error": "Empty filename"}, 400
    
    # Save the uploaded file
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)
    
    # Extract form parameters including speaker options
    language = request.form.get("language", "").strip() or None
    user_prompt = request.form.get("prompt", "").strip() or ""
    
    # New speaker-related parameters
    enable_speakers = request.form.get("enable_speakers", "false").lower() == "true"
    speaker_names = request.form.get("speaker_names", "").strip()
    
    # Build the enhanced prompt
    final_prompt = build_speaker_aware_prompt(user_prompt, speaker_names, enable_speakers)
    
    model = "gpt-4o-transcribe"
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # Load audio file to check both size and duration
        audio = AudioSegment.from_file(file_path)
        file_size = os.path.getsize(file_path)
        
        # OpenAI API limits for gpt-4o-transcribe model
        max_chunk_bytes = 24 * 1024 * 1024  # 24MB
        max_duration_ms = 1400 * 1000  # 1400 seconds in milliseconds
        
        audio_duration_ms = len(audio)
        print(f"Audio file: {file_size / (1024*1024):.2f}MB, {audio_duration_ms / 1000:.1f} seconds")
        
        # Add speaker info to console output
        if enable_speakers:
            print(f"Speaker detection enabled. Names: {speaker_names if speaker_names else 'Auto-detect'}")
        
        transcripts: list[str] = []
        previous_transcript = ""
        
        # Check if chunking is needed
        needs_chunking = file_size > max_chunk_bytes or audio_duration_ms > max_duration_ms
        
        if not needs_chunking:
            # Single request scenario
            with open(file_path, "rb") as f:
                params: dict[str, object] = {
                    "model": model,
                    "file": f,
                    "response_format": "text",
                    "stream": True,
                }
                if language:
                    params["language"] = language
                if final_prompt:
                    params["prompt"] = final_prompt
                
                resp_iter = client.audio.transcriptions.create(**params)
                
                # Aggregate streamed events (with fix for delta extraction)
                segment_text = ""
                for event in resp_iter:
                    delta_text = ""
                    if hasattr(event, "delta"):
                        delta_text = event.delta or ""
                    elif hasattr(event, "text"):
                        delta_text = event.text or ""
                    elif isinstance(event, dict):
                        delta_text = event.get("delta", event.get("text", ""))
                    elif isinstance(event, str):
                        delta_text = event
                    segment_text += delta_text
                
                transcripts.append(segment_text.strip())
        else:
            # Chunking required
            safe_duration_ms = 1300 * 1000  # 21.67 minutes with buffer
            num_chunks = math.ceil(audio_duration_ms / safe_duration_ms)
            chunk_length_ms = math.ceil(audio_duration_ms / num_chunks)
            
            print(f"Splitting into {num_chunks} chunks of ~{chunk_length_ms/1000:.1f} seconds each")
            
            chunk_index = 0
            for start_ms in range(0, audio_duration_ms, chunk_length_ms):
                end_ms = min(start_ms + chunk_length_ms, audio_duration_ms)
                chunk = audio[start_ms:end_ms]
                chunk_duration_s = len(chunk) / 1000
                
                print(f"Processing chunk {chunk_index + 1}/{num_chunks}: {chunk_duration_s:.1f} seconds")
                
                buf = io.BytesIO()
                chunk.export(buf, format="mp3", bitrate="128k")
                buf.seek(0)
                buf.name = f"chunk_{chunk_index}.mp3"
                chunk_index += 1
                
                params: dict[str, object] = {
                    "model": model,
                    "file": buf,
                    "response_format": "text",
                    "stream": True,
                }
                if language:
                    params["language"] = language
                
                # Build prompt with context and speaker instructions
                if previous_transcript or final_prompt:
                    # For chunked processing, we need to maintain speaker context
                    if enable_speakers:
                        context_prompt = "Continue the transcript maintaining the same speaker labels from the previous section:\n\n"
                        context_prompt += previous_transcript[-1500:] if len(previous_transcript) > 1500 else previous_transcript
                        context_prompt += "\n\n" + final_prompt
                    else:
                        combined_prompt = (previous_transcript + "\n\n" + final_prompt).strip()
                        if len(combined_prompt) > 2000:
                            prev_limit = 1500 - len(final_prompt)
                            if prev_limit > 0:
                                context_prompt = (previous_transcript[-prev_limit:] + "\n\n" + final_prompt).strip()
                            else:
                                context_prompt = final_prompt
                        else:
                            context_prompt = combined_prompt
                    
                    params["prompt"] = context_prompt
                
                resp_iter = client.audio.transcriptions.create(**params)
                
                # Aggregate streamed events
                segment_text = ""
                for event in resp_iter:
                    delta_text = ""
                    if hasattr(event, "delta"):
                        delta_text = event.delta or ""
                    elif hasattr(event, "text"):
                        delta_text = event.text or ""
                    elif isinstance(event, dict):
                        delta_text = event.get("delta", event.get("text", ""))
                    elif isinstance(event, str):
                        delta_text = event
                    segment_text += delta_text
                
                cleaned = segment_text.strip()
                transcripts.append(cleaned)
                previous_transcript = cleaned
        
        # Concatenate segments
        transcript = "\n\n".join(filter(None, transcripts))
        
        # Post-process for speaker consistency if enabled
        if enable_speakers and transcript:
            transcript = normalize_speaker_labels(transcript)
        
    except Exception as exc:
        print(f"Error during transcription: {exc!r}")
        return {"error": str(exc)}, 500
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
    
    return {"transcript": transcript}, 200


def normalize_speaker_labels(transcript: str) -> str:
    """
    Normalize speaker labels for consistency.
    Converts various formats to a standard format.
    """
    import re
    
    # Common patterns for speaker labels
    patterns = [
        r'\[([^\]]+)\]:',  # [Name]:
        r'([A-Z][a-z]+):',  # Name:
        r'Speaker ([A-Z\d]+):',  # Speaker A:
        r'SPEAKER ([A-Z\d]+):',  # SPEAKER 1:
    ]
    
    # Find all speaker labels
    speakers = set()
    for pattern in patterns:
        matches = re.findall(pattern, transcript)
        speakers.update(matches)
    
    # Normalize to [Speaker]: format
    for speaker in speakers:
        # Replace various formats with standard format
        transcript = re.sub(
            rf'({re.escape(speaker)}|SPEAKER\s+{re.escape(speaker)}|Speaker\s+{re.escape(speaker)})\s*:',
            f'[{speaker}]:',
            transcript
        )
    
    return transcript


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=True)