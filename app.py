"""
Flask web application for transcribing audio files using OpenAI's Whisper API.
Enhanced with speaker identification through smart prompting.

Fixed version addressing:
- Duplicate content in chunked transcriptions
- Prompt leakage into transcript
- Better speaker identification consistency
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
import re


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


def extract_last_speaker_context(transcript: str, max_chars: int = 500) -> str:
    """
    Extract the last portion of transcript with speaker context preserved.
    This helps maintain speaker continuity across chunks.
    """
    if not transcript:
        return ""
    
    # Take last max_chars characters
    if len(transcript) <= max_chars:
        return transcript
    
    context = transcript[-max_chars:]
    
    # Try to find a clean break point (sentence or speaker change)
    # Look for the last complete speaker label
    speaker_pattern = r'\[([^\]]+)\]:'
    matches = list(re.finditer(speaker_pattern, context))
    
    if matches:
        # Start from the last complete speaker label found
        last_match = matches[-1]
        context = context[last_match.start():]
    else:
        # Try to at least start at a sentence boundary
        sentence_breaks = ['. ', '? ', '! ', '\n']
        for break_char in sentence_breaks:
            pos = context.find(break_char)
            if pos > 0 and pos < len(context) // 2:
                context = context[pos + len(break_char):]
                break
    
    return context.strip()


def build_speaker_aware_prompt(user_prompt: str = "", speaker_names: str = "", 
                               enable_speakers: bool = False, is_continuation: bool = False,
                               context: str = "") -> str:
    """
    Build an enhanced prompt for speaker identification.
    
    Args:
        user_prompt: User's custom prompt/context
        speaker_names: Comma-separated list of speaker names
        enable_speakers: Whether to enable speaker detection
        is_continuation: Whether this is a continuation of a previous chunk
        context: Previous transcript context for continuations
    
    Returns:
        Enhanced prompt string
    """
    if not enable_speakers and not user_prompt:
        return ""
    
    prompt_parts = []
    
    if is_continuation and context:
        # For continuation chunks, provide minimal context
        prompt_parts.append("Continue transcribing. Previous context for reference:")
        prompt_parts.append(f"...{context}")
        prompt_parts.append("Continue from here with the same speaker format.")
    elif enable_speakers:
        # Initial prompt for speaker identification
        speaker_prompt = """Format with speaker labels like [Speaker Name]: text
Identify speakers by voice changes, introductions, and when people address each other.
Use actual names when identified, otherwise use Speaker A, B, etc."""
        
        # Add specific speaker names if provided
        if speaker_names:
            names = [n.strip() for n in speaker_names.split(',')]
            speaker_prompt += f"\nExpected speakers: {', '.join(names)}"
        
        prompt_parts.append(speaker_prompt)
    
    # Add user's custom context (only for first chunk to avoid repetition)
    if user_prompt and not is_continuation:
        prompt_parts.append(f"Context: {user_prompt}")
    
    return "\n".join(prompt_parts) if prompt_parts else ""


def clean_transcript_text(text: str) -> str:
    """
    Clean up transcript text by removing any prompt instructions that leaked through.
    """
    if not text:
        return text
    
    # Remove common prompt instruction patterns that might leak
    instruction_patterns = [
        r'Please format the transcript.*?\n\n',
        r'Continue transcribing.*?\n\n',
        r'Format with speaker labels.*?\n',
        r'Continue from here.*?\n',
        r'Previous context for reference:.*?\n',
        r'Expected speakers:.*?\n',
        r'Context:.*?\n\n',
        r'Additional context:.*?\n\n',
    ]
    
    cleaned = text
    for pattern in instruction_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove multiple consecutive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


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
    Enhanced with speaker identification support and fixed chunking.
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
        
        # Check if chunking is needed
        needs_chunking = file_size > max_chunk_bytes or audio_duration_ms > max_duration_ms
        
        if not needs_chunking:
            # Single request scenario
            with open(file_path, "rb") as f:
                # Build initial prompt
                initial_prompt = build_speaker_aware_prompt(
                    user_prompt, speaker_names, enable_speakers, 
                    is_continuation=False
                )
                
                params: dict[str, object] = {
                    "model": model,
                    "file": f,
                    "response_format": "text",
                    "stream": True,
                }
                if language:
                    params["language"] = language
                if initial_prompt:
                    params["prompt"] = initial_prompt
                
                resp_iter = client.audio.transcriptions.create(**params)
                
                # Aggregate streamed events
                segment_text = ""
                for event in resp_iter:
                    # Extract text from various event formats
                    if hasattr(event, "text"):
                        segment_text += event.text or ""
                    elif isinstance(event, str):
                        segment_text += event
                
                # Clean and store
                cleaned_text = clean_transcript_text(segment_text.strip())
                transcripts.append(cleaned_text)
        else:
            # Chunking required - with overlap for better continuity
            safe_duration_ms = 1300 * 1000  # 21.67 minutes with buffer
            overlap_ms = 5000  # 5 seconds overlap between chunks
            
            # Calculate chunks with overlap
            chunk_starts = []
            current_start = 0
            while current_start < audio_duration_ms:
                chunk_starts.append(current_start)
                current_start += safe_duration_ms - overlap_ms
            
            num_chunks = len(chunk_starts)
            print(f"Splitting into {num_chunks} chunks with {overlap_ms/1000}s overlap")
            
            previous_context = ""
            
            for chunk_idx, start_ms in enumerate(chunk_starts):
                end_ms = min(start_ms + safe_duration_ms, audio_duration_ms)
                chunk = audio[start_ms:end_ms]
                chunk_duration_s = len(chunk) / 1000
                
                print(f"Processing chunk {chunk_idx + 1}/{num_chunks}: {chunk_duration_s:.1f} seconds")
                
                # Export chunk to buffer
                buf = io.BytesIO()
                chunk.export(buf, format="mp3", bitrate="128k")
                buf.seek(0)
                buf.name = f"chunk_{chunk_idx}.mp3"
                
                # Build appropriate prompt
                if chunk_idx == 0:
                    # First chunk - full prompt
                    chunk_prompt = build_speaker_aware_prompt(
                        user_prompt, speaker_names, enable_speakers,
                        is_continuation=False
                    )
                else:
                    # Continuation chunks - minimal context
                    chunk_prompt = build_speaker_aware_prompt(
                        user_prompt="",  # Don't repeat user prompt
                        speaker_names="",  # Already established
                        enable_speakers=enable_speakers,
                        is_continuation=True,
                        context=previous_context
                    )
                
                params: dict[str, object] = {
                    "model": model,
                    "file": buf,
                    "response_format": "text",
                    "stream": True,
                }
                if language:
                    params["language"] = language
                if chunk_prompt:
                    params["prompt"] = chunk_prompt
                
                resp_iter = client.audio.transcriptions.create(**params)
                
                # Aggregate streamed events
                segment_text = ""
                for event in resp_iter:
                    if hasattr(event, "text"):
                        segment_text += event.text or ""
                    elif isinstance(event, str):
                        segment_text += event
                
                # Clean the text
                cleaned_text = clean_transcript_text(segment_text.strip())
                
                # Handle overlap - for chunks after the first, try to remove duplicate content
                if chunk_idx > 0 and overlap_ms > 0 and previous_context:
                    # Look for where the new chunk might overlap with previous context
                    # This is a simple approach - could be enhanced with fuzzy matching
                    overlap_text = previous_context[-200:]  # Last ~200 chars of previous
                    if overlap_text in cleaned_text[:400]:  # Look in first ~400 chars
                        # Found potential overlap, start after it
                        overlap_pos = cleaned_text.find(overlap_text) + len(overlap_text)
                        cleaned_text = cleaned_text[overlap_pos:].strip()
                
                transcripts.append(cleaned_text)
                
                # Update context for next chunk (last portion with speaker info)
                previous_context = extract_last_speaker_context(cleaned_text, max_chars=300)
        
        # Concatenate segments with proper spacing
        transcript = "\n\n".join(filter(None, transcripts))
        
        # Post-process for speaker consistency if enabled
        if enable_speakers and transcript:
            transcript = normalize_speaker_labels(transcript)
        
        # Final cleanup to remove any remaining artifacts
        transcript = clean_transcript_text(transcript)
        
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
    Normalize speaker labels for consistency and clean up formatting.
    """
    if not transcript:
        return transcript
    
    # First pass: standardize various speaker label formats
    # Convert "Speaker A:", "SPEAKER 1:", "Name:" to "[Name]:" format
    patterns = [
        (r'Speaker\s+([A-Z]):', r'[Speaker \1]:'),
        (r'SPEAKER\s+(\d+):', r'[Speaker \1]:'),
        (r'^([A-Z][a-z]+):', r'[\1]:'),  # Name at start of line
        (r'\n([A-Z][a-z]+):', r'\n[\1]:'),  # Name after newline
    ]
    
    normalized = transcript
    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.MULTILINE)
    
    # Ensure consistent spacing after speaker labels
    normalized = re.sub(r'\]\s*:\s*', ']: ', normalized)
    
    # Remove duplicate speaker labels on the same line
    normalized = re.sub(r'(\[[^\]]+\]:\s*)\1+', r'\1', normalized)
    
    return normalized


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=True)