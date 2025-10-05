"""
Flask web application for transcribing audio files using OpenAI’s Whisper API.

This app exposes a simple web interface where users can upload an audio file
and receive a transcript powered by OpenAI’s server‑hosted models. On first
use the app asks for an API key, which is stored securely in a hidden file
in the user’s home directory with strict permissions (read/write for the
owner only). Subsequent sessions reuse the stored key and skip the prompt.

The transcription endpoint accepts an uploaded file and forwards it to
OpenAI’s audio transcription API. By default the ``whisper-1`` model is
used – OpenAI’s hosted Whisper model – but clients can provide a different
model name in the form data. According to OpenAI’s documentation, the
gpt‑4o‑transcribe models launched in March 2025 offer superior accuracy
compared to Whisper v2 and v3【363171705166428†L186-L198】, so users should
consider specifying ``gpt-4o-transcribe`` once their account has access.

To run the app locally:

1. Create a virtual environment and install dependencies:

   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start the development server:

   ```
   python app.py
   ```

3. Open ``http://localhost:5500`` in your browser.  The default port
   (5500) is chosen to reduce conflicts with other development services.
   You can change it by setting the ``PORT`` environment variable before
   running the server.  The site will prompt you for your OpenAI API key
   on first visit. Afterwards you can upload audio files and download
   transcripts.

Note that this app does not implement user authentication beyond storing
the API key locally. Protect access to your machine appropriately.
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

# Directory used to temporarily store uploaded files. It is created
# relative to the application root and cleaned up after each request.
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Location to store the API key.  By default this uses the current user’s
# home directory on the host (`~/.openai_api_key`). When running in a
# container, you can override this path by setting the ``OPENAI_KEY_PATH``
# environment variable. If you mount a volume at that location, the key
# will persist across container runs.
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
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(API_KEY_FILE), exist_ok=True)
    with open(API_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())
    # Restrict permissions: read/write for user only
    os.chmod(API_KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)


@app.route("/")
def index() -> str:
    """Serve the main page.

    Determine whether an API key is already available.  A key can be
    provided via the ``OPENAI_API_KEY`` environment variable or stored
    locally in the hidden file.  The template uses ``has_key`` to decide
    whether to show the API key prompt.
    """
    env_key = os.getenv("OPENAI_API_KEY")
    has_key = bool(env_key or load_api_key())
    return render_template("index.html", has_key=has_key)


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
    Receive an audio file and return its transcript using OpenAI’s API.

    This version enforces a few design decisions based on user requests:

    * It always uses the ``gpt-4o-transcribe`` model, which offers higher
      accuracy than Whisper v2/v3 models【68619459314222†screenshot】.
    * It accepts an optional ``prompt`` field from the client and
      automatically prepends the transcript of the previous chunk when
      processing large audio files.  This improves consistency across
      segments as suggested in the OpenAI prompting guidelines【68619459314222†screenshot】.
    * It enables streaming mode (``stream=True``).  While the response is
      streamed from the API, the server aggregates the partial events into
      a single string before returning to the client.
    * The API enforces a 25 MB file size limit【50147549551891†screenshot】, so
      this handler divides larger files into smaller chunks using pydub.
    """
    # Prefer an API key provided via environment variable; fallback to saved file
    api_key = os.getenv("OPENAI_API_KEY") or load_api_key()
    if not api_key:
        return {"error": "API key not set"}, 400
    if "audio" not in request.files:
        return {"error": "No audio file provided"}, 400
    audio_file = request.files["audio"]
    if not audio_file.filename:
        return {"error": "Empty filename"}, 400
    # Save the uploaded file securely
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)
    # Extract optional form parameters.  The model is fixed to gpt-4o-transcribe.
    # Language is optional and may be blank.
    language = request.form.get("language", "").strip() or None
    user_prompt = request.form.get("prompt", "").strip() or ""
    model = "gpt-4o-transcribe"
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    try:
        # Determine file size to decide whether chunking is required.
        file_size = os.path.getsize(file_path)
        max_chunk_bytes = 24 * 1024 * 1024
        transcripts: list[str] = []
        previous_transcript = ""
        if file_size <= max_chunk_bytes:
            # Single request scenario
            with open(file_path, "rb") as f:
                params: dict[str, object] = {
                    "model": model,
                    "file": f,
                    "response_format": "text",
                    "stream": True,
                }
                # Include language if provided
                if language:
                    params["language"] = language
                # Build the prompt.  For a single file this is just the user prompt.
                if user_prompt:
                    params["prompt"] = user_prompt
                # Call the API.  When stream=True the client returns an iterator.
                resp_iter = client.audio.transcriptions.create(**params)
                # Aggregate streamed events into a single string
                segment_text = ""
                for event in resp_iter:
                    # Each event may expose a ``text`` attribute or key.  We
                    # append the delta text as we receive it.
                    delta = None
                    if hasattr(event, "text"):
                        try:
                            delta = event.text  # type: ignore[attr-defined]
                        except Exception:
                            delta = None
                    if delta is None:
                        try:
                            delta = event["text"]  # type: ignore[index]
                        except Exception:
                            delta = None
                    if delta is None:
                        delta = str(event)
                    segment_text += str(delta)
                transcripts.append(segment_text.strip())
        else:
            # Read the file using pydub and split into chunks based on duration.
            audio = AudioSegment.from_file(file_path)
            num_chunks = math.ceil(file_size / max_chunk_bytes)
            chunk_length_ms = math.ceil(len(audio) / num_chunks)
            chunk_index = 0
            for start_ms in range(0, len(audio), chunk_length_ms):
                end_ms = min(start_ms + chunk_length_ms, len(audio))
                chunk = audio[start_ms:end_ms]
                buf = io.BytesIO()
                # Export as MP3 to reduce size and assign a name so the
                # OpenAI client can infer the format
                chunk.export(buf, format="mp3")
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
                # Inject the previous transcript into the prompt for context
                if previous_transcript or user_prompt:
                    # Combine the user prompt and previous transcript.  The
                    # previous transcript is placed first so that user
                    # instructions can override it.  Leaving extra
                    # whitespace has no effect on the model.
                    combined_prompt = (previous_transcript + "\n\n" + user_prompt).strip()
                    params["prompt"] = combined_prompt
                # Call the API for this chunk.  Stream the partial results
                # and assemble them into a single string for this segment.
                resp_iter = client.audio.transcriptions.create(**params)
                segment_text = ""
                for event in resp_iter:
                    delta = None
                    if hasattr(event, "text"):
                        try:
                            delta = event.text  # type: ignore[attr-defined]
                        except Exception:
                            delta = None
                    if delta is None:
                        try:
                            delta = event["text"]  # type: ignore[index]
                        except Exception:
                            delta = None
                    if delta is None:
                        delta = str(event)
                    segment_text += str(delta)
                # Store the segment and update previous_transcript
                cleaned = segment_text.strip()
                transcripts.append(cleaned)
                previous_transcript = cleaned
        # Concatenate segments with blank lines
        transcript = "\n\n".join(filter(None, transcripts))
    except Exception as exc:  # noqa: BLE001
        print(f"Error during transcription: {exc!r}")
        return {"error": str(exc)}, 500
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
    return {"transcript": transcript}, 200


if __name__ == "__main__":
    """
    Run the Flask development server.

    The port can be configured via the ``PORT`` environment variable. A
    default of 5500 is used here to avoid conflicts with common
    development ports. When deployed via docker-compose the service maps
    this port to the host.
    """
    port = int(os.getenv("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=True)