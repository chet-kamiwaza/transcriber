# Whisper Web Transcription App

This project provides a simple web interface for transcribing audio files using
OpenAI’s hosted speech‐to‐text models. It is designed to run locally on
your machine and store your API key securely so you do not need to enter it
each time.

## Features

* **API key management:** The first time you visit the app, it prompts for
  your OpenAI API key. The key is saved to `~/.openai_api_key` with read/write
  permissions restricted to the current user. Subsequent visits detect the
  stored key automatically.
* **File upload and automatic chunking:** Select an audio file from your
  computer. The app accepts any format supported by OpenAI’s API (MP3,
  WAV, M4A, etc.). If the file exceeds the Audio API’s 25 MB limit【50147549551891†screenshot】,
  the server automatically splits it into smaller segments using
  [`pydub`](https://github.com/jiaaro/pydub) and re‑encodes them to MP3.  Each
  chunk is sent to the API with your prompt and the transcript of the
  previous segment to preserve context.  The results are concatenated
  into a single transcript before being returned to your browser.
* **Prompt and language input:** Alongside the file, you can provide a
  free‑form prompt describing the audio (names, acronyms, subject matter,
  etc.) and an optional [ISO 639‑1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
  The prompt is passed to OpenAI’s model to help correct specific words,
  maintain context across chunks and guide punctuation or stylistic choices【68619459314222†screenshot】.
* **Progress indicator:** A progress bar displays upload progress and then
  animates while the server processes your file.  Status updates show
  when the server is “Processing…” so you know work is ongoing.
* **Transcript viewer & download:** The resulting transcript appears in a
  text area and can be downloaded as a `.txt` file.  The app always uses
  the `gpt‑4o‑transcribe` model, which is more accurate than Whisper v2/v3
  and supports prompting【68619459314222†screenshot】.  For every request we
  set `stream=True` and request log‑probabilities of tokens【803826022809978†screenshot】.

## Installation

1. **Clone or extract** the project folder.
2. **Create a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The dependency list includes [`pydub`](https://github.com/jiaaro/pydub) for
   splitting large audio files. `pydub` uses [FFmpeg](https://ffmpeg.org) to
   encode and decode audio; ensure FFmpeg is available on your system. When
   running in Docker, FFmpeg is installed automatically as part of the
   container build.

4. **Run the app:**

   ```bash
   # By default the development server listens on port 5500.
   # You can change the port by setting the PORT environment variable
   # before running the app (e.g., PORT=6000 python app.py).
   python app.py
   ```

5. **Open in browser:** Navigate to `http://localhost:5500` (or whatever
   port you configured via the `PORT` environment variable).  A high port
   such as 5500 is chosen by default to avoid conflicts with common ports
   like 5000.

## Usage

Upon first loading the page, you’ll be prompted to enter your OpenAI API
key. Once saved, the API key input is hidden and you can begin uploading
audio files. Click **Transcribe** to send the audio to OpenAI and watch
the progress indicator as the file uploads and is processed. The
transcription will appear when finished. You can then download the text for
archival or editing.

### Large files and chunking

The Audio API currently limits file uploads to 25 MB【258415679780279†L32-L35】【863089295758505†screenshot】. If you
upload a file that exceeds this threshold, the app automatically splits it
into multiple segments, each under the limit. Each chunk is transcribed
individually and the transcripts are joined together. This process is
handled transparently by the server, so you can upload long recordings
without worrying about the limit. The progress bar continues to show the
overall progress while the chunks are processed.

## Model selection and context

Earlier versions of this app defaulted to `whisper-1` with an option to
specify a different model.  In March 2025 OpenAI released
`gpt-4o-transcribe` and `gpt-4o-mini-transcribe`, which reduce word error
rates by 10–20 % compared with Whisper【68619459314222†screenshot】.  The current
version always uses `gpt-4o-transcribe` for the highest accuracy and to
support longer prompts.  You no longer need to specify the model in
`script.js`; instead, adjust your prompt and (optional) language code to
guide the model.  When chunking long files, the server automatically
prepends the transcript of the preceding chunk to the prompt to preserve
context【68619459314222†screenshot】.

## Security considerations

Your API key is stored locally in a hidden file with restrictive
permissions. However, anyone with access to your user account could read
this file. Do not run the server on an untrusted machine or expose it to
the internet without additional safeguards (e.g. authentication or firewalls).

## Running in Docker

This repository includes a `Dockerfile` **and** a `docker-compose.yml` to simplify
deployment.  A persistent volume is used to store your API key and other
files so they survive container restarts.  The application listens on a
high port (5500) inside the container, which is mapped to the host port
by default.  You can adjust the host port by editing the `ports` mapping
in the compose file.

### Using docker compose

1. Build and start the service with compose:

   ```bash
   docker compose up -d --build
   ```

   This command builds the image using the included `Dockerfile` and
   creates a named volume called `whisper_data`.  The volume is mounted
   at `/data` inside the container, and your API key will be saved at
   `/data/.openai_api_key`.  Because `/data` is declared as a volume
   in the compose file, the key and any other files stored there persist
   across container restarts.

2. Navigate to `http://localhost:5500` in your browser.  On the first
   visit you’ll be prompted to enter your API key.  After saving, the key
   is reused automatically for subsequent requests.

3. To stop and remove the container, run:

   ```bash
   docker compose down
   ```

### Using plain Docker

If you prefer not to use compose, you can build and run the container
manually.  The following examples map the container’s port 5500 to a
port on the host.  You can change the host side of the mapping
(`HOST_PORT:5500`) to any high port if 5500 is already in use on your
machine.

```bash
# Build the Docker image
docker build -t whisper-web-app .

# Create a named volume to persist the API key
docker volume create whisper_data

# Run the container, mapping host port 5500 to the container’s port 5500.
# Persist data (including your API key) using the named volume.
docker run -p 5500:5500 \
  -v whisper_data:/data \
  whisper-web-app

# Alternatively, mount a single file from your host as the key file.
# This saves and reuses your API key without creating a named volume.
docker run -p 5500:5500 \
  -v ~/.openai_api_key:/data/.openai_api_key \
  whisper-web-app

# Or provide the key via an environment variable (no persistence).
docker run -p 5500:5500 \
  -e OPENAI_API_KEY=sk-yourkeyhere \
  whisper-web-app
```

After starting the container, open your browser to `http://localhost:5500`.
If a key file exists in the mounted volume or `OPENAI_API_KEY` is set,
the API key prompt will be skipped; otherwise the app will prompt you to
save the key in the persistent storage path.# transcriber
# transcriber
# transcriber
