# Whisper Web Transcription App

A secure, self-hosted web application for transcribing audio files using OpenAI's state-of-the-art speech-to-text models. Features automatic handling of large files through intelligent chunking, context preservation, and a user-friendly interface.

## 🚀 Features

* **Latest AI Model:** Uses OpenAI's `gpt-4o-transcribe` for 10-20% better accuracy than Whisper v2/v3
* **Smart Chunking:** Automatically handles files of ANY size or duration
  - Files >25MB are split by size
  - Files >23.3 minutes are split by duration
  - Context preserved between chunks for coherent transcripts
* **Secure API Key Management:** Keys stored locally with restricted permissions (never transmitted)
* **Progress Tracking:** Real-time upload progress and processing status
* **Multiple Audio Formats:** Supports MP3, WAV, M4A, and all OpenAI-compatible formats
* **Docker Support:** Production-ready containerization with persistent storage
* **Prompt Support:** Add context, vocabulary, or style instructions for better accuracy

## 📋 Requirements

- Python 3.11+
- FFmpeg (for audio processing)
- OpenAI API key
- Docker (optional, for containerized deployment)

## 🛠️ Installation


### Docker Deployment (Recommended)

1. **Clone the repository:**
```bash
git clone <repository-url>
cd whisper-web-app
```

2. **Build and start with Docker Compose:**
```bash
docker compose up -d --build
```

3. **Access the app:**
   Open http://localhost:5500 in your browser

4. **View logs:**
```bash
docker compose logs -f
```

5. **Stop the container:**
```bash
docker compose down
```

## 💻 Usage

### First Time Setup
1. Visit http://localhost:5500
2. Enter your OpenAI API key (stored securely)
3. The key is saved locally at `~/.openai_api_key` (or `/data/.openai_api_key` in Docker)

### Transcribing Audio
1. **Select your audio file** using the file picker
2. **(Optional) Add language code** (e.g., "en" for English, "es" for Spanish)
3. **(Optional) Add context prompt** to improve accuracy:
   - Include speaker names, technical terms, acronyms
   - Describe the content type (interview, lecture, podcast)
   - Specify desired formatting style
4. **Click "Transcribe"** and watch the progress bar
5. **Download the transcript** as a text file

### Large Files and Duration Limits

The Audio API has two important limits:
- **File size**: Maximum 25 MB per request
- **Duration**: Maximum 1400 seconds (23.33 minutes) per request

If you upload a file that exceeds either threshold, the app automatically 
splits it into multiple segments, each under both limits. For example:
- A 30-minute file will be split into 2 chunks of ~15 minutes each
- A large but short file (>25MB, <23 minutes) will be split by size
- A long but small file (<25MB, >23 minutes) will be split by duration

Each chunk is transcribed individually with context preservation - the 
transcript from the previous chunk is included in the prompt to maintain
continuity. The transcripts are then joined together seamlessly. This 
process is handled transparently by the server, so you can upload recordings
of any length without worrying about the limits.

## 🔒 Security Features

- **Local Key Storage:** API keys never leave your machine
- **Restricted Permissions:** Key file has 0600 permissions (owner read/write only)
- **Secure Filenames:** Protection against path traversal attacks
- **Automatic Cleanup:** Temporary files deleted after processing
- **No External Services:** Completely self-hosted solution

## 🐳 Docker Configuration

The Docker setup includes:
- **Base Image:** Python 3.11-slim for minimal footprint
- **Persistent Storage:** Named volume `whisper_data` for API keys
- **Auto-restart:** Container restarts unless explicitly stopped
- **FFmpeg Included:** Audio processing tools pre-installed
- **Port Mapping:** Container port 5500 mapped to host

### Environment Variables
- `OPENAI_KEY_PATH`: Path to API key file (default: `/data/.openai_api_key`)
- `PORT`: Server port (default: `5500`)
- `OPENAI_API_KEY`: Direct API key (alternative to file storage)

## 📊 Performance Characteristics

| File Size | Duration | Processing |
|-----------|----------|------------|
| <25MB & <23min | Any | Single API call |
| 30 min | ~50MB | 2 chunks × 15 min |
| 1 hour | ~100MB | 3 chunks × 20 min |
| 2 hours | ~200MB | 6 chunks × 20 min |
| Any size | Any duration | Automatic chunking |

## 🔧 Troubleshooting

### Common Issues


**Error: "API key not set"**
- Ensure you've entered your OpenAI API key on first visit
- Check the key file exists at `~/.openai_api_key`

**Error: "No audio file provided"**
- Ensure you've selected a file before clicking Transcribe


### Debug Mode

To see detailed processing information, the Flask app runs in debug mode by default. Check the console output for:
- File size and duration information
- Chunk processing progress
- API response details

## 🛠️ Technical Details

### Architecture
- **Backend:** Flask 2.3+ with OpenAI Python SDK
- **Audio Processing:** PyDub with FFmpeg
- **Frontend:** Vanilla JavaScript with XMLHttpRequest for progress tracking
- **Streaming:** Real-time response streaming from OpenAI API
- **Context Management:** Previous chunk's transcript passed as prompt

### File Structure
```
whisper-web-app/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Docker orchestration
├── templates/
│   └── index.html          # Web interface
├── static/
│   └── script.js           # Client-side logic
└── uploads/                # Temporary file storage (auto-created)
```

## 📝 Recent Updates

### v1.1.0 - Duration Limit Fix
- ✅ Added automatic chunking based on audio duration (23.3-minute limit)
- ✅ Improved context preservation between chunks
- ✅ Added progress logging for chunk processing
- ✅ Optimized MP3 encoding with bitrate control
- ✅ Added prompt length management to prevent overflow

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is available for use under standard open-source terms.

## ⚠️ Important Notes

1. **API Costs:** OpenAI charges for API usage. Monitor your usage at https://platform.openai.com
2. **Privacy:** Audio files are sent to OpenAI for processing. Don't upload sensitive content
3. **Rate Limits:** Subject to OpenAI's rate limits on your API key
4. **Production Use:** For production, consider using Gunicorn instead of Flask's development server

## 🙏 Acknowledgments

- OpenAI for the powerful speech-to-text API
- Flask community for the excellent web framework
- PyDub developers for audio processing capabilities

---

**Ready to transcribe?** Start the app and transform your audio into text with state-of-the-art AI!