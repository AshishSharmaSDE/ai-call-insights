# 🧠 AI Call Insights — Live Call Transcription & Sentiment Analysis (POC)

## 📋 Overview
**AI Call Insights** is an end-to-end Proof-of-Concept for **real-time call transcription and sentiment analysis**.  
It demonstrates how a healthcare or insurance call-center system can transcribe live audio, analyze emotion, and expose an API for further integration.

The architecture uses:
- 🎙️ **Whisper (OpenAI)** for speech-to-text (local or API)
- 💬 **LLaMA-2 / LLaMA-3** for sentiment classification
- ⚡ **FastAPI** backend with WebSocket streaming
- 🖥️ Simple frontend for testing live transcription

---

## 🧩 Features
- 🔄 Real-time audio streaming via **WebSocket**
- 🗣️ On-the-fly **speech-to-text** transcription
- ❤️ **Sentiment analysis** (Positive / Negative / Neutral)
- 🌐 API endpoints for integration with enterprise systems
- 🧱 Modular services — easy to switch between **local LLMs** and **cloud APIs**
- 📦 Ready for Docker / Azure deployment

---

## 🏗️ Project Structure
```
ai-call-insights/
│
├── app/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── core/
│   │   └── config.py           # Environment variables and settings
│   ├── routers/
│   │   ├── transcribe.py       # REST endpoint for file transcription
│   │   ├── sentiment.py        # REST endpoint for sentiment analysis
│   │   ├── process_call.py     # Combined call-processing route
│   │   └── realtime.py         # WebSocket route for live audio
│   ├── services/
│   │   ├── whisper_service.py  # Local/Remote Whisper logic
│   │   ├── llama_service.py    # LLaMA or external API call
│   │   └── realtime_service.py # Streaming queue and orchestration
│   ├── utils/
│   │   ├── audio_utils.py      # Audio helpers
│   │   └── schemas.py          # Data models
│
├── frontend/
│   └── realtime_test.html      # Simple WebSocket test page
│
├── .env.example                # Sample environment file
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### 1️⃣ Clone and enter project
```bash
git clone https://github.com/<your-org>/ai-call-insights.git
cd ai-call-insights
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate     # Windows
source .venv/bin/activate      # Linux/Mac
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
For local LLMs:
```bash
pip install openai-whisper pydub torch
```

If using **API models**, you can skip heavy dependencies.

---

## 🧾 .env Configuration
Copy `.env.example` to `.env` and set values:

```bash
HOST=127.0.0.1
PORT=8000

# Whisper (Speech-to-Text)
USE_LOCAL_WHISPER=True
WHISPER_MODEL=base
WHISPER_API_KEY=<if using external API>

# LLaMA / Sentiment Model
USE_LOCAL_LLAMA=True
LLAMA_MODEL=llama2
LLAMA_API_KEY=<if using external API>
```

---

## 🧠 Installing Local Models

### 🗣️ Option 1 — Local Whisper (OpenAI)
```bash
pip install openai-whisper
```
Whisper requires **ffmpeg**.  
- Download from: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)  
- Extract to `C:\ffmpeg\bin`
- Add to `PATH` or set in `.env`:
  ```bash
  FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
  FFPROBE_PATH=C:\ffmpeg\bin\ffprobe.exe
  ```

---

### 💬 Option 2 — Local LLaMA (via Ollama)
Install [Ollama](https://ollama.ai/download).

Then pull the models:
```bash
ollama pull llama2
ollama pull sentiment
```

Test locally:
```bash
ollama run llama2 "Classify this as positive, negative, or neutral: I’m happy today."
```

Set in `.env`:
```bash
USE_LOCAL_LLAMA=True
LLAMA_MODEL=llama2
```

---

### ☁️ Option 3 — Use APIs (Azure/OpenAI)
If you prefer external APIs (for production or scaling):

```bash
USE_LOCAL_WHISPER=False
USE_LOCAL_LLAMA=False

WHISPER_API_URL=https://api.openai.com/v1/audio/transcriptions
WHISPER_API_KEY=<your_key>

LLAMA_API_URL=https://api.groq.com/v1/chat/completions
LLAMA_API_KEY=<your_key>
```

These will be used automatically by the services.

---

## 🚀 Running Locally

### Backend
```bash
uvicorn app.main:app --reload
```

You should see:
```
✅ Using ffmpeg from: C:\ffmpeg\bin\ffmpeg.exe
✅ Using ffprobe from: C:\ffmpeg\bin\ffprobe.exe
INFO: Application startup complete.
```

### Frontend
Open in browser:
```
frontend/realtime_test.html
```
Then click **Start Streaming** to begin transcription.

---
