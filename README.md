---
title: AI Video Assistant
emoji: 🎬
colorFrom: purple
colorTo: cyan
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
python_version: "3.11"
---

# 🎬 AI Video Assistant

> Transform any YouTube video or audio file into actionable meeting intelligence — automatic transcription, AI summarisation, key decision extraction, and an interactive RAG chatbot powered by Mistral AI.

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1c3c3c?logo=langchain&logoColor=white)](https://langchain.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-mistral--small-ff7000)](https://mistral.ai/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Live Demo

**[→ Try it on Streamlit Cloud](https://ai-video-assistant.streamlit.app)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔗 **YouTube URL** | Paste any YouTube link — audio is downloaded automatically |
| 📁 **File Upload** | Upload `.mp4`, `.mp3`, `.wav`, `.webm`, `.m4a`, `.ogg` directly |
| 📝 **Transcription** | Local OpenAI Whisper (English) or Sarvam AI (Hinglish → English) |
| 📋 **AI Summary** | Map-reduce summarisation via Mistral AI — bullet-point meeting notes |
| ✅ **Action Items** | Extracts tasks with owners and deadlines |
| 🔑 **Key Decisions** | Highlights decisions made during the meeting |
| ❓ **Open Questions** | Surfaces unresolved topics for follow-up |
| 💬 **RAG Chatbot** | Ask anything about the meeting — answers grounded in the transcript |
| 📥 **Download** | Export transcript, summary, or full report as `.txt` |

---

## 🏗️ Architecture

```
User Input (YouTube URL / Uploaded File)
        │
        ▼
┌──────────────────┐
│  Audio Processor │  yt-dlp + pydub → 16kHz mono WAV → 10-min chunks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Transcriber    │  Whisper (english) │ Sarvam AI STT-Translate (hinglish)
└────────┬─────────┘
         │
         ├──────────────────────────────────────────────────────────┐
         ▼                                                          ▼
┌──────────────────┐                                   ┌───────────────────┐
│  LLM Pipeline    │  Mistral AI via LangChain LCEL    │   RAG Pipeline    │
│  • Title         │  map-reduce summarisation         │  ChromaDB +       │
│  • Summary       │  + extraction chains              │  HuggingFace      │
│  • Action Items  │                                   │  all-MiniLM-L6-v2 │
│  • Decisions     │                                   │  + Mistral LLM    │
│  • Questions     │                                   └────────┬──────────┘
└──────────────────┘                                            │
                                                                ▼
                                                     Interactive Q&A Chat
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit 1.35+ |
| **STT (English)** | OpenAI Whisper (local, `base` model) |
| **STT (Hinglish)** | Sarvam AI `saaras:v2.5` STT-Translate API |
| **LLM** | Mistral AI `mistral-small-latest` |
| **Orchestration** | LangChain LCEL |
| **Embeddings** | `all-MiniLM-L6-v2` via HuggingFace |
| **Vector Store** | ChromaDB (in-memory) |
| **Audio** | pydub + yt-dlp + ffmpeg |

---

## ⚡ Quick Start (Local)

### 1 — Clone the repo

```bash
git clone https://github.com/DevS-2004/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2 — Install dependencies

```bash
# Using uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or pip
pip install -r requirements.txt
```

> **Note:** ffmpeg must be installed separately.
> - Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
> - macOS: `brew install ffmpeg`
> - Linux: `sudo apt install ffmpeg`

### 3 — Configure API keys

```bash
cp .env.example .env   # or create .env manually
```

Edit `.env`:

```env
MISTRAL_API_KEY  = "your_mistral_key"
SARVAM_API_KEY   = "your_sarvam_key"   # only needed for Hinglish mode
WHISPER_MODEL    = "small"             # tiny | base | small | medium | large
```

Get your keys:
- **Mistral AI** → [console.mistral.ai](https://console.mistral.ai/)
- **Sarvam AI** → [dashboard.sarvam.ai](https://dashboard.sarvam.ai/)

### 4 — Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Community Cloud

1. **Push this repo to GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
MISTRAL_API_KEY = "your_mistral_key"
SARVAM_API_KEY  = "your_sarvam_key"
WHISPER_MODEL   = "tiny"
```

5. Click **Deploy** — you'll get a public URL in ~3 minutes 🎉

---

## 📁 Project Structure

```
AI-Video-Assistant/
├── app.py                    # Streamlit UI (entry point)
├── main.py                   # CLI entry point (alternative)
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages (ffmpeg for Streamlit Cloud)
├── .streamlit/
│   ├── config.toml           # Theme & server config
│   └── secrets.toml.example  # Secrets template
├── core/
│   ├── transcriber.py        # Whisper + Sarvam AI STT
│   ├── summarizer.py         # Map-reduce LLM summarisation
│   ├── extractor.py          # Action items, decisions, questions
│   ├── rag_engine.py         # RAG chain (retriever + LLM)
│   └── vector_store.py       # ChromaDB vector store
├── utils/
│   └── audio_processor.py    # yt-dlp download, pydub convert, chunk
└── downloades/               # Temporary audio files (gitignored)
```

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | — | Mistral AI API key (required) |
| `SARVAM_API_KEY` | — | Sarvam AI key (required for Hinglish) |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam model variant |

---

## 📄 License

MIT © [Devendra Singh](https://github.com/DevS-2004)
