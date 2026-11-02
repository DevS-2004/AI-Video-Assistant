---
title: AI Video Assistant
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🎬 AI Video & Meeting Assistant

> **An end-to-end meeting intelligence platform.** Ingest any YouTube video or audio/video recording to automatically transcribe speech, generate executive map-reduce summaries, calculate meeting health scorecards, draft follow-up collateral, track interactive tasks, and query conversations in real-time using ChromaDB Vector RAG.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1c3c3c?logo=langchain&logoColor=white)](https://langchain.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-open--mistral--7b-ff7000)](https://mistral.ai/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-fc5200)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Key Capabilities

### 📊 1. Executive Overview & Media Player
- **Integrated Video/Audio Player**: Watch embedded YouTube videos or stream uploaded audio directly alongside your notes and transcript.
- **⚡ Executive TL;DR**: A crisp 3-sentence executive summary detailing context, single primary decision, and critical next milestone.
- **⏱️ Smart Timeline Chapters `[mm:ss]`**: Chronological breakdown with timestamps and 1-sentence topic summaries.
- **🎯 Meeting Health & Productivity Scorecard**: Objective scoring (Productivity Score, Actionability Rating, Clarity/Focus Score, Pacing Density) plus 3 actionable tips for future meetings.
- **Stat Metrics**: Word counts, estimated reading time, speaking duration, sentence density, and pipeline benchmark execution times in seconds.
- **Interactive Keywords & Tag Cloud**: Keyword frequency analytics with Plotly bar visualization and interactive badge tags.
- **Live Search Highlighting**: Live text search within the transcript explorer with `<mark>` match highlighting and hit counter.

### 🔍 2. Deep Analysis & Interactive Tracking
- **Action Items & Deliverables**: Automated extraction of tasks with responsible owners and deadlines.
- **☑️ Interactive Task Checklist**: Live checkable task board with progress bar (`X of Y tasks completed (Z%)`).
- **Key Decisions Made**: Clear bulleted record of strategic resolutions and consensus.
- **Topics & Themes**: Core discussion topics tagged with contextual descriptions.
- **Open Questions & Blockers**: Surfaces unresolved points needing post-meeting follow-up.
- **⚠️ Risk & Blocker Register**: Discovers organizational/technical risks, assigns severity ratings (**HIGH / MEDIUM / LOW**), and notes discussed mitigations.
- **👥 Mentioned Entities (NER)**: Identifies People, Organizations, Tools & Tech, and Key Timelines/Dates.

### ⚡ 3. Executive AI Studio
- **✉️ Team Follow-Up Email Generator**: Drafts polished post-meeting recap emails with in-browser editor and `.txt` export.
- **📋 Formal Board-Ready Meeting Minutes**: Complete formal governance document with attendee list, agenda, resolutions, and action register.
- **📊 5-Slide Presentation Deck Outline**: Converts discussions into a ready-to-present executive deck outline with slide titles, bullets, and speaker notes.
- **🎟️ Jira & Linear Backlog Ticket Cards**: Converts action items into standardized engineering tickets with checkable acceptance criteria.
- **🌐 Multi-Language Summary Translator**: Translates executive summaries into **Hindi, Spanish, French, German, Japanese, Portuguese, Italian, or Arabic**.
- **🔬 Custom AI Extraction Query**: Ad-hoc natural language prompts directly evaluated against the session transcript.

### 💬 4. Grounded RAG Chat with Pinned Takeaways
- **ChromaDB Vector Retrieval**: Answers are factual and grounded strictly in the meeting transcript.
- **💡 1-Click Inquiry Chips**: Suggested prompts for instant insights.
- **📌 Pinned Takeaways Drawer**: Pin and bookmark assistant responses into a persistent takeaway drawer, exportable as a standalone memo.

### 📚 5. Persistent Session History Hub
- **Local JSON Persistence**: All meetings are saved locally in `history_data.json` across browser refreshes and application restarts.
- **Searchable Catalog**: Filter past sessions by title, date, or keywords with sentiment badges and duration pills.
- **1-Click Workspace Restore**: Restore past transcripts and rebuild vector search indexes on demand.
- **🔄 Multi-Meeting Comparison Tool**: Select any two historical sessions to compare topic evolution, carried-over tasks, decision velocity, and sentiment dynamics.

### 📥 6. Multi-Format Export
- Export clean deliverables in **Plain Text (`.txt`)**, **GitHub Flavored Markdown (`.md`)**, or complete structured **JSON packages**.

---

## 🏗️ Technical Architecture

```
                                      User Input
                           (YouTube URL or Audio/Video Upload)
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │   Audio Processor    │  yt-dlp + pydub
                                │ (16kHz WAV Chunking) │  ffmpeg mono conversion
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │  Speech Recognition  │  Whisper (English) /
                                │     (Transcriber)    │  Sarvam AI Saaras (Hinglish)
                                └──────────┬───────────┘
                                           │
              ┌────────────────────────────┴───────────────────────────┐
              ▼                                                        ▼
   ┌──────────────────────┐                                 ┌──────────────────────┐
   │ LangChain LCEL Chains│                                 │   Vector Store RAG   │
   │ • Map-Reduce Summary │                                 │ • all-MiniLM-L6-v2   │
   │ • Action Items & TLDR│                                 │ • ChromaDB Embeddings│
   │ • Decisions & Topics │                                 │ • Context Retriever  │
   │ • Executive AI Studio│                                 │ • Mistral Chat Chain │
   └──────────┬───────────┘                                 └──────────┬───────────┘
              │                                                        │
              └────────────────────────────┬───────────────────────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │ Streamlit SaaS UI    │  Overview · Deep Analysis
                                │  & Local Persistence │  AI Studio · RAG Chat
                                │ (history_data.json)  │  History Hub · Export
                                └──────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Frontend UI** | Streamlit 1.35+ | Custom editorial SaaS theme with responsive grid & tabs |
| **Speech-to-Text (EN)** | OpenAI Whisper | Local offline model (`base`, `small`, `medium`) |
| **Speech-to-Text (HI)** | Sarvam AI STT API | `saaras:v2.5` model for natural Hinglish audio |
| **LLM Inference** | Mistral AI | `open-mistral-7b` / `ministral-8b-latest` / `mistral-small` |
| **Orchestration** | LangChain Core & LCEL | Modular prompt chains, parsers, and runnables |
| **Embeddings** | HuggingFace | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | ChromaDB | Local vector indexing and semantic retrieval |
| **Audio Processing** | pydub + yt-dlp + ffmpeg | Audio extraction, sample rate conversion & chunking |
| **Persistence** | History Manager (JSON) | Persistent session storage across app lifecycles |

---

## ⚡ Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/DevS-2004/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2. Set Up Virtual Environment & Dependencies
```powershell
# Using uv (fastest)
uv venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt

# Or using standard python venv
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on FFmpeg:** FFmpeg is required for audio extraction.
> - **Windows:** `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
> - **macOS:** `brew install ffmpeg`
> - **Linux:** `sudo apt install ffmpeg`

### 3. Configure API Credentials
Create a `.env` file in the root directory (or copy from `.streamlit/secrets.toml.example`):
```env
MISTRAL_API_KEY="your_mistral_api_key"
SARVAM_API_KEY="your_sarvam_api_key"    # Optional: only required for Hinglish audio
MISTRAL_MODEL="open-mistral-7b"         # open-mistral-7b | ministral-8b-latest | mistral-small-latest
WHISPER_MODEL="base"                    # tiny | base | small | medium | large
```

Get API keys:
- **Mistral AI:** [console.mistral.ai](https://console.mistral.ai/)
- **Sarvam AI:** [dashboard.sarvam.ai](https://dashboard.sarvam.ai/)

### 4. Launch the Application
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📁 Repository Structure

```
AI-Video-Assistant/
├── app.py                     # Main Streamlit application & interactive UI
├── requirements.txt           # Python dependencies
├── packages.txt               # System dependencies for cloud deployment (ffmpeg)
├── Dockerfile                 # Container deployment specification
├── history_data.json          # Persistent session storage (created automatically)
├── core/
│   ├── transcriber.py         # Whisper & Sarvam AI STT orchestration
│   ├── summarizer.py          # Map-reduce hierarchical summarization chains
│   ├── extractor.py           # Action items, TL;DR, risks, entities, emails, minutes
│   ├── rag_engine.py          # Grounded vector search RAG pipeline
│   └── vector_store.py        # ChromaDB vector store initialization
├── utils/
│   ├── audio_processor.py     # yt-dlp downloader, WAV conversion & audio chunker
│   ├── ffmpeg_setup.py        # Automatic FFmpeg discovery & verification
│   └── history_manager.py     # Local persistence & session management utilities
└── .streamlit/
    ├── config.toml            # Streamlit theme & server configuration
    └── secrets.toml.example   # Secrets template for Streamlit Cloud
```

---

## ☁️ Deployment

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) and create a **New app**.
3. Point to your repository and set Main file path to `app.py`.
4. Under **Advanced settings → Secrets**, paste your API credentials:
   ```toml
   MISTRAL_API_KEY = "your_key"
   SARVAM_API_KEY = "your_key"
   WHISPER_MODEL = "base"
   ```
5. Click **Deploy**.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

Author: **[Devendra Singh](https://github.com/DevS-2004)**
