import streamlit as st
import time
import os
import re
import html
from collections import Counter
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env locally; silently skipped on Streamlit Cloud
except ImportError:
    pass

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
    extract_topics,
    generate_tldr,
    extract_named_entities,
    extract_risks,
    generate_followup_email,
    generate_formal_minutes,
    translate_text,
    custom_extract,
    generate_chapters,
    generate_presentation_outline,
    generate_jira_tickets,
    generate_meeting_scorecard,
    compare_meetings,
)
from core.rag_engine import build_rag_chain, ask_question
from utils.audio_processor import process_input, process_uploaded_file
from utils.ffmpeg_setup import ensure_ffmpeg
from utils.history_manager import (
    load_all_history,
    save_session,
    delete_session,
    clear_all_history,
)

ensure_ffmpeg()

# ── Read secrets from Streamlit Cloud (st.secrets) or fall back to .env ────────
def get_secret(key: str, default: str = "") -> str:
    """Works on both Streamlit Cloud (st.secrets) and local (.env via dotenv)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

for secret_key in [
    "MISTRAL_API_KEY",
    "MISTRAL_MODEL",
    "SARVAM_API_KEY",
    "WHISPER_MODEL",
    "SARVAM_STT_MODEL",
]:
    val = get_secret(secret_key)
    if val and secret_key not in os.environ:
        os.environ[secret_key] = val

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant | Meeting Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Clean Editorial SaaS Theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cal+Sans:wght@600&family=Geist+Mono:wght@400;500&display=swap');

/* ─── Design Tokens ─── */
:root {
    --bg:          #0d0d0d;
    --bg-2:        #141414;
    --surface:     #1a1a1a;
    --surface-2:   #202020;
    --border:      #2a2a2a;
    --border-hi:   #3a3a3a;
    --accent:      #ff6b35;
    --accent-dim:  rgba(255, 107, 53, 0.12);
    --accent-2:    #00d2ff;
    --text:        #ededed;
    --text-2:      #a0a0a0;
    --text-3:      #666666;
    --success:     #22c55e;
    --warning:     #f59e0b;
    --danger:      #ef4444;
    --radius:      10px;
    --radius-lg:   16px;
}

/* ─── Base Reset ─── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-size: 15px;
    -webkit-font-smoothing: antialiased;
}

.stApp { background: var(--bg) !important; }

/* Subtle noise-texture background overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ─── Header Branding ─── */
.brand-mark {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.brand-icon {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    flex-shrink: 0;
}
.brand-name {
    font-size: 1.0rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}
.brand-tag {
    font-size: 0.7rem;
    color: var(--text-3);
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ─── Page Hero ─── */
.page-hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.page-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 0.5rem;
}
.page-title span {
    color: var(--accent);
}
.page-subtitle {
    font-size: 0.95rem;
    color: var(--text-2);
    font-weight: 400;
    line-height: 1.6;
    max-width: 520px;
}
.page-meta {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.page-meta-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: var(--text-3);
}
.page-meta-dot {
    width: 6px; height: 6px;
    background: var(--success);
    border-radius: 50%;
}

/* ─── Cards ─── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover { border-color: var(--border-hi); }

.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-3);
    display: flex; align-items: center; gap: 0.4rem;
}
.card-accent-bar {
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    border-radius: 2px;
    margin-bottom: 1rem;
    width: 36px;
}
.card-body {
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--text-2);
    white-space: pre-line;
}
.card-title-text {
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1.3;
}

/* ─── Stat Pill Row ─── */
.stat-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.stat-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.82rem;
    color: var(--text-2);
    transition: border-color 0.2s;
}
.stat-pill:hover { border-color: var(--border-hi); }
.stat-pill-value {
    font-weight: 700;
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.stat-pill-icon { font-size: 1rem; }

/* ─── Section Divider ─── */
.section-sep {
    height: 1px;
    background: var(--border);
    margin: 1.75rem 0;
}

/* ─── Keyword Tags ─── */
.tag-cloud { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.tag {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.28rem 0.75rem;
    font-size: 0.78rem;
    color: var(--text-2);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.15s;
}
.tag:hover {
    border-color: var(--accent);
    color: var(--accent);
}
.tag-count {
    background: var(--border-hi);
    border-radius: 4px;
    padding: 0.05rem 0.35rem;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-3);
}

/* ─── Inline Status Badge ─── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.pill-orange { background: rgba(255,107,53,.14); color: #ff6b35; border: 1px solid rgba(255,107,53,.3); }
.pill-blue   { background: rgba(0,210,255,.10);  color: #00d2ff; border: 1px solid rgba(0,210,255,.25); }
.pill-green  { background: rgba(34,197,94,.12);  color: #22c55e; border: 1px solid rgba(34,197,94,.25); }
.pill-amber  { background: rgba(245,158,11,.12); color: #f59e0b; border: 1px solid rgba(245,158,11,.25); }
.pill-red    { background: rgba(239,68,68,.12);  color: #ef4444; border: 1px solid rgba(239,68,68,.25); }
.pill-gray   { background: rgba(160,160,160,.1); color: #a0a0a0; border: 1px solid rgba(160,160,160,.2); }

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--surface) !important;
    padding: 4px !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    color: var(--text-2) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 18px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; }
.stTabs [aria-selected="true"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-hi) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
}

/* ─── Status Timeline ─── */
.status-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.55rem 0.85rem;
    border-radius: var(--radius);
    margin: 3px 0;
    font-size: 0.8rem;
    color: var(--text-2);
    background: var(--surface);
    border: 1px solid var(--border);
}
.s-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.s-pending { background: var(--border-hi); }
.s-active  { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: blink 1.2s infinite; }
.s-done    { background: var(--success); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ─── Transcript Box ─── */
.transcript-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    font-size: 0.83rem;
    line-height: 1.85;
    max-height: 360px;
    overflow-y: auto;
    color: var(--text-2);
    font-family: 'Geist Mono', 'Courier New', monospace;
    white-space: pre-wrap;
    word-break: break-word;
}
mark.hl {
    background: rgba(255,107,53,0.25);
    color: #fff;
    border-radius: 3px;
    padding: 1px 3px;
    font-weight: 600;
}

/* ─── Chat ─── */
.chat-wrap {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.chat-turn { margin-bottom: 1rem; }
.chat-role {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.role-you { color: var(--accent); }
.role-ai  { color: var(--accent-2); }
.bubble {
    display: inline-block;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    font-size: 0.875rem;
    line-height: 1.6;
    max-width: 85%;
}
.bubble-you {
    background: rgba(255,107,53,0.1);
    border: 1px solid rgba(255,107,53,0.25);
    color: var(--text);
}
.bubble-ai {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-2);
}

/* ─── Inputs & Controls ─── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.15) !important;
}
label { color: var(--text-3) !important; font-size: 0.78rem !important; font-weight: 500 !important; }

/* ─── Buttons ─── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.55rem 1.3rem !important;
    transition: opacity 0.15s, transform 0.15s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--border-hi) !important;
    color: var(--text) !important;
}
.stDownloadButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    border-radius: var(--radius) !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ─── Empty State ─── */
.empty-hero {
    padding: 4rem 2rem;
    text-align: center;
}
.empty-title {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.empty-sub {
    font-size: 0.9rem;
    color: var(--text-2);
    max-width: 440px;
    margin: 0 auto 2.5rem;
    line-height: 1.65;
}
.feat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-top: 1.5rem;
    text-align: left;
}
.feat-item {
    background: var(--surface);
    padding: 1.4rem 1.6rem;
    transition: background 0.15s;
}
.feat-item:hover { background: var(--surface-2); }
.feat-emoji { font-size: 1.5rem; margin-bottom: 0.6rem; }
.feat-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.25rem;
}
.feat-detail {
    font-size: 0.78rem;
    color: var(--text-3);
    line-height: 1.55;
}

/* ─── Progress bar color ─── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text-2) !important; }

/* ─── Scrollbars ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ─── TL;DR Brief ─── */
.tldr-card {
    background: linear-gradient(135deg, rgba(255, 107, 53, 0.08), rgba(0, 210, 255, 0.04));
    border: 1px solid rgba(255, 107, 53, 0.35);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
.tldr-title {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.tldr-text {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #f3f4f6;
    font-weight: 450;
}

/* ─── Timing Chips ─── */
.timing-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.timing-chip {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.72rem;
    color: var(--text-3);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.timing-chip strong {
    color: var(--text-2);
}

/* ─── Studio Cards ─── */
.studio-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
}
.studio-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.studio-desc {
    font-size: 0.82rem;
    color: var(--text-2);
    margin-bottom: 1rem;
    line-height: 1.5;
}

/* ─── Pinned Chat Takeaways ─── */
.pinned-wrap {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}
.pinned-item {
    border-left: 2px solid var(--accent);
    padding-left: 0.85rem;
    margin-bottom: 0.85rem;
}
.pinned-q {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.25rem;
}
.pinned-a {
    font-size: 0.84rem;
    color: var(--text-2);
    line-height: 1.6;
}

/* ─── History Session Cards ─── */
.history-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}
.history-card:hover {
    border-color: var(--border-hi);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.history-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.35rem;
}
.history-meta {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 0.75rem;
}
.history-preview {
    font-size: 0.84rem;
    color: var(--text-2);
    line-height: 1.6;
    max-height: 70px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    margin-bottom: 1rem;
}

/* ─── Misc overrides ─── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }
.stAlert { border-radius: var(--radius) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ───────────────────────────────────────────────────────────
STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "that", "is", "was", "for", "on", "it", "with",
    "as", "by", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but",
    "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an",
    "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about",
    "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like",
    "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number", "no",
    "way", "could", "people", "my", "than", "first", "been", "call", "who", "its", "now",
    "find", "long", "down", "day", "did", "get", "come", "made", "may", "part", "i", "you",
    "they", "he", "she", "me", "us", "our", "ours", "yours", "his", "hers", "just", "also",
    "really", "going", "know", "think", "okay", "yeah", "well", "right", "want", "need",
    "thing", "things", "talk", "sure", "mean", "actually", "basically", "let", "something",
    "very", "much", "even", "back", "take", "any", "good", "new", "give", "most", "us", "are"
}

def compute_stats(text: str) -> dict:
    """Computes word count, sentences, reading time, characters, and estimated speaking duration."""
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(1, len(sentences))
    char_count = len(text)
    reading_time_min = max(1, round(word_count / 200))
    duration_min = max(1, round(word_count / 140))
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": char_count,
        "reading_time_min": reading_time_min,
        "duration_min": duration_min,
    }

def get_top_keywords(text: str, n: int = 15) -> list[tuple[str, int]]:
    """Extracts top n non-stopword keywords from text by frequency."""
    tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in tokens if w not in STOPWORDS]
    return Counter(filtered).most_common(n)

def analyze_sentiment(text: str) -> dict:
    """Performs keyword-based meeting sentiment analysis returning label, badge, and signals."""
    positive_words = {
        "success", "successful", "great", "good", "excellent", "positive", "agree",
        "resolved", "solved", "progress", "win", "benefit", "effective", "happy",
        "love", "perfect", "improve", "improvement", "growth", "advantage", "achieve",
        "opportunity", "promising", "recommend", "collaborate", "partnership", "accomplished"
    }
    negative_words = {
        "problem", "issue", "bug", "delay", "delayed", "blocker", "blocked", "fail",
        "failure", "difficult", "risk", "risks", "concern", "negative", "conflict",
        "error", "bad", "loss", "drop", "decline", "disagree", "fault", "stress",
        "threat", "bottleneck", "obstacle", "critical", "vulnerable", "struggle"
    }
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    total = pos_count + neg_count

    if total == 0:
        return {
            "label": "Neutral / Informative",
            "score": 0.0,
            "badge_class": "badge-cyan",
            "icon": "⚖️",
            "pos_count": 0,
            "neg_count": 0,
            "desc": "Balanced tone with no significant positive or critical bias."
        }
    ratio = (pos_count - neg_count) / total
    if ratio > 0.15:
        return {
            "label": "Constructive & Positive",
            "score": round(ratio, 2),
            "badge_class": "badge-green",
            "icon": "📈",
            "pos_count": pos_count,
            "neg_count": neg_count,
            "desc": f"Constructive tone with {pos_count} positive signals vs {neg_count} risks/concerns."
        }
    elif ratio < -0.15:
        return {
            "label": "Critical / Attention Needed",
            "score": round(ratio, 2),
            "badge_class": "badge-danger",
            "icon": "⚠️",
            "pos_count": pos_count,
            "neg_count": neg_count,
            "desc": f"Focuses on problem resolution and risk mitigation ({neg_count} risk signals vs {pos_count} positive)."
        }
    else:
        return {
            "label": "Balanced & Objective",
            "score": round(ratio, 2),
            "badge_class": "badge-purple",
            "icon": "⚖️",
            "pos_count": pos_count,
            "neg_count": neg_count,
            "desc": f"Evenly balanced discussion ({pos_count} positive, {neg_count} concerns)."
        }

def highlight_text(text: str, query: str) -> str:
    """Safely escapes HTML and wraps query matches in high-contrast <mark> tags."""
    if not query or not query.strip():
        return html.escape(text)
    escaped_query = re.escape(query.strip())
    escaped_text = html.escape(text)
    pattern = re.compile(f"({escaped_query})", re.IGNORECASE)
    return pattern.sub(r'<mark class="hl">\1</mark>', escaped_text)

def generate_markdown_report(r: dict) -> str:
    """Creates a comprehensive, beautifully structured GitHub Flavored Markdown report."""
    stats = r.get("stats", {})
    sentiment = r.get("sentiment", {})
    tldr_section = f"## ⚡ Executive TL;DR\n\n{r.get('tldr', '')}\n\n---\n\n" if r.get('tldr') else ""
    risks_section = f"## ⚠️ Risks & Blockers\n\n{r.get('risks', '')}\n\n---\n\n" if r.get('risks') else ""
    entities_section = f"## 👥 Identified Entities\n\n{r.get('entities', '')}\n\n---\n\n" if r.get('entities') else ""
    
    return (
        f"# 🎬 Meeting Intelligence Report: {r['title']}\n\n"
        f"**Generated:** {r.get('timestamp', 'N/A')} | **AI Assistant:** Mistral AI & Whisper / Sarvam\n\n"
        f"---\n\n"
        f"## 📊 Executive Overview & Metrics\n"
        f"- **Word Count:** {stats.get('word_count', 0):,} words\n"
        f"- **Sentence Count:** {stats.get('sentence_count', 0):,} sentences\n"
        f"- **Estimated Reading Time:** ~{stats.get('reading_time_min', 1)} min\n"
        f"- **Estimated Speaking Duration:** ~{stats.get('duration_min', 1)} min\n"
        f"- **Sentiment:** {sentiment.get('icon', '')} {sentiment.get('label', 'Neutral')} ({sentiment.get('desc', '')})\n\n"
        f"---\n\n"
        f"{tldr_section}"
        f"## 📋 Executive Summary\n\n{r['summary']}\n\n"
        f"---\n\n"
        f"## 🏷️ Key Topics & Themes\n\n{r.get('topics', 'No distinct topics identified.')}\n\n"
        f"---\n\n"
        f"## ✅ Action Items & Owners\n\n{r['action_items']}\n\n"
        f"---\n\n"
        f"## 🔑 Key Decisions Made\n\n{r['key_decisions']}\n\n"
        f"---\n\n"
        f"## ❓ Open Questions & Unresolved Topics\n\n{r['open_questions']}\n\n"
        f"---\n\n"
        f"{risks_section}"
        f"{entities_section}"
        f"## 📝 Full Transcript\n\n```text\n{r['transcript']}\n```\n"
    )

# ─── Session State Initialization ───────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "pipeline_times": {},
    "history": [],
    "search_query": "",
    "selected_tab": 0,
    "bookmarks": [],
    "tldr": None,
    "entities": None,
    "risks": None,
    "email_draft": None,
    "minutes": None,
    "translation": None,
    "custom_result": None,
    "checklist": {},
    "chapters": None,
    "scorecard": None,
    "presentation": None,
    "jira_tickets": None,
    "diff_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state.history:
    st.session_state.history = load_all_history()

def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active": return "s-active"
    if s == "done":   return "s-done"
    return "s-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    t = st.session_state.pipeline_times.get(key)
    time_str = f" · {t}s" if t else ""
    st.markdown(
        f"""
        <div class="status-row">
            <div class="s-dot {css}"></div>
            <span>{icon}&nbsp;{label}{time_str}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_history_hub():
    sessions = load_all_history()
    st.session_state.history = sessions
    
    if not sessions:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 1rem;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">📂</div>
            <div style="font-weight:600;color:var(--text);margin-bottom:0.25rem;">No saved sessions yet</div>
            <div style="font-size:0.82rem;color:var(--text-3);max-width:380px;margin:0 auto;">
                Analyze a YouTube video or uploaded audio file to automatically save your meeting intelligence reports here.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Search & Clear controls
    search_col, clear_col = st.columns([4, 1], gap="medium")
    with search_col:
        hist_query = st.text_input(
            "Search history",
            placeholder="Search by title, topic, or date…",
            label_visibility="collapsed",
            key="hist_search_box",
        )
    with clear_col:
        if st.button("🗑️ Clear All", key="clear_all_sessions_hub_btn", type="secondary", use_container_width=True):
            clear_all_history()
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()

    filtered_sessions = sessions
    if hist_query.strip():
        q_lower = hist_query.strip().lower()
        filtered_sessions = [
            s for s in sessions 
            if q_lower in s.get("title", "").lower() 
            or q_lower in s.get("timestamp", "").lower()
            or q_lower in s.get("result", {}).get("summary", "").lower()
        ]

    # ── Multi-Meeting Comparison Tool ──
    if len(sessions) >= 2:
        with st.expander("🔄 Compare Two Meetings (Side-by-Side & Momentum Diff)", expanded=False):
            st.markdown('<div style="font-size:0.82rem;color:var(--text-2);margin-bottom:0.75rem;">Select any two past sessions to compare focus shifts, carried-over action items, decision velocity, and tone.</div>', unsafe_allow_html=True)
            cmp_col1, cmp_col2 = st.columns(2)
            titles = [f"{s.get('title', 'Untitled')} ({s.get('timestamp', 'Recent')})" for s in sessions]
            with cmp_col1:
                idx_a = st.selectbox("Meeting 1", range(len(sessions)), format_func=lambda i: titles[i], key="cmp_m1")
            with cmp_col2:
                idx_b = st.selectbox("Meeting 2", range(len(sessions)), index=min(1, len(sessions)-1), format_func=lambda i: titles[i], key="cmp_m2")
            
            if st.button("Generate Comparison Diff", key="do_cmp_btn"):
                with st.spinner("Analyzing differences, decision momentum, and topic shifts..."):
                    sess_a = sessions[idx_a].get("result", {})
                    sess_b = sessions[idx_b].get("result", {})
                    st.session_state.diff_result = compare_meetings(sess_a, sess_b)
            
            if st.session_state.diff_result:
                st.markdown(f'<div class="card" style="margin-top:1rem;">{st.session_state.diff_result}</div>', unsafe_allow_html=True)
                with st.expander("Copy comparison report", expanded=False):
                    st.code(st.session_state.diff_result, language="markdown")

    st.markdown(f'<div style="font-size:0.75rem;color:var(--text-3);margin-bottom:0.75rem;">Showing {len(filtered_sessions)} of {len(sessions)} saved session(s)</div>', unsafe_allow_html=True)

    for idx, s in enumerate(filtered_sessions):
        s_id = s.get("id") or str(idx)
        res = s.get("result", {})
        s_title = s.get("title", "Untitled Session")
        s_time = s.get("timestamp", "Recent")
        stats = res.get("stats", {})
        sentiment = res.get("sentiment", {})
        summary_preview = res.get("summary", "")[:260] + "…" if len(res.get("summary", "")) > 260 else res.get("summary", "")

        sent_cls = {
            "badge-green": "pill-green",
            "badge-danger": "pill-red",
            "badge-purple": "pill-orange",
        }.get(sentiment.get("badge_class", ""), "pill-gray")

        st.markdown(f"""
        <div class="history-card">
            <div class="history-title">🎬 {html.escape(s_title)}</div>
            <div class="history-meta">
                <span class="pill {sent_cls}">{sentiment.get('icon','⚖️')} {sentiment.get('label','Neutral')}</span>
                <span class="pill pill-gray">🕒 {s_time}</span>
                <span class="pill pill-gray">📝 {stats.get('word_count', 0):,} words</span>
                <span class="pill pill-gray">⏱ ~{stats.get('duration_min', 1)} min spoken</span>
            </div>
            <div class="history-preview">{html.escape(summary_preview)}</div>
        </div>
        """, unsafe_allow_html=True)

        act_col1, act_col2, act_col3 = st.columns([1.5, 1.2, 3], gap="small")
        with act_col1:
            if st.button("📂 Open Session", key=f"open_s_{s_id}_{idx}", use_container_width=True):
                if "rag_chain" not in res or res["rag_chain"] is None:
                    try:
                        res["rag_chain"] = build_rag_chain(res["transcript"])
                    except Exception:
                        res["rag_chain"] = None
                st.session_state.result = res
                st.session_state.pipeline_done = True
                st.session_state.chat_history = []
                st.session_state.bookmarks = []
                st.rerun()
        with act_col2:
            if st.button("🗑️ Delete", key=f"del_s_{s_id}_{idx}", type="secondary", use_container_width=True):
                delete_session(s_id)
                st.session_state.history = load_all_history()
                if st.session_state.result and st.session_state.result.get("id") == s_id:
                    st.session_state.result = None
                st.rerun()
        st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-mark">
        <div class="brand-icon">🎬</div>
        <div>
            <div class="brand-name">AI Video Assistant</div>
            <div class="brand-tag">Meeting Intelligence · RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:var(--border);margin:1rem 0;"></div>', unsafe_allow_html=True)

    # ── Input method ────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Input Source",
        ["🔗 YouTube URL", "📁 Upload File"],
        horizontal=True,
        label_visibility="visible",
    )

    source = ""
    uploaded_file = None

    if input_mode == "🔗 YouTube URL":
        source = st.text_input(
            "YouTube Video URL",
            placeholder="https://youtube.com/watch?v=...",
            help="Paste any valid YouTube video URL",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload Audio / Video",
            type=["mp4", "mp3", "wav", "webm", "m4a", "ogg", "mkv"],
            help="Supported: mp4, mp3, wav, webm, m4a, ogg, mkv  (Max 200MB)",
        )

    language = st.selectbox(
        "Language",
        ["english", "hinglish"],
        index=0,
        help="English → Whisper local · Hinglish → Sarvam AI STT",
    )

    # ── Advanced Settings ───────────────────────────────────────────────────────
    with st.expander("⚙️ Model Settings", expanded=False):
        current_model = os.getenv("MISTRAL_MODEL", "open-mistral-7b")
        model_options = ["open-mistral-7b", "ministral-8b-latest", "mistral-small-latest"]
        idx = model_options.index(current_model) if current_model in model_options else 0
        selected_model = st.selectbox("Mistral LLM", model_options, index=idx)
        os.environ["MISTRAL_MODEL"] = selected_model

        has_mistral = bool(os.getenv("MISTRAL_API_KEY"))
        has_sarvam  = bool(os.getenv("SARVAM_API_KEY"))
        st.markdown(
            f"""
            <div style="font-size:0.75rem;margin-top:0.5rem;color:var(--text-3);">
                Mistral Key: {'🟢 OK' if has_mistral else '🔴 Missing'}<br>
                Sarvam Key: {'🟢 OK' if has_sarvam else '🟡 Optional (Hinglish)'}
            </div>
            """,
            unsafe_allow_html=True
        )

    run_btn = st.button("▶  Analyse Meeting", use_container_width=True)

    # ── Live Pipeline Status ─────────────────────────────────────────────────────
    if st.session_state.pipeline_steps:
        st.markdown('<div style="height:1px;background:var(--border);margin:1rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-3);margin-bottom:0.5rem;">Pipeline</div>', unsafe_allow_html=True)
        pipeline_defs = [
            ("audio",      "🔊", "Audio Ingestion"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title"),
            ("summary",    "📋", "Summarization"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Index"),
        ]
        for step_k, icon, label in pipeline_defs:
            render_step_bar(label, step_k, icon)

    # ── Session History ──────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown('<div style="height:1px;background:var(--border);margin:1rem 0;"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-3);margin-bottom:0.5rem;">Recent Sessions ({len(st.session_state.history)})</div>', unsafe_allow_html=True)
        for h_idx, item in enumerate(st.session_state.history[:6]):
            short_title = item['title'][:24] + "…" if len(item['title']) > 24 else item['title']
            if st.button(
                f"📁  {short_title}",
                key=f"hist_btn_{h_idx}",
                use_container_width=True,
                type="secondary",
            ):
                res = item["result"]
                if "rag_chain" not in res or res["rag_chain"] is None:
                    try:
                        res["rag_chain"] = build_rag_chain(res["transcript"])
                    except Exception:
                        res["rag_chain"] = None
                st.session_state.result = res
                st.session_state.pipeline_done = True
                st.session_state.chat_history = []
                st.session_state.bookmarks = []
                st.rerun()

        if st.button("🗑  Clear History", key="clear_hist_btn", type="secondary", use_container_width=True):
            clear_all_history()
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()

# ─── Main Hero Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="page-title">AI Video<br><span>Assistant</span></div>
    <div class="page-subtitle">
        Transcribe recordings, extract decisions, action items, risks &amp; entities, then
        chat with your meetings using grounded RAG.
    </div>
    <div class="page-meta">
        <div class="page-meta-item">
            <div class="page-meta-dot"></div>
            Whisper · Sarvam AI
        </div>
        <div class="page-meta-item">· Mistral LLM</div>
        <div class="page-meta-item">· ChromaDB RAG</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Execution Logic ────────────────────────────────────────────────────────────
if run_btn:
    if input_mode == "🔗 YouTube URL" and not source.strip():
        st.error("⚠️ Please enter a valid YouTube URL.")
    elif input_mode == "📁 Upload File" and uploaded_file is None:
        st.error("⚠️ Please upload an audio or video file.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.pipeline_times = {}

        progress_box = st.empty()
        step_times = {}

        def update_step(k, status):
            st.session_state.pipeline_steps[k] = status
            if status == "active":
                step_times[k] = time.time()
            elif status == "done" and k in step_times:
                elapsed = round(time.time() - step_times[k], 1)
                st.session_state.pipeline_times[k] = elapsed

        try:
            # ── 1. Audio Ingestion ──────────────────────────────────────────────
            progress_box.info("🔊 Step 1/6: Ingesting audio and splitting into processing chunks...")
            update_step("audio", "active")
            if uploaded_file is not None:
                chunks = process_uploaded_file(uploaded_file)
            else:
                chunks = process_input(source.strip())
            update_step("audio", "done")

            # ── 2. Transcription ──────────────────────────────────────────────
            progress_box.info(f"📝 Step 2/6: Transcribing {len(chunks)} audio chunk(s) using {language} model...")
            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            # ── 3. Title Generation ────────────────────────────────────────────
            progress_box.info("🏷️ Step 3/6: Analyzing context to generate a descriptive session title...")
            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            # ── 4. Summarization ───────────────────────────────────────────────
            progress_box.info("📋 Step 4/6: Running map-reduce summarization with Mistral AI...")
            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            # ── 5. Extraction ──────────────────────────────────────────────────
            progress_box.info("🔍 Step 5/6: Extracting action items, key decisions, questions, topics & TL;DR...")
            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            topics = extract_topics(transcript)
            tldr = generate_tldr(transcript)
            update_step("extract", "done")

            # ── 6. RAG Engine ──────────────────────────────────────────────────
            progress_box.info("🧠 Step 6/6: Indexing transcript into ChromaDB vector store for instant Q&A...")
            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            # Compute stats, sentiment & keywords
            stats = compute_stats(transcript)
            sentiment = analyze_sentiment(transcript)
            keywords = get_top_keywords(transcript, n=15)
            timestamp_str = time.strftime("%b %d, %Y · %H:%M")

            result_payload = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "tldr": tldr,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "topics": topics,
                "rag_chain": rag_chain,
                "chunks_count": len(chunks),
                "stats": stats,
                "sentiment": sentiment,
                "keywords": keywords,
                "timestamp": timestamp_str,
                "pipeline_times": dict(st.session_state.pipeline_times),
                "source_type": "youtube" if input_mode == "🔗 YouTube URL" else "file",
                "source_url": source.strip() if input_mode == "🔗 YouTube URL" else (uploaded_file.name if uploaded_file else ""),
            }

            st.session_state.result = result_payload
            st.session_state.pipeline_done = True

            # Save to persistent disk history
            save_session(result_payload)
            st.session_state.history = load_all_history()

            progress_box.success("✨ Analysis completed successfully!")
            time.sleep(0.5)
            progress_box.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_box.error(f"❌ Analysis failed: {e}")

# ─── Results View (Tabs) ────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    stats = r.get("stats", compute_stats(r["transcript"]))
    sentiment = r.get("sentiment", analyze_sentiment(r["transcript"]))
    keywords = r.get("keywords", get_top_keywords(r["transcript"]))
    p_times = r.get("pipeline_times", {})

    tab_overview, tab_analysis, tab_studio, tab_chat, tab_history, tab_export = st.tabs([
        "📊 Overview",
        "🔍 Deep Analysis",
        "⚡ AI Studio",
        "💬 RAG Chat",
        "📚 History",
        "📥 Export",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ════════════════════════════════════════════════════════════════════════════
    with tab_overview:
        # ── Title + Sentiment + Timestamp ──
        sent_pill_cls = {
            "badge-green": "pill-green",
            "badge-danger": "pill-red",
            "badge-purple": "pill-orange",
        }.get(sentiment.get("badge_class", ""), "pill-gray")

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Session Title</div>
                <div class="card-accent-bar"></div>
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.75rem;">
                    <div class="card-title-text">{r['title']}</div>
                    <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">
                        <span class="pill {sent_pill_cls}">{sentiment.get('icon','⚖️')} {sentiment.get('label','Neutral')}</span>
                        <span class="pill pill-gray">🕒 {r.get('timestamp','Recent')}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── TL;DR Executive Brief ──
        if r.get("tldr"):
            st.markdown(
                f"""
                <div class="tldr-card">
                    <div class="tldr-title">⚡ Executive TL;DR Brief</div>
                    <div class="tldr-text">{r['tldr']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ── Stat Pills Row ──
        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-pill">
                    <span class="stat-pill-icon">📝</span>
                    <span class="stat-pill-value">{stats.get('word_count',0):,}</span> words
                </div>
                <div class="stat-pill">
                    <span class="stat-pill-icon">💬</span>
                    <span class="stat-pill-value">{stats.get('sentence_count',0):,}</span> sentences
                </div>
                <div class="stat-pill">
                    <span class="stat-pill-icon">⏱</span>
                    ~<span class="stat-pill-value">{stats.get('reading_time_min',1)}</span> min read
                </div>
                <div class="stat-pill">
                    <span class="stat-pill-icon">🎙</span>
                    ~<span class="stat-pill-value">{stats.get('duration_min',1)}</span> min spoken
                </div>
                <div class="stat-pill">
                    <span class="stat-pill-icon">🔀</span>
                    <span class="stat-pill-value">{r.get('chunks_count',1)}</span> chunk(s)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Pipeline Timing Benchmark (if available) ──
        if p_times:
            timing_html = '<div class="timing-row">'
            icons = {"audio": "🔊 Audio", "transcript": "📝 Transcribe", "title": "🏷️ Title", "summary": "📋 Summary", "extract": "🔍 Extract", "rag": "🧠 RAG"}
            for k, t_val in p_times.items():
                lbl = icons.get(k, k)
                timing_html += f'<div class="timing-chip">{lbl}: <strong>{t_val}s</strong></div>'
            timing_html += '</div>'
            st.markdown(timing_html, unsafe_allow_html=True)

        # ── Media Player Preview (YouTube / Video) ──
        if r.get("source_type") == "youtube" and r.get("source_url"):
            with st.expander("🎬 Watch Video Recording", expanded=False):
                st.video(r["source_url"])

        # ── Smart Chapters & Meeting Health Row ──
        ov_tool_col1, ov_tool_col2 = st.columns(2, gap="medium")
        with ov_tool_col1:
            with st.expander("⏱️ Smart Chapters & Timeline [mm:ss]", expanded=False):
                if not r.get("chapters"):
                    if st.button("Generate Timeline Chapters", key="gen_chapters_btn", use_container_width=True):
                        with st.spinner("Analyzing timeline and generating chapters..."):
                            r["chapters"] = generate_chapters(r["transcript"], stats.get("duration_min", 15))
                            save_session(r)
                            st.rerun()
                    else:
                        st.caption("Auto-segment video into logical timestamped chapters.")
                else:
                    st.markdown(r["chapters"])
                    st.code(r["chapters"], language="text")

        with ov_tool_col2:
            with st.expander("🎯 Meeting Health & Productivity Scorecard", expanded=False):
                if not r.get("scorecard"):
                    if st.button("Generate Productivity Scorecard", key="gen_scorecard_btn", use_container_width=True):
                        with st.spinner("Auditing meeting health and effectiveness..."):
                            r["scorecard"] = generate_meeting_scorecard(r["transcript"], stats.get("word_count", 0), stats.get("duration_min", 15))
                            save_session(r)
                            st.rerun()
                    else:
                        st.caption("Objective productivity score, actionability rating & tips.")
                else:
                    st.markdown(r["scorecard"])

        # ── Summary Card ──
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">📋 Executive Summary</div>
                <div class="card-accent-bar"></div>
                <div class="card-body">{r['summary']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.expander("Copy summary text", expanded=False):
            st.code(r["summary"], language="markdown")

        # ── Keywords ──
        col_kw1, col_kw2 = st.columns([3, 2], gap="medium")

        with col_kw1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">🏷 Keyword Frequency</div><div class="card-accent-bar"></div>', unsafe_allow_html=True)
            if keywords:
                df_kw = pd.DataFrame(keywords[:10], columns=["Keyword", "Frequency"])
                if HAS_PLOTLY:
                    import plotly.express as px
                    fig = px.bar(
                        df_kw, x="Frequency", y="Keyword",
                        orientation="h", color="Frequency",
                        color_continuous_scale=["#2a2a2a", "#ff6b35"],
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#a0a0a0", family="Inter"),
                        margin=dict(l=0, r=0, t=5, b=5),
                        height=260,
                        coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed"),
                    )
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(df_kw.set_index("Keyword"))
            else:
                st.caption("No significant keywords found.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_kw2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">🏷 Keyword Tags</div><div class="card-accent-bar"></div>', unsafe_allow_html=True)
            if keywords:
                pills_html = '<div class="tag-cloud">'
                for kw, cnt in keywords:
                    pills_html += f'<span class="tag">{kw}<span class="tag-count">{cnt}</span></span>'
                pills_html += '</div>'
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.caption("No keywords available.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Transcript Viewer ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📝 Transcript Explorer</div><div class="card-accent-bar"></div>', unsafe_allow_html=True)

        search_q = st.text_input(
            "Search transcript",
            placeholder="Search for any word or phrase…",
            key="transcript_search_input",
            label_visibility="collapsed",
        )
        if search_q.strip():
            hits = len(re.findall(re.escape(search_q.strip()), r["transcript"], re.IGNORECASE))
            st.markdown(
                f'<div style="font-size:0.78rem;color:var(--accent);margin-bottom:0.5rem;">'
                f'🔍 {hits} match(es) for "<em>{html.escape(search_q.strip())}</em>"</div>',
                unsafe_allow_html=True
            )

        highlighted_body = highlight_text(r["transcript"], search_q)
        st.markdown(f'<div class="transcript-box">{highlighted_body}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Copy raw transcript", expanded=False):
            st.code(r["transcript"], language="text")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2: DEEP ANALYSIS
    # ════════════════════════════════════════════════════════════════════════════
    with tab_analysis:
        col_act, col_dec = st.columns(2, gap="medium")

        with col_act:
            st.markdown(
                f"""
                <div class="card" style="min-height:280px;">
                    <div class="card-label">✅ Action Items</div>
                    <div class="card-accent-bar"></div>
                    <div class="card-body">{r['action_items']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander("Copy action items", expanded=False):
                st.code(r["action_items"], language="markdown")

            # ── Interactive Task Checklist ──
            task_lines = [l.strip() for l in r['action_items'].split("\n") if l.strip() and (l.strip()[0].isdigit() or l.strip().startswith("-") or l.strip().startswith("*"))]
            if task_lines:
                with st.expander(f"☑️ Interactive Task Checklist ({len(task_lines[:10])})", expanded=True):
                    checked_count = 0
                    for l_idx, line_text in enumerate(task_lines[:10]):
                        clean_text = re.sub(r'^\d+[\.\)]\s*', '', line_text).lstrip("-* ")
                        is_chk = st.checkbox(clean_text, key=f"act_chk_{l_idx}", value=st.session_state.checklist.get(l_idx, False))
                        st.session_state.checklist[l_idx] = is_chk
                        if is_chk:
                            checked_count += 1
                    pct = round((checked_count / len(task_lines[:10])) * 100) if task_lines[:10] else 0
                    st.progress(pct / 100)
                    st.caption(f"Completion: {checked_count} of {len(task_lines[:10])} tasks done ({pct}%)")

        with col_dec:
            st.markdown(
                f"""
                <div class="card" style="min-height:280px;">
                    <div class="card-label">🔑 Key Decisions</div>
                    <div class="card-accent-bar"></div>
                    <div class="card-body">{r['key_decisions']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander("Copy decisions", expanded=False):
                st.code(r["key_decisions"], language="markdown")

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">🏷 Topics &amp; Themes</div>
                <div class="card-accent-bar"></div>
                <div class="card-body">{r.get('topics', 'No distinct topics identified.')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">❓ Open Questions &amp; Follow-ups</div>
                <div class="card-accent-bar"></div>
                <div class="card-body">{r['open_questions']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.expander("Copy open questions", expanded=False):
            st.code(r["open_questions"], language="markdown")

        # ── Risks & Blockers Section ──
        st.markdown(
            """
            <div class="card">
                <div class="card-label">⚠️ Risks &amp; Blockers Register</div>
                <div class="card-accent-bar"></div>
            """,
            unsafe_allow_html=True
        )
        if not r.get("risks"):
            if st.button("🔍 Extract Risks & Blockers with Severity", key="extract_risks_btn"):
                with st.spinner("Analyzing transcript for risks and severity..."):
                    r["risks"] = extract_risks(r["transcript"])
                    save_session(r)
                    st.rerun()
            else:
                st.caption("Click above to identify blockers, severity ratings, and discussed mitigations.")
        else:
            st.markdown(f'<div class="card-body">{r["risks"]}</div>', unsafe_allow_html=True)
            with st.expander("Copy risk register", expanded=False):
                st.code(r["risks"], language="markdown")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Mentioned Entities (NER) Section ──
        st.markdown(
            """
            <div class="card">
                <div class="card-label">👥 Mentioned Entities (People, Orgs, Tech, Dates)</div>
                <div class="card-accent-bar"></div>
            """,
            unsafe_allow_html=True
        )
        if not r.get("entities"):
            if st.button("🔍 Extract Named Entities", key="extract_entities_btn"):
                with st.spinner("Extracting individuals, companies, tools, and timelines..."):
                    r["entities"] = extract_named_entities(r["transcript"])
                    save_session(r)
                    st.rerun()
            else:
                st.caption("Click above to extract key stakeholders, organizations, tools, and dates mentioned.")
        else:
            st.markdown(f'<div class="card-body">{r["entities"]}</div>', unsafe_allow_html=True)
            with st.expander("Copy entities list", expanded=False):
                st.code(r["entities"], language="markdown")
        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3: AI STUDIO (NEW IMPRESSIVE FEATURES)
    # ════════════════════════════════════════════════════════════════════════════
    with tab_studio:
        st.markdown(
            """
            <div class="card">
                <div class="card-label">⚡ Executive AI Studio</div>
                <div class="card-accent-bar"></div>
                <div style="font-size:0.86rem;color:var(--text-2);">
                    Turn your session analysis into polished, distribution-ready collateral: team follow-up emails,
                    formal meeting minutes, multi-language translations, and custom queries.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        studio_col1, studio_col2 = st.columns(2, gap="medium")

        # ── 1. Follow-up Email Generator ──
        with studio_col1:
            st.markdown(
                """
                <div class="studio-card">
                    <div class="studio-title">✉️ Follow-Up Email Draft</div>
                    <div class="studio-desc">Generate a professional, ready-to-send recap email with action items and decisions for your team.</div>
                """,
                unsafe_allow_html=True
            )
            if st.button("✨ Generate Follow-Up Email", key="gen_email_btn", use_container_width=True):
                with st.spinner("Drafting executive follow-up email..."):
                    st.session_state.email_draft = generate_followup_email(
                        r['title'], r['summary'], r['action_items'], r['key_decisions']
                    )
                    r['email_draft'] = st.session_state.email_draft
                    save_session(r)
                    st.rerun()

            current_email = r.get("email_draft") or st.session_state.email_draft
            if current_email:
                edited_email = st.text_area("Email Content", value=current_email, height=240, key="email_edit_area")
                st.download_button("📥 Download Email (.txt)", data=edited_email, file_name="follow_up_email.txt", mime="text/plain", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── 2. Formal Meeting Minutes ──
        with studio_col2:
            st.markdown(
                """
                <div class="studio-card">
                    <div class="studio-title">📋 Formal Meeting Minutes</div>
                    <div class="studio-desc">Generate official, boardroom-formatted meeting minutes with attendee list, decisions register, and action owners.</div>
                """,
                unsafe_allow_html=True
            )
            if st.button("📑 Generate Formal Minutes", key="gen_minutes_btn", use_container_width=True):
                with st.spinner("Drafting official meeting minutes..."):
                    st.session_state.minutes = generate_formal_minutes(
                        r['title'], r['transcript'], r['key_decisions'], r['action_items'], r.get('timestamp', 'Recent')
                    )
                    r['minutes'] = st.session_state.minutes
                    save_session(r)
                    st.rerun()

            current_minutes = r.get("minutes") or st.session_state.minutes
            if current_minutes:
                with st.expander("View & Copy Meeting Minutes", expanded=True):
                    st.markdown(current_minutes)
                    st.code(current_minutes, language="markdown")
                st.download_button("📥 Download Minutes (.md)", data=current_minutes, file_name="meeting_minutes.md", mime="text/markdown", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── 3. Multi-Language Summary Translation ──
        st.markdown(
            """
            <div class="studio-card">
                <div class="studio-title">🌐 Multi-Language Translator</div>
                <div class="studio-desc">Translate the executive summary into another language for global team members or stakeholders.</div>
            """,
            unsafe_allow_html=True
        )
        trans_col1, trans_col2 = st.columns([2, 3], gap="medium")
        with trans_col1:
            target_lang = st.selectbox("Target Language", ["Hindi", "Spanish", "French", "German", "Japanese", "Portuguese", "Italian", "Arabic"], index=0)
            trans_btn = st.button(f"Translate to {target_lang}", key="do_trans_btn", use_container_width=True)
        with trans_col2:
            if trans_btn:
                with st.spinner(f"Translating to {target_lang}..."):
                    trans_text = translate_text(r['summary'], target_lang)
                    if "translations" not in r:
                        r["translations"] = {}
                    r["translations"][target_lang] = trans_text
                    save_session(r)
                    st.session_state.translation = {target_lang: trans_text}
                    st.rerun()

            active_trans = r.get("translations", {})
            if active_trans:
                selected_lang = list(active_trans.keys())[-1]
                st.markdown(f"**Translated ({selected_lang}):**")
                st.markdown(f'<div class="card-body" style="font-size:0.88rem;background:var(--surface-2);padding:1rem;border-radius:8px;">{active_trans[selected_lang]}</div>', unsafe_allow_html=True)
                with st.expander("Copy translation", expanded=False):
                    st.code(active_trans[selected_lang], language="text")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 4. Custom AI Extraction Prompt ──
        st.markdown(
            """
            <div class="studio-card">
                <div class="studio-title">🔬 Custom AI Extraction Query</div>
                <div class="studio-desc">Ask the AI model to extract or evaluate anything custom from this specific meeting transcript.</div>
            """,
            unsafe_allow_html=True
        )
        custom_q = st.text_input("Custom Extraction Prompt", placeholder="e.g. List all budgetary numbers or quotes regarding pricing…", key="custom_extract_input", label_visibility="collapsed")
        if st.button("Run Custom Extraction", key="run_custom_extract_btn") and custom_q.strip():
            with st.spinner("Extracting from transcript..."):
                st.session_state.custom_result = custom_extract(r['transcript'], custom_q.strip())
                st.rerun()

        if st.session_state.custom_result:
            st.markdown(f'<div class="card-body" style="background:var(--surface-2);padding:1.2rem;border-radius:8px;margin-top:0.75rem;">{st.session_state.custom_result}</div>', unsafe_allow_html=True)
            with st.expander("Copy result", expanded=False):
                st.code(st.session_state.custom_result, language="markdown")
        st.markdown('</div>', unsafe_allow_html=True)

        studio_col3, studio_col4 = st.columns(2, gap="medium")

        # ── 5. Executive Presentation Deck Outline ──
        with studio_col3:
            st.markdown(
                """
                <div class="studio-card">
                    <div class="studio-title">📊 5-Slide Presentation Deck Outline</div>
                    <div class="studio-desc">Transform meeting decisions into a ready-to-present executive deck outline with slide titles & speaker notes.</div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Generate Slide Deck Outline", key="gen_pres_btn", use_container_width=True):
                with st.spinner("Structuring executive presentation..."):
                    st.session_state.presentation = generate_presentation_outline(
                        r['title'], r['summary'], r['key_decisions'], r['action_items']
                    )
                    r['presentation'] = st.session_state.presentation
                    save_session(r)
                    st.rerun()

            current_pres = r.get("presentation") or st.session_state.presentation
            if current_pres:
                with st.expander("View Slide Deck Outline", expanded=True):
                    st.markdown(current_pres)
                    st.code(current_pres, language="markdown")
                st.download_button("📥 Download Slide Outline (.md)", data=current_pres, file_name="presentation_deck_outline.md", mime="text/markdown", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── 6. Jira / Linear Backlog Issue Formatter ──
        with studio_col4:
            st.markdown(
                """
                <div class="studio-card">
                    <div class="studio-title">🎟️ Jira &amp; Linear Backlog Issues</div>
                    <div class="studio-desc">Convert informal action items into standardized engineering backlog cards with acceptance criteria.</div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Generate Jira / Linear Issues", key="gen_jira_btn", use_container_width=True):
                with st.spinner("Formatting action items into sprint tickets..."):
                    st.session_state.jira_tickets = generate_jira_tickets(r['action_items'])
                    r['jira_tickets'] = st.session_state.jira_tickets
                    save_session(r)
                    st.rerun()

            current_jira = r.get("jira_tickets") or st.session_state.jira_tickets
            if current_jira:
                with st.expander("View Sprint Tickets", expanded=True):
                    st.markdown(current_jira)
                    st.code(current_jira, language="markdown")
                st.download_button("📥 Download Tickets (.md)", data=current_jira, file_name="jira_tickets.md", mime="text/markdown", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4: CHAT (WITH PINNED TAKEAWAYS)
    # ════════════════════════════════════════════════════════════════════════════
    with tab_chat:
        st.markdown(
            """
            <div class="card" style="margin-bottom:1rem;">
                <div class="card-label">🧠 RAG Chat</div>
                <div class="card-accent-bar"></div>
                <div style="font-size:0.85rem;color:var(--text-2);">
                    Ask anything about this session. Answers are grounded in
                    the meeting transcript via ChromaDB vector search. You can pin key answers as permanent takeaways.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Pinned Takeaways Accordion (if any)
        if st.session_state.bookmarks:
            with st.expander(f"📌 Pinned Takeaways ({len(st.session_state.bookmarks)})", expanded=True):
                st.markdown('<div class="pinned-wrap">', unsafe_allow_html=True)
                for b_idx, bm in enumerate(st.session_state.bookmarks):
                    st.markdown(f"""
                    <div class="pinned-item">
                        <div class="pinned-q">Q: {html.escape(bm['q'])}</div>
                        <div class="pinned-a">{html.escape(bm['a'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                p_col1, p_col2 = st.columns([1, 4])
                with p_col1:
                    if st.button("Clear Pins", key="clear_pins_btn", type="secondary"):
                        st.session_state.bookmarks = []
                        st.rerun()

        # Suggested questions
        st.markdown('<div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-3);margin-bottom:0.5rem;">Suggested questions</div>', unsafe_allow_html=True)
        suggested_inquiries = [
            "What were the main decisions made?",
            "What action items were assigned?",
            "What questions remain unresolved?",
            "Give me a 3-bullet executive recap.",
        ]

        sq_cols = st.columns(len(suggested_inquiries))
        clicked_prompt = None
        for idx, sq in enumerate(suggested_inquiries):
            with sq_cols[idx]:
                if st.button(sq, key=f"sugg_q_{idx}", use_container_width=True, type="secondary"):
                    clicked_prompt = sq

        # Chat history
        if st.session_state.chat_history:
            chat_html = '<div class="chat-wrap">'
            for msg_idx, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    chat_html += f"""
                    <div class="chat-turn" style="text-align:right;">
                        <div class="chat-role role-you">You</div>
                        <div class="bubble bubble-you" style="float:right;">{html.escape(msg['content'])}</div>
                        <div style="clear:both;"></div>
                    </div>"""
                else:
                    chat_html += f"""
                    <div class="chat-turn">
                        <div class="chat-role role-ai">AI Assistant</div>
                        <div class="bubble bubble-ai">{html.escape(msg['content'])}</div>
                    </div>"""
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            # Option to bookmark the latest answer
            last_bot_msg = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
            last_user_msg = [m for m in st.session_state.chat_history if m["role"] == "user"]
            if last_bot_msg and last_user_msg:
                b_q = last_user_msg[-1]["content"]
                b_a = last_bot_msg[-1]["content"]
                if st.button("📌 Pin Latest Answer to Takeaways", key="pin_latest_btn", type="secondary"):
                    if not any(bm['q'] == b_q for bm in st.session_state.bookmarks):
                        st.session_state.bookmarks.append({"q": b_q, "a": b_a})
                        st.success("Pinned to takeaways!")
                        time.sleep(0.4)
                        st.rerun()

        else:
            st.markdown(
                """
                <div class="card" style="text-align:center;padding:2.5rem 1rem;">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">💬</div>
                    <div style="font-weight:600;color:var(--text);margin-bottom:0.25rem;">Start a conversation</div>
                    <div style="font-size:0.82rem;color:var(--text-2);">
                        Type a question below or click a suggested prompt above.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Input row
        chat_col1, chat_col2 = st.columns([5, 1], gap="small")
        with chat_col1:
            user_input = st.text_input(
                "Question",
                placeholder="Ask about the meeting…",
                label_visibility="collapsed",
                key="rag_chat_input_box",
            )
        with chat_col2:
            send_btn = st.button("Send", use_container_width=True)

        prompt_to_run = clicked_prompt if clicked_prompt else (user_input.strip() if send_btn else None)

        if prompt_to_run:
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_run})
            with st.spinner("Thinking…"):
                if r.get("rag_chain") is None:
                    r["rag_chain"] = build_rag_chain(r["transcript"])
                answer = ask_question(r["rag_chain"], prompt_to_run)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat", key="clear_chat_history_btn", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 5: HISTORY (FULL DEDICATED HISTORY HUB)
    # ════════════════════════════════════════════════════════════════════════════
    with tab_history:
        render_history_hub()

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 6: EXPORT & SHARE
    # ════════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown(
            """
            <div class="card">
                <div class="card-label">📥 Download Reports &amp; Assets</div>
                <div class="card-accent-bar"></div>
                <div style="font-size:0.85rem;color:var(--text-2);">
                    Export clean deliverables in plain text, structured Markdown, or JSON packages.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        md_report = generate_markdown_report(r)
        plain_full_report = (
            f"TITLE: {r['title']}\n"
            f"DATE:  {r.get('timestamp', 'N/A')}\n\n"
            f"{'='*60}\nTL;DR\n{'='*60}\n{r.get('tldr', 'N/A')}\n\n"
            f"{'='*60}\nSUMMARY\n{'='*60}\n{r['summary']}\n\n"
            f"{'='*60}\nTOPICS\n{'='*60}\n{r.get('topics', 'N/A')}\n\n"
            f"{'='*60}\nACTION ITEMS\n{'='*60}\n{r['action_items']}\n\n"
            f"{'='*60}\nKEY DECISIONS\n{'='*60}\n{r['key_decisions']}\n\n"
            f"{'='*60}\nOPEN QUESTIONS\n{'='*60}\n{r['open_questions']}\n\n"
            f"{'='*60}\nFULL TRANSCRIPT\n{'='*60}\n{r['transcript']}"
        )

        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4, gap="small")
        with dl_col1:
            st.download_button("📄 Transcript (.txt)", data=r["transcript"],
                file_name="transcript.txt", mime="text/plain", use_container_width=True)
        with dl_col2:
            st.download_button("📋 Summary (.txt)", data=r["summary"],
                file_name="summary.txt", mime="text/plain", use_container_width=True)
        with dl_col3:
            st.download_button("📊 Full Report (.txt)", data=plain_full_report,
                file_name="meeting_report.txt", mime="text/plain", use_container_width=True)
        with dl_col4:
            st.download_button("📑 Markdown (.md)", data=md_report,
                file_name="meeting_report.md", mime="text/markdown", use_container_width=True)

        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

        # Extra artifacts if generated
        extra_cols = st.columns(3, gap="small")
        with extra_cols[0]:
            if r.get("email_draft"):
                st.download_button("✉️ Email Draft (.txt)", data=r["email_draft"], file_name="follow_up_email.txt", mime="text/plain", use_container_width=True)
        with extra_cols[1]:
            if r.get("minutes"):
                st.download_button("📋 Minutes (.md)", data=r["minutes"], file_name="meeting_minutes.md", mime="text/markdown", use_container_width=True)
        with extra_cols[2]:
            if st.session_state.bookmarks:
                pins_text = "\n\n".join([f"Q: {bm['q']}\nA: {bm['a']}" for bm in st.session_state.bookmarks])
                st.download_button("📌 Pinned Takeaways (.txt)", data=pins_text, file_name="pinned_takeaways.txt", mime="text/plain", use_container_width=True)

        extra_cols2 = st.columns(4, gap="small")
        with extra_cols2[0]:
            if r.get("presentation"):
                st.download_button("📊 Slide Outline (.md)", data=r["presentation"], file_name="slides_outline.md", mime="text/markdown", use_container_width=True)
        with extra_cols2[1]:
            if r.get("jira_tickets"):
                st.download_button("🎟️ Jira Tickets (.md)", data=r["jira_tickets"], file_name="jira_tickets.md", mime="text/markdown", use_container_width=True)
        with extra_cols2[2]:
            if r.get("scorecard"):
                st.download_button("🎯 Scorecard (.txt)", data=r["scorecard"], file_name="meeting_scorecard.txt", mime="text/plain", use_container_width=True)
        with extra_cols2[3]:
            if r.get("chapters"):
                st.download_button("⏱️ Chapters (.txt)", data=r["chapters"], file_name="chapters.txt", mime="text/plain", use_container_width=True)

        st.markdown('<div class="card-label" style="margin-bottom:0.75rem;">📑 Formatted Report Preview</div>', unsafe_allow_html=True)
        st.markdown(md_report)

        with st.expander("Copy raw Markdown", expanded=False):
            st.code(md_report, language="markdown")

else:
    # ─── Empty State (Dual View: Quick Start & Saved Sessions) ───────────────────
    st.markdown(
        """
        <div class="empty-hero">
            <div class="empty-title">Drop in a meeting.<br>Get instant intelligence.</div>
            <div class="empty-sub">
                Paste a YouTube URL or upload an audio/video file in the sidebar,
                choose your language, and hit <strong>Analyse</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    empty_tab_quick, empty_tab_history = st.tabs([
        "✨ Quick Start & Features",
        f"📚 Saved Sessions ({len(st.session_state.history)})",
    ])

    with empty_tab_quick:
        st.markdown(
            """
            <div class="feat-grid">
                <div class="feat-item">
                    <div class="feat-emoji">🎙️</div>
                    <div class="feat-name">Speech-to-Text</div>
                    <div class="feat-detail">Whisper (English) and Sarvam AI (Hinglish) with chunked audio processing.</div>
                </div>
                <div class="feat-item">
                    <div class="feat-emoji">📋</div>
                    <div class="feat-name">Smart Summaries</div>
                    <div class="feat-detail">Map-reduce summarization via Mistral AI — works on hour-long recordings.</div>
                </div>
                <div class="feat-item">
                    <div class="feat-emoji">✅</div>
                    <div class="feat-name">Action Extraction</div>
                    <div class="feat-detail">Tasks, owners, deadlines, decisions and open questions — automatically.</div>
                </div>
                <div class="feat-item">
                    <div class="feat-emoji">⚡</div>
                    <div class="feat-name">Executive AI Studio</div>
                    <div class="feat-detail">Generate post-meeting emails, formal minutes, multi-language translations & custom queries.</div>
                </div>
                <div class="feat-item">
                    <div class="feat-emoji">💬</div>
                    <div class="feat-name">RAG Chat &amp; Pins</div>
                    <div class="feat-detail">Ask anything — grounded answers via ChromaDB vector search with takeaway pinning.</div>
                </div>
                <div class="feat-item">
                    <div class="feat-emoji">📚</div>
                    <div class="feat-name">Persistent History</div>
                    <div class="feat-detail">All past meeting analyses saved locally so you can restore and browse anytime.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with empty_tab_history:
        render_history_hub()
