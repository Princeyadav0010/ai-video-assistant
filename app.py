import streamlit as st
import time
import os
from io import BytesIO

from dotenv import load_dotenv
from pypdf import PdfReader

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI RETRY HELPER
# ─────────────────────────────────────────────────────────────────────────────

from google import genai

GEMINI_MODEL = "gemini-3.5-flash-lite"


def _is_retryable_gemini_error(error: Exception) -> bool:
    error_text = str(error).lower()

    return (
        "429" in error_text
        or "rate limit" in error_text
        or "rate_limited" in error_text
        or "resource exhausted" in error_text
        or "500" in error_text
        or "503" in error_text
        or "service unavailable" in error_text
    )


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file "
            "or Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


def _invoke_gemini_with_retry(
    prompt: str,
    max_retries: int = 3
) -> str:
    client = _get_gemini_client()
    delays = [5, 10, 20]

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            time.sleep(2)
            return response.text.strip()

        except Exception as e:
            print("GEMINI ERROR:", repr(e))

            if not _is_retryable_gemini_error(e):
                raise

            if attempt >= max_retries:
                raise RuntimeError(
                    "Gemini API request failed after multiple retries. "
                    "Please try again later."
                ) from e

            time.sleep(delays[attempt])


# ─────────────────────────────────────────────────────────────────────────────
# HINGLISH CONVERSION
# ─────────────────────────────────────────────────────────────────────────────


def convert_output_to_hinglish(text: str) -> str:
    if not text or not text.strip():
        return text

    prompt = f"""
Convert the following meeting-analysis output into natural Hinglish.

IMPORTANT RULES:
- Write Hindi using ONLY English/Roman letters.
- NEVER use Devanagari/Hindi script.
- If the source is English, translate its meaning into Hindi written in Roman letters.
- Keep common English and technical terms in English.
- Preserve every fact, name, number, bullet, heading and list item.
- Do not summarize, shorten, remove, or add information.
- Return ONLY the converted text.

Text:
{text}
"""

    return _invoke_gemini_with_retry(prompt)


def output_in_selected_language(
    text: str,
    transcript_language: str
) -> str:
    if transcript_language.lower() == "hinglish":
        return convert_output_to_hinglish(text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');


/* ── Root Variables ── */

:root {
    --bg: #101a18;
    --surface: #172522;
    --surface-2: #20332e;
    --border: #315149;
    --accent: #14b8a6;
    --accent-glow: #5eead4;
    --accent-2: #f0a35b;
    --text: #e9f4ee;
    --text-muted: #98b2a7;
    --success: #34d399;
    --warning: #f3b35f;
    --danger: #fb7185;
}


/* ── Global Reset ── */

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}


.stApp {
    background: var(--bg) !important;
}


/* ── Edge-to-edge app canvas ── */

header[data-testid="stHeader"],
footer {
    display: none !important;
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] .block-container {
    padding: 0 !important;
    max-width: none !important;
}


/* ── Animated grid background ── */

.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    background-image:
        linear-gradient(
            rgba(94, 234, 212, 0.045) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(94, 234, 212, 0.045) 1px,
            transparent 1px
        );

    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}


/* ── Single workspace layout ── */

[data-testid="stSidebar"] {
    display: none !important;
}


[data-testid="stSidebar"] * {
    color: var(--text) !important;
}


/* ── Headings ── */

h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}


/* ── Hero Title ── */

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;

    background: linear-gradient(
        135deg,
        #f4fbf7 0%,
        var(--accent-glow) 50%,
        var(--accent-2) 100%
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
}


.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    text-align: center;
}


/* ── Cards ── */

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}


.card:hover {
    border-color: var(--accent);
}


/* ── Input workspace ── */

.input-panel {
    background: linear-gradient(135deg, rgba(23,37,34,0.98), rgba(32,51,46,0.92));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.35rem 0.35rem;
    margin: 1.5rem 0 1.25rem;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
}

.input-panel .panel-kicker {
    color: var(--accent-2);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.panel-kicker {
    color: #c4b5fd;
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.25rem, 2.2vw, 1.8rem);
    font-weight: 800;
    letter-spacing: 0.02em;
    margin: 1.5rem 0 0.8rem;
    text-align: center;
    text-transform: uppercase;
    animation: workspace-heading 1s ease both;
}

[data-testid="stHorizontalBlock"] {
    align-items: end !important;
    gap: 2.4rem !important;
}

.stTextInput > div > div > input {
    min-height: 3.35rem !important;
    padding: 0.8rem 1rem !important;
    font-size: 1rem !important;
}

[data-testid="stFileUploader"] {
    min-height: 3.35rem;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    color: #dcebe5 !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}

[data-testid="stRadio"] {
    padding-top: 0.5rem !important;
    padding-bottom: 0.8rem !important;
}

[data-testid="stAlert"] {
    width: min(620px, calc(100% - 2rem)) !important;
    margin: 1.4rem auto !important;
    border-radius: 14px !important;
    animation: alert-pop-in 420ms cubic-bezier(0.22, 1, 0.36, 1) both;
    box-shadow: 0 14px 36px rgba(248,113,113,0.2) !important;
}

.input-type-row {
    display: flex;
    justify-content: center;
}

.source-caption {
    color: var(--text-muted);
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    margin: 1rem 0 0.25rem;
    text-align: center;
    text-transform: uppercase;
}

@keyframes workspace-heading {
    from { opacity: 0; transform: translateY(-12px) scale(0.96); letter-spacing: 0.12em; }
    to { opacity: 1; transform: translateY(0) scale(1); letter-spacing: 0.02em; }
}

@keyframes alert-pop-in {
    from { opacity: 0; transform: translateY(-14px) scale(0.94); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes analysis-dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}

.input-panel .panel-title {
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.pipeline-panel {
    background: rgba(23,37,34,0.96);
    border: 1px solid rgba(20,184,166,0.3);
    border-radius: 14px;
    padding: 1rem 1.1rem 0.75rem;
    margin: 1rem 0 1.25rem;
}

.pipeline-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.7rem;
}

.pipeline-title {
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
}

.pipeline-subtitle {
    color: var(--text-muted);
    font-size: 0.72rem;
}

.info-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1.25rem 0 1.5rem;
}

.info-item {
    background: rgba(23,37,34,0.8);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}

.info-item strong {
    display: block;
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    margin-bottom: 0.25rem;
}

.info-item span {
    color: var(--text-muted);
    font-size: 0.7rem;
    line-height: 1.5;
}

.empty-state {
    min-height: 70vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
}

.empty-state-title {
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 5rem);
    font-weight: 800;
    line-height: 1;
    margin: 0.9rem 0 1rem;
    text-shadow: 0 0 24px rgba(94,234,212,0.14);
    animation: home-title-reveal 0.9s ease both;
}

.empty-state-copy {
    max-width: 680px;
    color: var(--text-muted);
    font-size: clamp(1rem, 1.5vw, 1.25rem);
    line-height: 1.7;
    margin-bottom: 1.75rem;
}

.empty-state .info-strip {
    width: min(1100px, 100%);
    margin-top: 2.5rem;
}

.assistant-scene {
    width: 170px;
    height: 170px;
    margin: 0 auto 1.25rem;
    display: grid;
    place-items: center;
    perspective: 700px;
}

.assistant-orbit,
.assistant-orbit::before,
.assistant-orbit::after {
    position: absolute;
    width: 150px;
    height: 52px;
    border: 1px solid rgba(94,234,212,0.48);
    border-radius: 50%;
    content: '';
}

.assistant-orbit {
    transform: rotateX(68deg) rotateZ(-18deg);
    animation: orbit-spin 8s linear infinite;
}

.assistant-orbit::before {
    transform: rotateY(62deg) rotateZ(34deg);
    border-color: rgba(240,163,91,0.55);
    animation: orbit-spin-reverse 6s linear infinite;
}

.assistant-orbit::after {
    transform: rotateY(62deg) rotateZ(-34deg);
    border-color: rgba(94,234,212,0.3);
}

.assistant-core {
    width: 76px;
    height: 76px;
    border-radius: 50%;
    position: relative;
    background: radial-gradient(circle at 32% 25%, #d8fff5 0 5%, #5eead4 14%, #14b8a6 42%, #0b5d58 76%, #082f31 100%);
    box-shadow: 0 0 24px rgba(20,184,166,0.7), 0 0 70px rgba(20,184,166,0.24);
    animation: assistant-float 3.6s ease-in-out infinite;
}

.assistant-core::before,
.assistant-core::after {
    position: absolute;
    content: '';
    border-radius: 50%;
}

.assistant-core::before {
    inset: 12px;
    border: 1px solid rgba(255,255,255,0.68);
    box-shadow: inset 0 0 12px rgba(255,255,255,0.35);
}

.assistant-core::after {
    width: 10px;
    height: 10px;
    top: 22px;
    left: 24px;
    background: #ffffff;
    box-shadow: 22px 8px 0 -1px #ffffff, 9px 23px 0 -2px rgba(255,255,255,0.85);
}

.analysis-loader {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    width: min(620px, calc(100% - 2rem));
    margin: 1.25rem auto;
    padding: 1rem 1.25rem;
    border: 1px solid rgba(94,234,212,0.32);
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(20,184,166,0.14), rgba(124,58,237,0.18));
    box-shadow: 0 0 30px rgba(20,184,166,0.14);
}

.analysis-loader .assistant-scene {
    width: 64px;
    height: 64px;
    margin: 0;
    flex: 0 0 64px;
    transform: scale(0.5);
    transform-origin: center;
}

.analysis-loader-status {
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
}

.analysis-loader-status::after {
    content: '...';
    display: inline-block;
    width: 1.25rem;
    color: var(--accent-2);
    animation: analysis-dots 1.2s steps(4, end) infinite;
}

.home-screen {
    min-height: 100vh;
    width: 100vw;
    margin: 0;
    display: grid;
    place-items: center;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(167,139,250,0.22);
    border-radius: 0;
    background:
        radial-gradient(circle at 50% 35%, rgba(124,58,237,0.28), transparent 34%),
        linear-gradient(145deg, #17142d 0%, #111a25 58%, #101a18 100%);
    box-shadow: inset 0 0 80px rgba(124,58,237,0.12), 0 24px 80px rgba(0,0,0,0.22);
}

.home-screen::before,
.home-screen::after {
    content: '';
    position: absolute;
    border: 1px solid rgba(167,139,250,0.16);
    border-radius: 50%;
    pointer-events: none;
}

.home-screen::before {
    width: 620px;
    height: 620px;
    animation: home-ring 14s linear infinite;
}

.home-screen::after {
    width: 820px;
    height: 820px;
    border-color: rgba(45,212,191,0.1);
    animation: home-ring-reverse 18s linear infinite;
}

.home-content {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 4rem 1.5rem;
    width: 100vw;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.home-eyebrow {
    color: #c4b5fd;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    animation: home-reveal 0.9s ease both;
}

.home-title {
    color: #f7f5ff;
    font-family: 'Syne', sans-serif;
    width: 100vw;
    font-size: clamp(3.8rem, 10.8vw, 10rem);
    font-weight: 800;
    letter-spacing: 0.015em;
    line-height: 0.94;
    margin: 0;
    text-shadow: 0 0 34px rgba(167,139,250,0.18);
    animation: home-title-reveal 1.1s cubic-bezier(0.22, 1, 0.36, 1) 0.12s both;
}

.ai-word {
    display: inline-flex;
    align-items: baseline;
    gap: 0.03em;
    color: #f7f5ff;
}

.moving-i {
    display: inline-block;
    position: relative;
    color: #a78bfa;
}

.moving-i::before {
    content: '';
    position: absolute;
    width: 0.17em;
    height: 0.17em;
    top: -0.18em;
    left: 50%;
    border-radius: 50%;
    background: #2dd4bf;
    box-shadow: 0 0 14px #2dd4bf, 0 0 28px rgba(45,212,191,0.8);
    animation: dot-dance 2.2s ease-in-out infinite;
}

.home-subtitle {
    max-width: 720px;
    color: #b7c4c7;
    font-size: clamp(1rem, 1.8vw, 1.25rem);
    line-height: 1.75;
    margin: 1.3rem auto 2rem;
    animation: home-reveal 0.9s ease 0.38s both;
}

.route-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.5rem 2rem 2.75rem;
}

.route-label {
    color: var(--text-muted);
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.home-route-link {
    color: #c4b5fd !important;
    border: 1px solid rgba(196,181,253,0.35);
    border-radius: 999px;
    padding: 0.55rem 1rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-decoration: none !important;
    transition: all 0.2s ease;
}

.home-route-link:hover {
    color: #ffffff !important;
    border-color: #a78bfa;
    background: rgba(124,58,237,0.2);
    box-shadow: 0 0 20px rgba(124,58,237,0.25);
}

.home-orb {
    margin: 0 auto 1.8rem;
    transform: scale(1.35);
}

.stButton > button[kind="primary"],
[data-testid="stButton"] button[kind="primary"] {
    position: relative !important;
    overflow: hidden !important;
    min-width: 290px !important;
    min-height: 64px !important;
    border: 1px solid rgba(255,255,255,0.36) !important;
    border-radius: 999px !important;
    background: linear-gradient(110deg, #5b21b6 0%, #7c3aed 42%, #14b8a6 100%) !important;
    background-size: 180% 180% !important;
    box-shadow: 0 0 34px rgba(217,70,239,0.5), 0 0 12px rgba(20,184,166,0.32), inset 0 1px 0 rgba(255,255,255,0.42) !important;
    color: #ffffff !important;
    font-size: 0.8rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.13em !important;
    padding: 0.7rem 1.7rem !important;
    transition: transform 180ms ease, filter 180ms ease !important;
    animation: cta-gradient 2.8s ease infinite, cta-pulse 2.2s ease-in-out infinite;
}

.stButton > button[kind="primary"]::before,
[data-testid="stButton"] button[kind="primary"]::before {
    content: '';
    position: absolute;
    inset: 0;
    width: 42%;
    transform: translateX(-180%) skewX(-18deg);
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.72), transparent);
    pointer-events: none;
    animation: cta-shine 3.6s ease-in-out infinite;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(110deg, #7c3aed 0%, #db2777 48%, #06b6d4 100%) !important;
    background-size: 180% 180% !important;
    filter: saturate(1.35) brightness(1.12) !important;
    animation: cta-gradient 1.6s ease infinite;
}

[data-testid="stButton"]:has(button[kind="primary"]) {
    position: fixed !important;
    left: 50% !important;
    bottom: 11vh !important;
    z-index: 20 !important;
    display: flex !important;
    justify-content: center !important;
    margin: 0 !important;
    transform: translateX(-50%) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-4px) scale(1.04) !important;
    box-shadow: 0 0 42px rgba(219,39,119,0.6), 0 0 72px rgba(6,182,212,0.34) !important;
}

.stButton > button[kind="primary"]:active {
    transform: translateY(-1px) scale(0.98) !important;
}

.stHorizontalBlock {
    gap: 1.5rem !important;
}

div[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stRadio"] {
    transform: none;
}

div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stRadio"] {
    transform: none;
}

[data-testid="stRadio"]:has(input[id*="video_language"]) {
    transform: translateX(1cm);
}

[data-testid="stRadio"]:has(input[id*="transcript_language"]) {
    transform: translateX(1cm);
}

.stTextInput,
.stFileUploader,
.stRadio {
    margin-bottom: 0.9rem !important;
}

.card,
.pipeline-panel,
.chat-container {
    margin-top: 1.25rem;
    margin-bottom: 1.5rem;
}

.chat-input-spacer {
    height: 1.25rem;
}

.summary-card {
    width: min(920px, 100%);
    margin-left: auto;
    margin-right: auto;
}

.summary-card .card-title {
    text-align: center;
}

@keyframes dot-dance {
    0%, 100% { transform: translate(-50%, 0) scale(1); }
    50% { transform: translate(-50%, -0.18em) scale(1.25); }
}

@keyframes home-ring {
    from { transform: rotate(0deg) scale(1); }
    to { transform: rotate(360deg) scale(1.04); }
}

@keyframes home-ring-reverse {
    from { transform: rotate(360deg) scale(1.02); }
    to { transform: rotate(0deg) scale(1); }
}

@keyframes orbit-spin {
    from { transform: rotateX(68deg) rotateZ(-18deg); }
    to { transform: rotateX(68deg) rotateZ(342deg); }
}

@keyframes orbit-spin-reverse {
    from { transform: rotateY(62deg) rotateZ(34deg); }
    to { transform: rotateY(62deg) rotateZ(-326deg); }
}

@keyframes assistant-float {
    0%, 100% { transform: translateY(0) scale(1); }
    50% { transform: translateY(-8px) scale(1.04); }
}

@keyframes cta-gradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes cta-pulse {
    0%, 100% { box-shadow: 0 0 28px rgba(217,70,239,0.32), inset 0 1px 0 rgba(255,255,255,0.32); }
    50% { box-shadow: 0 0 42px rgba(20,184,166,0.48), 0 0 68px rgba(217,70,239,0.18), inset 0 1px 0 rgba(255,255,255,0.42); }
}

@keyframes cta-shine {
    0%, 35% { transform: translateX(-180%) skewX(-18deg); }
    65%, 100% { transform: translateX(420%) skewX(-18deg); }
}

@keyframes home-reveal {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes home-title-reveal {
    from { opacity: 0; transform: translateY(28px) scaleX(1.06) scaleY(0.94); filter: blur(8px); }
    to { opacity: 1; transform: translateY(0) scaleX(1.15) scaleY(1); filter: blur(0); }
}

@media (prefers-reduced-motion: reduce) {
    .assistant-orbit,
    .assistant-orbit::before,
    .assistant-core,
    .home-screen::before,
    .home-screen::after,
    .moving-i::before,
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"]::before,
    [data-testid="stAlert"],
    .home-eyebrow,
    .home-title,
    .home-subtitle,
    .panel-kicker {
        animation: none;
    }
}


@media (max-width: 700px) {
    .home-screen {
        width: calc(100% + 2rem);
        margin-left: -1rem;
        margin-right: -1rem;
    }

    .home-content {
        padding: 3rem 1rem;
    }

    .home-title {
        width: 100%;
        font-size: clamp(3rem, 15vw, 5rem);
        transform: none;
    }

    .stButton > button[kind="primary"] {
        min-width: 240px !important;
        min-height: 58px !important;
    }

    div[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stRadio"] {
        transform: none;
    }

    div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stRadio"] {
        transform: none;
    }

    [data-testid="stRadio"]:has(input[id*="video_language"]) {
        transform: none;
    }

    [data-testid="stRadio"]:has(input[id*="transcript_language"]) {
        transform: none;
    }

    [data-testid="stButton"]:has(button[kind="primary"]) {
        bottom: 8vh !important;
    }
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;

    background: linear-gradient(
        180deg,
        var(--accent),
        var(--accent-2)
    );
}


.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;

    display: flex;
    align-items: center;
    gap: 0.5rem;
}


.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}


/* ── Accent Badge ── */

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}


.badge-purple {
    background: rgba(15,118,110,0.1);
    color: var(--accent-glow);
    border: 1px solid rgba(15,118,110,0.25);
}


.badge-cyan {
    background: rgba(194,106,22,0.1);
    color: var(--accent-2);
    border: 1px solid rgba(194,106,22,0.25);
}


.badge-green {
    background: rgba(22,128,93,0.1);
    color: var(--success);
    border: 1px solid rgba(22,128,93,0.25);
}


/* ── Input & Buttons ── */

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}


.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(15,118,110,0.16) !important;
}


.stButton > button {
    background: linear-gradient(
        135deg,
        var(--accent),
        #0d9488
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 8px !important;

    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;

    padding: 0.6rem 1.5rem !important;

    transition: all 0.2s !important;
    text-transform: uppercase !important;
}


.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(15,118,110,0.24) !important;
}


.input-panel .stButton > button {
    min-height: 2.65rem !important;
    margin-top: 1.62rem !important;
    border-radius: 9px !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    box-shadow: 0 8px 20px rgba(15,118,110,0.16) !important;
}


/* ── Secondary button ── */

.stButton > button[kind="secondary"] {
    min-height: 3.25rem !important;
    margin-top: 1.15rem !important;
    padding: 0.75rem 1.6rem !important;
    background: linear-gradient(110deg, rgba(20,184,166,0.24), rgba(124,58,237,0.3)) !important;
    border: 1px solid rgba(94,234,212,0.42) !important;
    border-radius: 999px !important;
    color: #e9f4ee !important;
    font-size: 0.9rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.08em !important;
    box-shadow: 0 8px 24px rgba(20,184,166,0.12) !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: #5eead4 !important;
    box-shadow: 0 0 28px rgba(20,184,166,0.28) !important;
    transform: translateY(-2px) !important;
}


/* ── Progress / Status ── */

.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;

    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}


.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}


.dot-active {
    background: var(--accent-glow);
    box-shadow: 0 0 8px var(--accent-glow);
    animation: pulse 1.5s infinite;
}


.dot-done {
    background: var(--success);
}


.dot-pending {
    background: var(--border);
}


@keyframes pulse {

    0%, 100% {
        opacity: 1;
    }

    50% {
        opacity: 0.4;
    }

}


/* ── Chat ── */

.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}


.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}


.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}


.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}


.user-label {
    color: var(--accent-glow);
}


.bot-label {
    color: var(--accent-2);
}


.user-bubble {
    background: rgba(15,118,110,0.09);
    border: 1px solid rgba(15,118,110,0.2);
    align-self: flex-end;
}


.bot-bubble {
    background: rgba(194,106,22,0.08);
    border: 1px solid rgba(194,106,22,0.18);
    align-self: flex-start;
}


/* ── Divider ── */

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}


/* ── Transcript box ── */

.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}


/* ── Streamlit elements ── */

.stProgress > div > div > div {
    background: var(--accent) !important;
}


.stSpinner > div {
    border-top-color: var(--accent) !important;
}


[data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
}


label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}


/* ── Scrollbar ── */

::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}


::-webkit-scrollbar-track {
    background: var(--bg);
}


::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}


::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "home_open": False,
}.items():

    if key not in st.session_state:
        st.session_state[key] = default


    current_page = st.query_params.get("page", "home")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def step_status(
    steps: dict,
    key: str
) -> str:

    s = steps.get(
        key,
        "pending"
    )

    if s == "active":
        return "dot-active"

    if s == "done":
        return "dot-done"

    return "dot-pending"


def render_step_bar(
    label: str,
    key: str,
    icon: str
):

    css = step_status(
        st.session_state.pipeline_steps,
        key
    )

    st.markdown(
        f"""
<div class="status-bar">
<div class="status-dot {css}"></div>
<span>{icon} {label}</span>
</div>
""",
        unsafe_allow_html=True
    )


PIPELINE_STEPS = [
    ("audio", "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title", "🏷️", "Title Generation"),
    ("summary", "📋", "Summarisation"),
    ("extract", "🔍", "Extraction"),
    ("rag", "🧠", "RAG Engine"),
]


def render_pipeline_monitor(container, title, subtitle):

    with container.container():

        st.markdown(
            f'''
<div class="pipeline-panel">
<div class="pipeline-heading">
<div class="pipeline-title">{title}</div>
<div class="pipeline-subtitle">{subtitle}</div>
</div>
''',
            unsafe_allow_html=True
        )

        for step, icon, label in PIPELINE_STEPS:

            render_step_bar(
                label,
                step,
                icon
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


if current_page == "home":

    st.markdown(
        '''
<section class="home-screen">
<div class="home-content">
<div class="home-eyebrow">Welcome to the AI world</div>
<div class="assistant-scene home-orb"><div class="assistant-orbit"></div><div class="assistant-core"></div></div>
<h1 class="home-title">Meet your <span class="ai-word">A<span class="moving-i">i</span></span><br>Video Assistant</h1>
<p class="home-subtitle">Turn long videos and meetings into clear, searchable intelligence with one thoughtful AI workspace.</p>
''',
        unsafe_allow_html=True
    )

    if st.button(
        "Click here to analyse  →",
        type="primary",
        key="open_assistant",
        help="Open AI Video Assistant"
    ):

        st.query_params["page"] = "analyze"
        st.rerun()

    st.markdown(
        '''
</div>
</section>
''',
        unsafe_allow_html=True
    )

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

if current_page != "home":

    route_label_col, route_home_col = st.columns([5, 1], gap="small")

    with route_label_col:

        if st.button(
            "← Home",
            key="back_home",
            type="secondary"
        ):

            st.query_params["page"] = "home"
            st.rerun()

st.markdown(
    '<div class="hero-title">AI Video Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="panel-kicker">Workspace input</div>',
    unsafe_allow_html=True
)

input_type_left, input_type_center, input_type_right = st.columns(
    [1, 2, 1],
    gap="small"
)

with input_type_center:

    input_type = st.radio(
        "Input Type",
        [
            "YouTube / Audio / Video",
            "PDF"
        ],
        horizontal=True
    )

st.markdown(
    '<div class="source-caption">Add a YouTube URL or local file path below</div>'
    if input_type == "YouTube / Audio / Video"
    else '<div class="source-caption">Upload your PDF document below</div>',
    unsafe_allow_html=True
)

source = ""
uploaded_pdf = None

if input_type == "YouTube / Audio / Video":

    source_outer_left, source_col, source_outer_right = st.columns(
        [1, 10, 1],
        gap="small"
    )

    with source_col:

        source = st.text_input(
            "YouTube URL or File Path",
            placeholder="https://youtube.com/watch?v=..."
        )

    controls_outer_left, controls_col, controls_outer_right = st.columns(
        [2, 8, 2],
        gap="small"
    )

    with controls_col:

        video_col, transcript_col = st.columns(
            [1, 1],
            gap="small"
        )

    with video_col:

        video_language = st.radio(
            "Video Language",
            [
                "english",
                "hindi"
            ],
            key="video_language",
            horizontal=True,
            format_func=str.title
        )

    with transcript_col:

        transcript_language = st.radio(
            "Transcript",
            [
                "english",
                "hinglish"
            ],
            key="transcript_language",
            horizontal=True,
            format_func=str.title
        )

    action_outer_left, action_col, action_outer_right = st.columns(
        [3, 2, 3],
        gap="small"
    )

    with action_col:

        run_btn = st.button(
            "⚡ Analyse",
            use_container_width=True
        )

else:

    pdf_outer_left, pdf_upload_col, pdf_outer_right = st.columns(
        [1.5, 9, 1.5],
        gap="small"
    )

    with pdf_upload_col:

        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload a text-based PDF to analyse it."
        )

    pdf_settings_left, pdf_settings_col, pdf_settings_right = st.columns(
        [3, 6, 3],
        gap="small"
    )

    with pdf_settings_col:

        transcript_col, action_col = st.columns(
            [2.4, 1.6],
            gap="small"
        )

    with transcript_col:

        transcript_language = st.radio(
            "Transcript",
            [
                "english",
                "hinglish"
            ],
            horizontal=True,
            format_func=str.title
        )

    with action_col:

        run_btn = st.button(
            "⚡ Analyse",
            use_container_width=True
        )

    video_language = "pdf"

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:

    if input_type == "PDF" and uploaded_pdf is None:

        st.error(
            "Please upload a PDF file."
        )

    elif input_type != "PDF" and not source.strip():

        st.error(
            "Please enter a YouTube URL or file path."
        )


    else:

        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.processing = True


        progress_placeholder = st.empty()
        pipeline_placeholder = st.empty()

        render_pipeline_monitor(
            pipeline_placeholder,
            "Live analysis pipeline",
            "Starting..."
        )


        def update_step(
            key,
            state
        ):

            st.session_state.pipeline_steps[key] = state

            current_label = next(
                (
                    label
                    for step, _, label in PIPELINE_STEPS
                    if step == key
                ),
                "Processing"
            )

            render_pipeline_monitor(
                pipeline_placeholder,
                "Live analysis pipeline",
                f"{current_label} in progress"
                if state == "active"
                else f"{current_label} complete"
            )


        try:

            with progress_placeholder.container():

                st.markdown(
                    '''
    <div class="analysis-loader">
    <div class="assistant-scene"><div class="assistant-orbit"></div><div class="assistant-core"></div></div>
    <div class="analysis-loader-status">AI is analysing</div>
    </div>
    ''',
                    unsafe_allow_html=True
                )

                st.info(
                    "⚙️ Pipeline running — live stage status is shown below."
                )


            # ─────────────────────────────────────────────────────────────
            # AUDIO / PDF
            # ─────────────────────────────────────────────────────────────

            update_step(
                "audio",
                "active"
            )


            if input_type == "PDF":

                pdf_bytes = uploaded_pdf.getvalue()

                reader = PdfReader(
                    BytesIO(pdf_bytes)
                )


                pages = []


                for page in reader.pages:

                    page_text = (
                        page.extract_text()
                        or ""
                    )

                    if page_text.strip():

                        pages.append(
                            page_text.strip()
                        )


                transcript = (
                    "\n\n".join(pages)
                    .strip()
                )


                if not transcript:

                    raise ValueError(
                        "No selectable text found in this PDF. "
                        "Scanned/image-only PDFs need OCR support."
                    )


                update_step(
                    "audio",
                    "done"
                )


                # ─────────────────────────────────────────────────────────
                # TRANSCRIPT
                # ─────────────────────────────────────────────────────────

                update_step(
                    "transcript",
                    "active"
                )


                if transcript_language == "hinglish":

                    from core.transcriber import convert_to_hinglish

                    transcript = convert_to_hinglish(
                        transcript,
                        "hindi"
                    )


                update_step(
                    "transcript",
                    "done"
                )


            else:

                chunks = process_input(
                    source
                )


                update_step(
                    "audio",
                    "done"
                )


                update_step(
                    "transcript",
                    "active"
                )


                transcript = transcribe_all(
                    chunks,
                    video_language=video_language,
                    transcript_language=transcript_language
                )


                update_step(
                    "transcript",
                    "done"
                )


            # ─────────────────────────────────────────────────────────────
            # TITLE
            # ─────────────────────────────────────────────────────────────

            update_step(
                "title",
                "active"
            )


            title = generate_title(
                transcript
            )


            title = output_in_selected_language(
                title,
                transcript_language
            )


            update_step(
                "title",
                "done"
            )


            # ─────────────────────────────────────────────────────────────
            # SUMMARY
            # ─────────────────────────────────────────────────────────────

            update_step(
                "summary",
                "active"
            )


            summary = summarize(
                transcript
            )


            summary = output_in_selected_language(
                summary,
                transcript_language
            )


            update_step(
                "summary",
                "done"
            )


            # ─────────────────────────────────────────────────────────────
            # EXTRACTION
            # ─────────────────────────────────────────────────────────────

            update_step(
                "extract",
                "active"
            )


            action_items = extract_action_items(
                transcript
            )


            decisions = extract_key_decisions(
                transcript
            )


            questions = extract_questions(
                transcript
            )


            action_items = output_in_selected_language(
                action_items,
                transcript_language
            )


            decisions = output_in_selected_language(
                decisions,
                transcript_language
            )


            questions = output_in_selected_language(
                questions,
                transcript_language
            )


            update_step(
                "extract",
                "done"
            )


            # ─────────────────────────────────────────────────────────────
            # RAG
            # ─────────────────────────────────────────────────────────────

            update_step(
                "rag",
                "active"
            )


            rag_chain = build_rag_chain(
                transcript
            )


            update_step(
                "rag",
                "done"
            )


            # ─────────────────────────────────────────────────────────────
            # SAVE RESULTS
            # ─────────────────────────────────────────────────────────────

            st.session_state.result = {

                "title": title,

                "transcript": transcript,

                "summary": summary,

                "action_items": action_items,

                "key_decisions": decisions,

                "open_questions": questions,

                "rag_chain": rag_chain,
            }


            st.session_state.pipeline_done = True
            st.session_state.processing = False


            progress_placeholder.success(
                "✅ Analysis complete!"
            )


            time.sleep(0.5)

            progress_placeholder.empty()

            st.rerun()


        except Exception as e:

            st.session_state.processing = False

            for k in [
                "audio",
                "transcript",
                "title",
                "summary",
                "extract",
                "rag"
            ]:

                if (
                    st.session_state.pipeline_steps.get(k)
                    == "active"
                ):

                    st.session_state.pipeline_steps[k] = "pending"


            progress_placeholder.error(
                f"❌ Error: {e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.result:

    r = st.session_state.result


    # ─────────────────────────────────────────────────────────────────────────
    # TITLE BANNER
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown(
        f"""
<div class="card">
<div class="card-title">📌 Session Title</div>
<div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
{r['title']}
</div>
</div>
""",
        unsafe_allow_html=True
    )


    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY + TRANSCRIPT
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown(
        f"""
<div class="card summary-card">
<div class="card-title">📋 Summary</div>
<div class="card-content">{r['summary']}</div>
</div>
""",
        unsafe_allow_html=True
    )


    transcript_left, transcript_center, transcript_right = st.columns(
        [1, 4, 1],
        gap="medium"
    )


    with transcript_center:

        with st.expander(
            "📝 Full Transcript",
            expanded=False
        ):

            st.markdown(
                f"""
<div class="transcript-box">
{r["transcript"]}
</div>
""",
                unsafe_allow_html=True
            )


    # ─────────────────────────────────────────────────────────────────────────
    # ACTION ITEMS / DECISIONS / QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────

    c1, c2, c3 = st.columns(
        3,
        gap="medium"
    )


    with c1:

        st.markdown(
            f"""
<div class="card">
<div class="card-title">✅ Action Items</div>
<div class="card-content">{r['action_items']}</div>
</div>
""",
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
<div class="card">
<div class="card-title">🔑 Key Decisions</div>
<div class="card-content">{r['key_decisions']}</div>
</div>
""",
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
<div class="card">
<div class="card-title">❓ Open Questions</div>
<div class="card-content">{r['open_questions']}</div>
</div>
""",
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ─────────────────────────────────────────────────────────────────────────
    # RAG CHAT
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>',
        unsafe_allow_html=True
    )


    # ─────────────────────────────────────────────────────────────────────────
    # CHAT HISTORY
    # ─────────────────────────────────────────────────────────────────────────

    if st.session_state.chat_history:

        chat_html = '<div class="chat-container">'


        for msg in st.session_state.chat_history:

            if msg["role"] == "user":

                chat_html += f"""
<div class="chat-msg" style="align-items:flex-end">
<span class="chat-label user-label">You</span>
<div class="chat-bubble user-bubble">{msg['content']}</div>
</div>
"""


            else:

                chat_html += f"""
<div class="chat-msg" style="align-items:flex-start">
<span class="chat-label bot-label">🤖 Assistant</span>
<div class="chat-bubble bot-bubble">{msg['content']}</div>
</div>
"""


        chat_html += "</div>"


        st.markdown(
            chat_html,
            unsafe_allow_html=True
        )


    else:

        # Empty chat state
        # Kept as single-line HTML strings to avoid
        # Streamlit treating indented HTML as code.

        st.markdown(
            '<div class="card" style="text-align:center;padding:2rem;">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div style="font-size:2rem;margin-bottom:0.5rem;">💬</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div style="color:var(--text-muted);font-size:0.85rem;">Ask anything about your meeting transcript</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ─────────────────────────────────────────────────────────────────────────
    # CHAT INPUT
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown(
        '<div class="chat-input-spacer"></div>',
        unsafe_allow_html=True
    )

    with st.form(
        "chat_form",
        clear_on_submit=True
    ):

        chat_col1, chat_col2 = st.columns(
            [5, 1],
            gap="small"
        )


        with chat_col1:

            user_input = st.text_input(
                "Your question",
                placeholder="What were the main decisions made?",
                label_visibility="collapsed"
            )


        with chat_col2:

            send_btn = st.form_submit_button(
                "Send →",
                use_container_width=True
            )


    # ─────────────────────────────────────────────────────────────────────────
    # SEND CHAT
    # ─────────────────────────────────────────────────────────────────────────

    if send_btn and not user_input.strip():

        st.error(
            "Please type a question before sending."
        )

    if send_btn and user_input.strip():

        with st.spinner("Thinking…"):

            question = user_input.strip()


            if transcript_language == "hinglish":

                question = (
                    question
                    + "\n\nAnswer in natural Hinglish "
                      "using ONLY English/Roman letters. "
                      "Do NOT use Devanagari/Hindi script."
                )


            answer = ask_question(
                r["rag_chain"],
                question
            )


            answer = output_in_selected_language(
                answer,
                transcript_language
            )


        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input.strip()
            }
        )


        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        st.rerun()


    # ─────────────────────────────────────────────────────────────────────────
    # CLEAR CHAT
    # ─────────────────────────────────────────────────────────────────────────

    if st.session_state.chat_history:

        if st.button(
            "🗑️ Clear Chat",
            type="secondary"
        ):

            st.session_state.chat_history = []

            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.markdown(
        '''
<div class="empty-state">
<div class="assistant-scene"><div class="assistant-orbit"></div><div class="assistant-core"></div></div>
<div class="empty-state-title">Ready to Analyse</div>
<div class="empty-state-copy">Choose YouTube/Audio/Video or PDF, select your transcript language, and hit <strong>Analyse</strong> to get started.</div>
<div style="display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;">
<span class="badge badge-purple">Transcription</span>
<span class="badge badge-cyan">Summarisation</span>
<span class="badge badge-green">RAG Chat</span>
</div>
<div class="info-strip">
<div class="info-item"><strong>1. Add your source</strong><span>Use a YouTube link, local media file, or PDF document.</span></div>
<div class="info-item"><strong>2. Get clear insights</strong><span>Transcripts, summaries, decisions, and action items in one place.</span></div>
<div class="info-item"><strong>3. Ask follow-up questions</strong><span>Chat with your meeting content through grounded RAG answers.</span></div>
</div>
</div>
''',
        unsafe_allow_html=True
    )