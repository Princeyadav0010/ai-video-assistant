import os
import time
import whisper

from google import genai


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

GEMINI_MODEL = "gemini-3.5-flash-lite"


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file."
        )

    return genai.Client(api_key=api_key)


def convert_to_hinglish(text: str, video_language: str) -> str:
    client = get_gemini_client()

    if video_language.lower() == "hindi":

        prompt = f"""
The following is a Hindi transcript.

Convert it from Devanagari Hindi into Roman Hindi (Hinglish).

IMPORTANT RULES:
- Do NOT translate Hindi into English.
- Preserve the exact Hindi meaning.
- Hindi words MUST be written using English/Roman letters.
- Do NOT use Devanagari/Hindi script.
- Keep common English and technical words in English.
- Do NOT summarize.
- Do NOT remove information.
- Do NOT add new information.
- Return ONLY the Roman Hindi transcript.

Example:

Hindi:
आज हम क्रिकेट के बारे में बात करेंगे।

Hinglish:
Aaj hum cricket ke baare mein baat karenge.

Transcript:
{text}
"""

    else:

        prompt = f"""
The following is an English transcript.

Translate the transcript into natural Hindi, but write Hindi
using ONLY English/Roman letters (Hinglish).

IMPORTANT RULES:
- Translate the meaning into Hindi.
- Do NOT simply repeat the English sentence.
- Do NOT use Devanagari/Hindi script.
- Hindi words MUST be written in Roman letters.
- Keep common English and technical words in English.
- Preserve the complete meaning.
- Do NOT summarize.
- Do NOT remove information.
- Do NOT add new information.
- Return ONLY the Roman Hindi transcript.

Example:

English:
Today we will discuss artificial intelligence.

Hinglish:
Aaj hum artificial intelligence ke baare mein discuss karenge.

Transcript:
{text}
"""

    delays = [5, 10, 20]

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as e:

            print("GEMINI ERROR:", repr(e))

            error_text = str(e).lower()

            is_retryable = (
                "429" in error_text
                or "rate limit" in error_text
                or "resource exhausted" in error_text
                or "500" in error_text
                or "503" in error_text
                or "service unavailable" in error_text
            )

            if not is_retryable:
                raise

            if attempt >= 3:
                raise RuntimeError(
                    "Gemini API request failed after multiple retries."
                ) from e

            time.sleep(delays[attempt])


def transcribe_chunk_whisper(
    chunk_path: str,
    video_language: str = "english",
    transcript_language: str = "english"
) -> str:

    model = load_model()

    video_language = video_language.lower()
    transcript_language = transcript_language.lower()

    # =========================================================
    # HINDI VIDEO → ENGLISH TRANSCRIPT
    # =========================================================

    if (
        video_language == "hindi"
        and transcript_language == "english"
    ):

        result = model.transcribe(
            chunk_path,
            language="hi",
            task="translate"
        )

        return result["text"].strip()

    # =========================================================
    # HINDI VIDEO → HINGLISH TRANSCRIPT
    # =========================================================

    if (
        video_language == "hindi"
        and transcript_language == "hinglish"
    ):

        result = model.transcribe(
            chunk_path,
            language="hi",
            task="transcribe"
        )

        hindi_text = result["text"].strip()

        return convert_to_hinglish(
            hindi_text,
            video_language
        )

    # =========================================================
    # ENGLISH VIDEO → ENGLISH TRANSCRIPT
    # =========================================================

    if (
        video_language == "english"
        and transcript_language == "english"
    ):

        result = model.transcribe(
            chunk_path,
            language="en",
            task="transcribe"
        )

        return result["text"].strip()

    # =========================================================
    # ENGLISH VIDEO → HINGLISH TRANSCRIPT
    # =========================================================

    if (
        video_language == "english"
        and transcript_language == "hinglish"
    ):

        result = model.transcribe(
            chunk_path,
            language="en",
            task="transcribe"
        )

        english_text = result["text"].strip()

        return convert_to_hinglish(
            english_text,
            video_language
        )

    raise ValueError(
        f"Unsupported combination: "
        f"video_language={video_language}, "
        f"transcript_language={transcript_language}"
    )


def transcribe_chunk(
    chunk_path: str,
    video_language: str = "english",
    transcript_language: str = "english"
) -> str:

    return transcribe_chunk_whisper(
        chunk_path,
        video_language,
        transcript_language
    )


def transcribe_all(
    chunks: list,
    video_language: str = "english",
    transcript_language: str = "english"
) -> str:

    full_transcript = ""

    print(f"Video language: {video_language}")
    print(f"Transcript language: {transcript_language}")

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            video_language=video_language,
            transcript_language=transcript_language
        )

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()