import whisper
import os
from langchain_mistralai import ChatMistralAI

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def convert_to_hinglish(text: str, video_language: str) -> str:

    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0
    )

    # Hindi video → Roman Hindi
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

    # English video → Hindi meaning in Roman Hindi
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

    response = llm.invoke(prompt)

    return response.content.strip()


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