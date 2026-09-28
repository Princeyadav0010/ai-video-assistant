from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import time


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to Streamlit Secrets "
            "or your local .env file."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# GEMINI API CALL WITH RETRY
# =========================================================

def _invoke_with_retry(prompt: str, max_retries: int = 3) -> str:
    """
    Call Gemini with retry for temporary API errors.
    """

    client = get_llm()

    delays = [5, 10, 20]

    for attempt in range(max_retries + 1):

        try:
            response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

            time.sleep(2)

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as e:

            # Print exact error in Streamlit Cloud logs
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

            # Non-temporary error
            if not is_retryable:
                raise

            # Retries exhausted
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini API error after retries: {repr(e)}"
                ) from e

            time.sleep(delays[attempt])


# =========================================================
# SPLIT TRANSCRIPT
# =========================================================

def split_transcript(transcript: str) -> list:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


# =========================================================
# SUMMARIZE
# =========================================================

def summarize(transcript: str) -> str:

    chunks = split_transcript(transcript)

    chunk_summaries = []

    for chunk in chunks:

        prompt = f"""
Summarize this portion of a meeting transcript concisely.

Transcript:
{chunk}
"""

        summary = _invoke_with_retry(prompt)

        chunk_summaries.append(summary)

    combined = "\n\n".join(chunk_summaries)

    final_prompt = f"""
You are an expert meeting summarizer.

Combine the following partial summaries into one final
professional meeting summary.

Use clear bullet points.
Remove repetition.
Keep the important information, decisions, tasks and context.

Partial summaries:

{combined}
"""

    return _invoke_with_retry(final_prompt)


# =========================================================
# GENERATE TITLE
# =========================================================

def generate_title(transcipt: str) -> str:

    prompt = f"""
Based on the meeting transcript below, generate a short
professional meeting title.

Rules:
- Maximum 8 words
- Only return the title
- Do not add quotation marks
- Do not add explanations

Transcript:

{transcipt[:2000]}
"""

    return _invoke_with_retry(prompt)