from google import genai
import os
import time


def get_llm():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file."
        )

    return genai.Client(api_key=api_key)


def build_chain(system_prompt: str):

    client = get_llm()

    def invoke(transcript: str):

        prompt = f"""
{system_prompt}

Meeting transcript:

{transcript}
"""

        delays = [5, 10, 20]

        for attempt in range(4):

            try:

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                time.sleep(2)

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip()

            except Exception as e:

                error_text = str(e).lower()

                is_retryable = (
                    "429" in error_text
                    or "rate limit" in error_text
                    or "resource exhausted" in error_text
                    or "500" in error_text
                    or "503" in error_text
                )

                if not is_retryable:
                    raise

                if attempt >= 3:
                    raise RuntimeError(
                        "Gemini API request failed after multiple retries. "
                        "Please try again later."
                    ) from e

                time.sleep(delays[attempt])

    return invoke


def get_output_language(language: str) -> str:

    if language.lower() == "hinglish":

        return """
Write the answer in natural Hinglish.

IMPORTANT:
- Use ONLY English/Roman letters.
- Do NOT use Hindi/Devanagari script.
- Keep common English and technical words in English.
"""

    return """
Write the answer in clear English.
"""


def extract_action_items(
    transcript: str,
    language: str = "english"
) -> str:

    output_language = get_output_language(language)

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all action items. "
        "For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. "
        "If none found say 'No action items found.'\n\n"
        + output_language
    )

    return chain(transcript)


def extract_key_decisions(
    transcript: str,
    language: str = "english"
) -> str:

    output_language = get_output_language(language)

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all key decisions made. "
        "Format as a numbered list. "
        "If none found say 'No key decisions found.'\n\n"
        + output_language
    )

    return chain(transcript)


def extract_questions(
    transcript: str,
    language: str = "english"
) -> str:

    output_language = get_output_language(language)

    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. "
        "Format as a numbered list. "
        "If none found say 'No open questions found.'\n\n"
        + output_language
    )

    return chain(transcript)