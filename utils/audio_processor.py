import os
import uuid
import shutil
import subprocess
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _get_ffmpeg_path():
    """
    Find ffmpeg executable.
    Works locally and on Streamlit Cloud.
    """
    path = shutil.which("ffmpeg")

    if path:
        return path

    return "ffmpeg"


def _download_with_options(url: str, options: dict) -> str:
    """
    Download YouTube audio using the supplied yt-dlp options.
    Returns downloaded WAV path.
    """

    file_id = uuid.uuid4().hex

    output_template = os.path.join(
        DOWNLOAD_DIR,
        f"{file_id}.%(ext)s"
    )

    options = options.copy()

    options["outtmpl"] = output_template
    options["ffmpeg_location"] = _get_ffmpeg_path()

    with yt_dlp.YoutubeDL(options) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        downloaded = ydl.prepare_filename(info)

        base = os.path.splitext(downloaded)[0]

        possible_files = [
            base + ".wav",
            base + ".webm",
            base + ".m4a",
            base + ".mp4",
            base + ".opus",
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                return file_path

        # Search directory as final fallback
        for filename in os.listdir(DOWNLOAD_DIR):

            path = os.path.join(
                DOWNLOAD_DIR,
                filename
            )

            if filename.startswith(file_id):
                return path

    raise FileNotFoundError(
        "Downloaded audio file could not be found."
    )


def download_youtube_audio(url: str) -> str:

    url = url.strip()

    if not url.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            "Please provide a valid YouTube URL."
        )

    common = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,

        # Let yt-dlp solve YouTube's current JS challenges
        "remote_components": "ejs:npm",

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    # --------------------------------------------------
    # Attempt 1
    # web_embedded does not require a PO token.
    # --------------------------------------------------

    options_1 = {
        **common,
        "format": "bestaudio/best",
        "extractor_args": {
            "youtube": {
                "player_client": [
                    "web_embedded"
                ]
            }
        },
    }

    try:

        return _download_with_options(
            url,
            options_1
        )

    except Exception as first_error:

        print(
            "YouTube embedded client failed:",
            first_error
        )

    # --------------------------------------------------
    # Attempt 2
    # Android VR client fallback.
    # --------------------------------------------------

    options_2 = {
        **common,
        "format": "bestaudio/best",
        "extractor_args": {
            "youtube": {
                "player_client": [
                    "android_vr"
                ]
            }
        },
    }

    try:

        return _download_with_options(
            url,
            options_2
        )

    except Exception as second_error:

        print(
            "YouTube Android VR client failed:",
            second_error
        )

    # --------------------------------------------------
    # Attempt 3
    # Normal yt-dlp extraction.
    # No Chrome cookies.
    # --------------------------------------------------

    options_3 = {
        **common,
        "format": "bestaudio/best",
    }

    try:

        return _download_with_options(
            url,
            options_3
        )

    except Exception as third_error:

        raise RuntimeError(
            "YouTube download failed. "
            "YouTube blocked the request from the "
            "deployment server. "
            f"Details: {third_error}"
        )


def convert_to_wav(input_path: str) -> str:

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(
        input_path
    )

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav"
    )

    return output_path


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    audio = AudioSegment.from_wav(
        wav_path
    )

    chunk_ms = (
        chunk_minutes
        * 60
        * 1000
    )

    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:

    source = source.strip()

    # --------------------------------------------------
    # YouTube URL
    # --------------------------------------------------

    if source.startswith(
        ("http://", "https://")
    ):

        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        audio_path = download_youtube_audio(
            source
        )

    # --------------------------------------------------
    # Local file
    # --------------------------------------------------

    else:

        print(
            "Detected local audio/video file."
        )

        if not os.path.exists(source):

            raise FileNotFoundError(
                f"File not found: {source}"
            )

        audio_path = source

    print(
        "Converting audio to WAV..."
    )

    wav_path = convert_to_wav(
        audio_path
    )

    print(
        "Chunking audio..."
    )

    chunks = chunk_audio(
        wav_path
    )

    print(
        f"Audio ready - "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks