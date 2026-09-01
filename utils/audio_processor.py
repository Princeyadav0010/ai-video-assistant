import os
import uuid
import shutil
import yt_dlp
from pydub import AudioSegment


# =========================================================
# DOWNLOAD DIRECTORY
# =========================================================

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# =========================================================
# FFMPEG
# =========================================================

def _get_ffmpeg_path():
    """
    Find ffmpeg executable.
    Works locally and on Streamlit Cloud.
    """

    path = shutil.which("ffmpeg")

    if path:
        return path

    return "ffmpeg"


# =========================================================
# YOUTUBE AUDIO DOWNLOAD
# =========================================================

def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL.

    Designed to work both locally and on
    Streamlit Cloud.

    Does NOT use Chrome cookies.
    """

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "Please provide a valid YouTube URL."
        )

    file_id = uuid.uuid4().hex

    output_template = os.path.join(
        DOWNLOAD_DIR,
        f"{file_id}.%(ext)s"
    )

    options = {
        "format": "bestaudio/best",

        "outtmpl": output_template,

        "noplaylist": True,

        "quiet": True,

        "no_warnings": False,

        "ffmpeg_location": _get_ffmpeg_path(),

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    try:

        with yt_dlp.YoutubeDL(options) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            downloaded = ydl.prepare_filename(
                info
            )

            base = os.path.splitext(
                downloaded
            )[0]

            # After FFmpeg conversion
            possible_files = [
                base + ".wav",
                base + ".webm",
                base + ".m4a",
                base + ".opus",
                base + ".mp4",
            ]

            for path in possible_files:

                if os.path.exists(path):
                    return path

            # Final fallback
            for filename in os.listdir(
                DOWNLOAD_DIR
            ):

                if filename.startswith(file_id):

                    path = os.path.join(
                        DOWNLOAD_DIR,
                        filename
                    )

                    if os.path.isfile(path):
                        return path

    except Exception as e:

        raise RuntimeError(
            f"YouTube download failed: {e}"
        )

    raise FileNotFoundError(
        "Downloaded audio file could not be found."
    )


# =========================================================
# CONVERT AUDIO / VIDEO TO WAV
# =========================================================

def convert_to_wav(input_path: str) -> str:
    """
    Convert audio/video to WAV.

    Output:
    - Mono
    - 16 kHz
    """

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


# =========================================================
# SPLIT AUDIO INTO CHUNKS
# =========================================================

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:
    """
    Split WAV audio into chunks.

    Default chunk size:
    10 minutes
    """

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
        range(
            0,
            len(audio),
            chunk_ms
        )
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


# =========================================================
# MAIN INPUT PROCESSOR
# =========================================================

def process_input(source: str) -> list:
    """
    Process either:

    1. YouTube URL
    2. Local audio/video file

    Returns:
        List of WAV chunk paths.
    """

    source = source.strip()

    # -----------------------------------------------------
    # YOUTUBE URL
    # -----------------------------------------------------

    if source.startswith(
        ("http://", "https://")
    ):

        print(
            "Detected YouTube URL."
        )

        print(
            "Downloading audio..."
        )

        audio_path = download_youtube_audio(
            source
        )

    # -----------------------------------------------------
    # LOCAL AUDIO / VIDEO FILE
    # -----------------------------------------------------

    else:

        print(
            "Detected local audio/video file."
        )

        if not os.path.exists(source):

            raise FileNotFoundError(
                f"File not found: {source}"
            )

        audio_path = source

    # -----------------------------------------------------
    # CONVERT TO WAV
    # -----------------------------------------------------

    print(
        "Converting audio to WAV..."
    )

    wav_path = convert_to_wav(
        audio_path
    )

    # -----------------------------------------------------
    # CHUNK AUDIO
    # -----------------------------------------------------

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