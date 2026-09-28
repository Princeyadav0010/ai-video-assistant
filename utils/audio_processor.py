import os
import uuid
import shutil
import subprocess

import yt_dlp
from pydub import AudioSegment


# =========================================================
# DOWNLOAD DIRECTORY
# =========================================================

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# =========================================================
# BGUTIL PO TOKEN PROVIDER
# =========================================================

BGUTIL_VERSION = "1.3.1"
BGUTIL_DIR = "/tmp/bgutil-ytdlp-pot-provider"
BGUTIL_SERVER_DIR = os.path.join(BGUTIL_DIR, "server")
BGUTIL_SCRIPT = os.path.join(
    BGUTIL_SERVER_DIR,
    "build",
    "generate_once.js"
)


def _get_ffmpeg_path():
    path = shutil.which("ffmpeg")

    if path:
        return path

    return "ffmpeg"


def _get_node_path():
    path = shutil.which("node")

    if path:
        return path

    raise RuntimeError(
        "Node.js was not found. "
        "Please make sure nodejs is present in packages.txt."
    )


def _setup_bgutil_provider():
    """
    Download and build the bgutil PO-token generation script
    once per Streamlit runtime.
    """

    node_path = _get_node_path()

    # Already built
    if os.path.exists(BGUTIL_SCRIPT):
        return BGUTIL_SCRIPT

    print("Setting up bgutil PO-token provider...")

    # Remove incomplete previous setup
    if os.path.exists(BGUTIL_DIR):
        shutil.rmtree(BGUTIL_DIR, ignore_errors=True)

    # Clone provider
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            BGUTIL_VERSION,
            "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git",
            BGUTIL_DIR,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Install Node dependencies
    subprocess.run(
        [
            "npm",
            "ci",
        ],
        cwd=BGUTIL_SERVER_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    # Compile TypeScript
    subprocess.run(
        [
            "npx",
            "tsc",
        ],
        cwd=BGUTIL_SERVER_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    if not os.path.exists(BGUTIL_SCRIPT):
        raise RuntimeError(
            "bgutil PO-token generation script was not built."
        )

    print(
        f"bgutil provider ready. Node: {node_path}"
    )

    return BGUTIL_SCRIPT


# =========================================================
# YOUTUBE AUDIO DOWNLOAD
# =========================================================

def download_youtube_audio(url: str) -> str:
    """
    Download audio from YouTube.

    Uses:
    - mweb YouTube client
    - bgutil PO-token generation script

    Designed for local and Streamlit Cloud.
    """

    url = url.strip()

    if not url.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            "Please provide a valid YouTube URL."
        )

    script_path = _setup_bgutil_provider()
    node_path = _get_node_path()

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

        "remote_components": "ejs:npm",

        # Use Node for JS runtime
        "js_runtimes": {
          "node": {
          "path": node_path
         }
        },

        # YouTube client + PO-token provider
        "extractor_args": {
            "youtube": {
                "player_client": ["mweb"]
            },
            "youtubepot-bgutilscript": {
                "script_path": script_path
            },
        },

        # Convert audio to WAV
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    print("Downloading YouTube audio using mweb + bgutil...")

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

            possible_files = [
                base + ".wav",
                base + ".webm",
                base + ".m4a",
                base + ".opus",
                base + ".mp4",
            ]

            for path in possible_files:

                if os.path.exists(path):
                    print(
                        "YouTube download successful."
                    )
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
        ) from e

    raise FileNotFoundError(
        "Downloaded audio file could not be found."
    )


# =========================================================
# CONVERT AUDIO / VIDEO TO WAV
# =========================================================

def convert_to_wav(input_path: str) -> str:
    """
    Convert audio/video to:
    - Mono
    - 16 kHz WAV
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
# SPLIT AUDIO
# =========================================================

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:
    """
    Split WAV into 10-minute chunks by default.
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
    Process:
    1. YouTube URL
    2. Local audio/video file

    Returns:
        List of WAV chunk paths.
    """

    source = source.strip()

    # -----------------------------------------------------
    # YOUTUBE
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
    # LOCAL FILE
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
    # CONVERT
    # -----------------------------------------------------

    print(
        "Converting audio to WAV..."
    )

    wav_path = convert_to_wav(
        audio_path
    )

    # -----------------------------------------------------
    # CHUNK
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