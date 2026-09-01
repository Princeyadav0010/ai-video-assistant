import os
import uuid
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:

    file_id = uuid.uuid4().hex

    output_path = os.path.join(
        DOWNLOAD_DIR,
        f"{file_id}.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",

        "outtmpl": output_path,

        "noplaylist": True,

        "quiet": True,

        "no_warnings": True,

        # Let yt-dlp use its current supported YouTube
        # client selection instead of forcing Chrome cookies.
        "extractor_args": {
            "youtube": {
                "player_client": [
                    "default",
                    "web_safari",
                    "ios"
                ]
            }
        },

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        # Retry temporary network failures
        "retries": 3,
        "fragment_retries": 3,

        # Avoid unnecessary playlist processing
        "ignoreerrors": False,
    }

    try:

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            downloaded_file = ydl.prepare_filename(info)

            wav_file = (
                os.path.splitext(downloaded_file)[0]
                + ".wav"
            )

            if os.path.exists(wav_file):
                return wav_file

            # Some postprocessor/container combinations
            # may produce a slightly different path.
            base_name = os.path.splitext(
                downloaded_file
            )[0]

            possible_files = [
                base_name + ".wav",
                base_name + ".webm",
                base_name + ".m4a",
                base_name + ".mp3",
            ]

            for file_path in possible_files:

                if os.path.exists(file_path):

                    if file_path.endswith(".wav"):
                        return file_path

                    return convert_to_wav(file_path)

            raise FileNotFoundError(
                "YouTube audio was downloaded, "
                "but the WAV file could not be located."
            )

    except Exception as e:

        raise RuntimeError(
            f"YouTube download failed: {str(e)}"
        ) from e


def convert_to_wav(input_path: str) -> str:

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"Audio file not found: {input_path}"
        )

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(
        input_path
    )

    # Whisper works best with mono 16 kHz audio.
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

    if not os.path.exists(wav_path):

        raise FileNotFoundError(
            f"WAV file not found: {wav_path}"
        )

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


def process_input(source: str) -> list:

    source = source.strip()

    if not source:

        raise ValueError(
            "No input source provided."
        )

    # ------------------------------------------------
    # YouTube URL
    # ------------------------------------------------

    if (
        source.startswith("http://")
        or source.startswith("https://")
    ):

        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        audio_path = download_youtube_audio(
            source
        )

    # ------------------------------------------------
    # Local Audio / Video
    # ------------------------------------------------

    else:

        print(
            "Detected local audio/video file."
        )

        if not os.path.exists(source):

            raise FileNotFoundError(
                f"File not found: {source}"
            )

        audio_path = source

    # ------------------------------------------------
    # Convert
    # ------------------------------------------------

    print(
        "Converting audio to WAV..."
    )

    wav_path = convert_to_wav(
        audio_path
    )

    # ------------------------------------------------
    # Chunk
    # ------------------------------------------------

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