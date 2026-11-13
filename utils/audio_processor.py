"""
utils/audio_processor.py
=========================
Handles all audio acquisition and preparation:
  - Download audio from YouTube (yt-dlp)
  - Convert any local audio/video file to 16kHz mono WAV (pydub)
  - Chunk a WAV file into N-minute pieces (pydub)

New in this version:
  - process_input_audio() now returns (chunks: list[str], full_wav_path: str)
    so callers have access to the complete WAV needed for whole-file diarization.
  - get_full_wav(source: str) -> str  — returns a single complete WAV without
    chunking, used when you only need diarization on the whole file.
"""

import os
import yt_dlp
from pydub import AudioSegment

# Directory where downloaded and converted files are stored.
DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# YouTube download
# ---------------------------------------------------------------------------

def download_youtube_audio(url: str) -> str:
    """
    Download the best audio track from a YouTube URL and save it as WAV.

    Returns
    -------
    str
        Absolute path to the downloaded WAV file.
    """
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp appends .wav after the postprocessor replaces the container.
        filename = (
            ydl.prepare_filename(info)
            .replace(".webm", ".wav")
            .replace(".m4a", ".wav")
        )

    return filename


# ---------------------------------------------------------------------------
# Local file conversion
# ---------------------------------------------------------------------------

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to a 16kHz mono WAV using pydub.

    Parameters
    ----------
    input_path : str
        Path to the source file (MP3, MP4, MKV, …).

    Returns
    -------
    str
        Path to the new WAV file (placed next to the original).
    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    # Normalise to mono 16kHz — required by both Whisper and WhisperX.
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """
    Split a WAV file into equal-length chunks.

    Parameters
    ----------
    wav_path : str
        Path to the source WAV file.
    chunk_minutes : int
        Length of each chunk in minutes (default 10).

    Returns
    -------
    list[str]
        Ordered list of paths to the chunk WAV files.
        Chunk index i starts at i * chunk_minutes * 60 seconds in the
        original audio — transcriber.py uses this index for offset tracking.
    """
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def get_full_wav(source: str) -> str:
    """
    Acquire audio from *source* (URL or local path) and return a single
    complete WAV file — no chunking.  Used when the caller needs the entire
    audio file for diarization before or instead of chunked transcription.

    Parameters
    ----------
    source : str
        YouTube URL or local audio/video file path.

    Returns
    -------
    str
        Path to the complete WAV file.
    """
    if source.startswith("http://") or source.startswith("https://"):
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV format if necessary.")
        wav_path = convert_to_wav(source)
    return wav_path


def process_input_audio(source: str) -> tuple:
    """
    Full pipeline: acquire → convert → chunk.

    Parameters
    ----------
    source : str
        YouTube URL or local audio/video file path.

    Returns
    -------
    chunks : list[str]
        Ordered list of WAV chunk file paths (10-minute pieces).
    full_wav_path : str
        Path to the single complete WAV file (before chunking).
        Pass this to WhisperX diarization so it can operate on the full audio
        rather than individual chunks.
    """
    # Step 1: obtain the full WAV (download or convert).
    full_wav_path = get_full_wav(source)

    # Step 2: split into chunks for Whisper transcription.
    print("Chunking audio …")
    chunks = chunk_audio(full_wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks, full_wav_path
