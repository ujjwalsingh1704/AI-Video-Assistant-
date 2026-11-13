"""
core/transcriber.py
====================
Transcription engine using faster-whisper for local ASR and WhisperX for
speaker diarization. Supports English/auto-detect and Hinglish (via Sarvam API).

Outputs per transcribe_all() call:
  - full_transcript : str  — plain concatenated text (backward-compat for
                            summarize.py, extractor.py, rag_engine.py)
  - segments        : list[dict] — structured segments with speaker + timestamps
    Each dict: {"text": str, "speaker": str, "start": float, "end": float}
"""

import os
import logging
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
HF_TOKEN = os.getenv("HF_TOKEN")            # Needed for WhisperX diarization
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

# Sarvam sync API rejects audio longer than 30 s — slice into 25 s pieces.
SARVAM_PIECE_SECONDS = 25

# Chunk length used in audio_processor (10 min = 600 s).
CHUNK_SECONDS = 10 * 60

# ---------------------------------------------------------------------------
# Lazy model loader (faster-whisper)
# ---------------------------------------------------------------------------

_whisper_model: WhisperModel | None = None


def _load_whisper_model() -> WhisperModel:
    """Load the faster-whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading faster-whisper model: %s …", WHISPER_MODEL_SIZE)
        print(f"Loading faster-whisper model: {WHISPER_MODEL_SIZE} …")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        print("faster-whisper model loaded.")
    return _whisper_model


# ---------------------------------------------------------------------------
# Sarvam helpers (Hinglish path — kept from original)
# ---------------------------------------------------------------------------

def _send_to_sarvam(piece_path: str) -> str:
    """POST a single ≤30 s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )
    if not response.ok:
        logger.error("Sarvam returned %s: %s", response.status_code, response.text)
        response.raise_for_status()
    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30 s audio. Split the chunk into 25-second
    pieces, send each separately, and join the resulting transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    full_text = ""

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} …")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# ---------------------------------------------------------------------------
# faster-whisper transcription helpers
# ---------------------------------------------------------------------------

def _transcribe_chunk_faster_whisper(chunk_path: str) -> list[dict]:
    """
    Transcribe a single WAV chunk with faster-whisper.

    Returns a list of raw segment dicts:
        {"text": str, "start": float, "end": float}
    VAD filtering is enabled to skip silent regions.
    """
    model = _load_whisper_model()
    segments, _info = model.transcribe(
        chunk_path,
        language=None,          # auto-detect
        vad_filter=True,
        word_timestamps=False,
    )
    result = []
    for seg in segments:
        result.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end,
        })
    return result


# ---------------------------------------------------------------------------
# WhisperX diarization helper
# ---------------------------------------------------------------------------

def _diarize_and_assign(chunk_path: str, raw_segments: list[dict]) -> list[dict]:
    """
    Run WhisperX diarization on *chunk_path* and assign speaker labels to
    *raw_segments* by matching each segment's midpoint to the diarization
    timeline.

    Falls back to speaker="SPEAKER_00" for all segments if:
      - HF_TOKEN is not set
      - whisperx is not installed
      - diarization fails for any reason
    """
    if not HF_TOKEN:
        logger.warning(
            "HF_TOKEN not set — skipping diarization; all segments → SPEAKER_00"
        )
        return [dict(seg, speaker="SPEAKER_00") for seg in raw_segments]

    try:
        import whisperx  # optional dependency

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN, device="cpu"
        )
        # DiarizationPipeline returns a DataFrame with columns:
        #   segment (Segment namedtuple), label, speaker
        diarize_df = diarize_model(chunk_path)

        # Build a list of (start, end, speaker) from the diarization output
        diarize_spans: list[tuple[float, float, str]] = []
        for _, row in diarize_df.iterrows():
            seg = row["segment"]
            diarize_spans.append((seg.start, seg.end, row["speaker"]))

        def _find_speaker(start: float, end: float) -> str:
            """Return the speaker whose span contains the segment midpoint."""
            mid = (start + end) / 2.0
            for d_start, d_end, spk in diarize_spans:
                if d_start <= mid <= d_end:
                    return spk
            # Fall back to the overlapping span with the most overlap
            best_spk = "SPEAKER_00"
            best_overlap = 0.0
            for d_start, d_end, spk in diarize_spans:
                overlap = max(0.0, min(end, d_end) - max(start, d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = spk
            return best_spk

        return [dict(seg, speaker=_find_speaker(seg["start"], seg["end"]))
                for seg in raw_segments]

    except Exception as exc:
        logger.warning(
            "Diarization failed (%s) — falling back to SPEAKER_00 for all segments.",
            exc,
        )
        return [dict(seg, speaker="SPEAKER_00") for seg in raw_segments]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Backward-compatible single-chunk transcription (plain string output).
    Routes to Sarvam for Hinglish, faster-whisper otherwise.
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    # Join all segment texts for a plain string
    segments = _transcribe_chunk_faster_whisper(chunk_path)
    return " ".join(seg["text"] for seg in segments).strip()


def transcribe_all(
    chunks: list,
    language: str = "auto",
) -> tuple[str, list[dict]]:
    """
    Transcribe all audio chunks and return both a plain transcript string
    (for backward compatibility) and a structured segment list (for new features).

    Parameters
    ----------
    chunks : list[str]
        Paths to WAV chunk files produced by audio_processor.chunk_audio().
    language : str
        "auto" or "english" → faster-whisper with auto language detection.
        "hinglish"          → Sarvam API for ASR/translation + diarization attempt.

    Returns
    -------
    full_transcript : str
        All transcribed text joined into a single string.
    segments : list[dict]
        Each element is:
        {
            "text"    : str,
            "speaker" : str,   # e.g. "SPEAKER_00", "SPEAKER_01"
            "start"   : float, # seconds from the beginning of the full audio
            "end"     : float,
        }
    """
    is_hinglish = language.lower() == "hinglish"
    engine_name = "Sarvam AI" if is_hinglish else "faster-whisper"
    print(f"Using {engine_name} for transcription.")

    all_segments: list[dict] = []

    for chunk_idx, chunk_path in enumerate(chunks):
        # Each chunk represents CHUNK_SECONDS of audio; compute its time offset.
        time_offset = chunk_idx * CHUNK_SECONDS
        print(f"Transcribing chunk {chunk_idx + 1}/{len(chunks)} "
              f"(offset {time_offset}s) …")

        if is_hinglish:
            # --- Hinglish path: Sarvam for text + optional diarization ---
            # Attempt diarization on the chunk audio before sending to Sarvam.
            # We generate raw segments with placeholder timing for diarization.
            audio = AudioSegment.from_wav(chunk_path)
            chunk_duration = len(audio) / 1000.0  # ms → seconds

            # Try diarization; Sarvam doesn't return segment-level timestamps,
            # so we produce a single segment spanning the whole chunk.
            sarvam_text = transcribe_chunk_sarvam(chunk_path)

            raw_segments = [{
                "text": sarvam_text,
                "start": 0.0,
                "end": chunk_duration,
            }]
            diarized = _diarize_and_assign(chunk_path, raw_segments)

        else:
            # --- English / auto path: faster-whisper ---
            raw_segments = _transcribe_chunk_faster_whisper(chunk_path)
            diarized = _diarize_and_assign(chunk_path, raw_segments)

        # Apply chunk time offset so timestamps are relative to full audio.
        for seg in diarized:
            seg["start"] = round(seg["start"] + time_offset, 3)
            seg["end"] = round(seg["end"] + time_offset, 3)

        all_segments.extend(diarized)

    # Build the plain full-transcript string (backward compat).
    full_transcript = " ".join(seg["text"] for seg in all_segments).strip()

    print("Transcription complete.")
    return full_transcript, all_segments
