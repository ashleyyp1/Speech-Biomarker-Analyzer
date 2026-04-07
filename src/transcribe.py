"""
Transcribe .wav audio with Whisper. Long audio is processed in chunks.
Designed for noisy / inconsistent inputs: preprocessing, per-chunk error isolation,
and structured metadata when ASR partially fails.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import whisper

from .utils import AUDIO_INPUTS_DIR, TRANSCRIPTS_DIR, ensure_dirs, safe_stem

logger = logging.getLogger(__name__)

MAX_CHUNK_SEC = 300.0
WHISPER_SAMPLE_RATE = 16000
# Skip ASR on extremely short clips (still return duration)
MIN_ASR_SECONDS = 0.25


def preprocess_audio_for_asr(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Normalize and lightly trim leading/trailing silence for more stable Whisper behavior.
    Does not aggressively denoise (Whisper is trained on varied conditions).
    """
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        return y
    y_trim, _ = librosa.effects.trim(y, top_db=32)
    # If trim removes almost everything, keep original (e.g. very noisy bed)
    if y_trim.size >= max(int(sr * MIN_ASR_SECONDS), int(0.05 * y.size)):
        y = y_trim
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = (y / peak * 0.98).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def _load_audio_mono(
    path: str | Path, sr: int = WHISPER_SAMPLE_RATE
) -> tuple[np.ndarray, float, str | None]:
    """
    Load mono audio. On total failure returns empty array and duration 0 (caller still runs).
    """
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        y = np.asarray(y, dtype=np.float32)
        duration = float(len(y) / sr)
        return y, duration, None
    except Exception as e:
        logger.warning("librosa.load failed for %s: %s — retrying with soundfile", path, e)
    try:
        import soundfile as sf

        raw, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(raw, "ndim", 1) > 1:
            raw = np.mean(raw, axis=1)
        y = librosa.resample(np.asarray(raw, dtype=np.float32), orig_sr=file_sr, target_sr=sr)
        duration = float(len(y) / sr)
        return y, duration, None
    except Exception as e:
        logger.error("Could not load audio %s: %s", path, e)
    return np.zeros(0, dtype=np.float32), 0.0, "audio_load_failed"


def _transcribe_array(
    model: whisper.Whisper,
    audio: np.ndarray,
    *,
    offset_start: float = 0.0,
) -> tuple[str, list[dict[str, Any]]]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return "", []
    result = model.transcribe(
        audio,
        fp16=False,
        verbose=False,
        temperature=0.0,
    )
    text = (result.get("text") or "").strip()
    segments_raw = result.get("segments") or []
    segments: list[dict[str, Any]] = []
    for seg in segments_raw:
        segments.append(
            {
                "id": seg.get("id"),
                "start": float(seg.get("start", 0.0)) + offset_start,
                "end": float(seg.get("end", 0.0)) + offset_start,
                "text": (seg.get("text") or "").strip(),
            }
        )
    return text, segments


def _transcribe_array_safe(
    model: whisper.Whisper,
    audio: np.ndarray,
    *,
    offset_start: float = 0.0,
) -> tuple[str, list[dict[str, Any]], str | None]:
    try:
        t, segs = _transcribe_array(model, audio, offset_start=offset_start)
        return t, segs, None
    except Exception as e:
        logger.exception("Whisper failed on chunk offset %.2fs", offset_start)
        return "", [], f"{type(e).__name__}: {e}"


def transcribe_wav(
    wav_path: str | Path,
    *,
    model_name: str = "base",
    model: whisper.Whisper | None = None,
    save: bool = True,
    chunk_seconds: float = MAX_CHUNK_SEC,
) -> dict[str, Any]:
    """
    Load a .wav file, transcribe with Whisper, optionally save JSON to transcripts/.

    Returns:
        transcript, segments, duration, and transcription_meta (issues, chunk errors).
    """
    ensure_dirs()
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio not found: {wav_path}")

    try:
        loaded = model or whisper.load_model(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load Whisper model {model_name!r} (disk space, network, or torch issue). {e}"
        ) from e
    y_raw, duration, load_err = _load_audio_mono(wav_path)
    y = preprocess_audio_for_asr(y_raw, WHISPER_SAMPLE_RATE)

    meta: dict[str, Any] = {
        "source": str(wav_path.resolve()),
        "model": model_name,
        "preprocess": "peak_norm_trim",
        "chunk_errors": [],
        "skipped_asr": False,
    }
    if load_err:
        meta["audio_load_error"] = load_err

    all_text_parts: list[str] = []
    all_segments: list[dict[str, Any]] = []

    if duration < MIN_ASR_SECONDS:
        meta["skipped_asr"] = True
        meta["reason"] = "duration_below_min"
        transcript = ""
    elif duration <= chunk_seconds:
        text, segments, err = _transcribe_array_safe(loaded, y)
        if err:
            meta["chunk_errors"].append({"chunk": 0, "error": err})
        all_text_parts.append(text)
        all_segments.extend(segments)
        transcript = " ".join(t for t in all_text_parts if t).strip()
    else:
        n_chunks = int(math.ceil(duration / chunk_seconds))
        samples_per_chunk = int(chunk_seconds * WHISPER_SAMPLE_RATE)
        for i in range(n_chunks):
            start = i * samples_per_chunk
            end = min(start + samples_per_chunk, len(y))
            chunk = y[start:end]
            offset = start / WHISPER_SAMPLE_RATE
            text, segments, err = _transcribe_array_safe(loaded, chunk, offset_start=offset)
            if err:
                meta["chunk_errors"].append({"chunk": i, "error": err})
            if text:
                all_text_parts.append(text)
            all_segments.extend(segments)
        transcript = " ".join(t for t in all_text_parts if t).strip()

    out: dict[str, Any] = {
        "transcript": transcript,
        "segments": all_segments,
        "duration": duration,
        "transcription_meta": meta,
    }

    if save:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = safe_stem(wav_path)
        out_path = TRANSCRIPTS_DIR / f"{stem}.json"
        payload = {
            "source_audio": str(wav_path.resolve()),
            "transcript": transcript,
            "segments": all_segments,
            "duration": duration,
            "model": model_name,
            "transcription_meta": meta,
        }
        try:
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Saved transcript: %s", out_path)
        except OSError as e:
            logger.warning("Could not save transcript JSON: %s", e)

    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    import sys

    ensure_dirs()
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        wavs = sorted(AUDIO_INPUTS_DIR.glob("*.wav"))
        if not wavs:
            print(f"No .wav files in {AUDIO_INPUTS_DIR}. Add a file or pass a path.")
            sys.exit(1)
        path = wavs[0]
        print(f"Using sample: {path}")

    result = transcribe_wav(path)
    print("duration:", result["duration"])
    print("transcript:", result["transcript"][:500] + ("..." if len(result["transcript"]) > 500 else ""))
    print("segments:", len(result["segments"]))
    print("meta:", result.get("transcription_meta"))


if __name__ == "__main__":
    main()
