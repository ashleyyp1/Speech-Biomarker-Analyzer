"""
Extract acoustic, linguistic, and semantic features from audio + transcript.
Tolerates bad audio IO, empty ASR, and embedding failures (degrades gracefully).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .utils import FEATURES_DIR, ensure_dirs, finite_float, safe_int, safe_stem

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
FILLER_PATTERN = re.compile(
    r"\b(um|uh|ugh|erm|er|ah|hm|hmm|like|you know)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Fallback when ASR omits punctuation (common under noise)
ROUGH_SENTENCE_SPLIT = re.compile(r"[\n\r]+|(?:\s{2,})")

_EMBEDDER_CACHE: dict[str, Any] = {}


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _sentences(text: str) -> list[str]:
    t = text.strip()
    if not t:
        return []
    parts = SENTENCE_SPLIT.split(t)
    out = [p.strip() for p in parts if p.strip()]
    if len(out) <= 1 and len(_word_tokens(t)) > 25:
        out = [p.strip() for p in ROUGH_SENTENCE_SPLIT.split(t) if p.strip()]
    if not out:
        out = [t]
    return out


def _type_token_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _filler_count(text: str) -> int:
    return len(FILLER_PATTERN.findall(text))


def _avg_sentence_length_words(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    lengths = [len(_word_tokens(s)) for s in sents]
    return float(np.mean(lengths))


def _repetition_ratio(words: list[str]) -> float:
    if len(words) < 2:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for w in words:
        if w in seen:
            repeats += 1
        else:
            seen.add(w)
    return repeats / (len(words) - 1)


def _get_sentence_embedder(model_name: str):
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def _pause_ratio_from_audio(y: np.ndarray, sr: int) -> float:
    """Silence ratio; multiple strategies for noisy recordings."""
    if len(y) == 0:
        return 1.0
    try:
        for top_db in (38, 32, 26):
            intervals = librosa.effects.split(y, top_db=top_db)
            speech = sum(int(e - s) for s, e in intervals)
            if speech > 0:
                return float(1.0 - (speech / len(y)))
    except Exception as e:
        logger.warning("pause_ratio split failed: %s", e)
    try:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if rms.size == 0:
            return 0.5
        thr = float(np.percentile(rms, 20))
        quiet = rms < max(thr, 1e-7)
        return float(np.mean(quiet))
    except Exception as e:
        logger.warning("pause_ratio RMS fallback failed: %s", e)
    return 0.5


def _semantic_drift(text: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.0
    try:
        model = _get_sentence_embedder(model_name)
        emb = model.encode(
            sents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
    except Exception as e:
        logger.warning("sentence embedding failed: %s", e)
        return 0.0
    if emb.shape[0] < 2:
        return 0.0
    sims = np.sum(emb[:-1] * emb[1:], axis=1)
    drift = float(np.mean(1.0 - sims))
    return float(np.clip(drift, 0.0, 1.0))


def _load_audio_mono_safe(path: Path) -> tuple[np.ndarray, int, str | None]:
    try:
        y, sr = librosa.load(str(path), sr=WHISPER_SAMPLE_RATE, mono=True)
        return y.astype(np.float32), sr, None
    except Exception as e:
        logger.warning("librosa.load failed for %s: %s", path, e)
    try:
        import soundfile as sf

        raw, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(raw, "ndim", 1) > 1:
            raw = np.mean(raw, axis=1)
        y = librosa.resample(
            np.asarray(raw, dtype=np.float32),
            orig_sr=file_sr,
            target_sr=WHISPER_SAMPLE_RATE,
        )
        return y, WHISPER_SAMPLE_RATE, None
    except Exception as e:
        logger.warning("soundfile fallback failed for %s: %s", path, e)
    return np.zeros(1, dtype=np.float32), WHISPER_SAMPLE_RATE, "audio_load_failed"


def _sanitize_feature_dict(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    for k in (
        "speech_rate",
        "pause_ratio",
        "ttr",
        "avg_sentence_len",
        "semantic_drift",
        "repetition_ratio",
        "duration_sec",
    ):
        if k in out:
            out[k] = finite_float(out[k], 0.0)
    if "filler_count" in out:
        out["filler_count"] = max(0, safe_int(out["filler_count"], 0))
    if "word_count" in out:
        out["word_count"] = max(0, safe_int(out["word_count"], 0))
    return out


def extract_features(
    wav_path: str | Path,
    transcript: str,
    *,
    duration_sec: float | None = None,
    save: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """
    Compute feature dict and optionally save JSON under features/.
    """
    ensure_dirs()
    wav_path = Path(wav_path)
    issues: list[str] = []

    text_in = transcript if isinstance(transcript, str) else str(transcript or "")

    y, sr, load_err = _load_audio_mono_safe(wav_path)
    if load_err:
        issues.append(load_err)

    if duration_sec is None:
        duration_sec = float(len(y) / sr) if sr else 0.0
    duration_sec = finite_float(duration_sec, 0.0)

    words = _word_tokens(text_in)
    word_count = len(words)
    wpm = (word_count / (duration_sec / 60.0)) if duration_sec > 1e-6 else 0.0

    try:
        pause = _pause_ratio_from_audio(y, sr)
    except Exception as e:
        logger.warning("pause_ratio failed: %s", e)
        issues.append("pause_ratio_failed")
        pause = 0.5

    try:
        drift = float(_semantic_drift(text_in, model_name=embedding_model))
    except Exception as e:
        logger.warning("semantic_drift failed: %s", e)
        issues.append("semantic_drift_failed")
        drift = 0.0

    out: dict[str, Any] = {
        "speech_rate": float(wpm),
        "pause_ratio": float(pause),
        "ttr": float(_type_token_ratio(words)),
        "filler_count": int(_filler_count(text_in)),
        "avg_sentence_len": float(_avg_sentence_length_words(text_in)),
        "semantic_drift": drift,
        "repetition_ratio": float(_repetition_ratio(words)),
        "duration_sec": float(duration_sec),
        "word_count": int(word_count),
        "feature_issues": issues,
    }

    out = _sanitize_feature_dict(out)

    if save:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        stem = safe_stem(wav_path)
        feat_path = FEATURES_DIR / f"{stem}.json"
        payload = {
            "source_audio": str(wav_path.resolve()),
            "transcript_excerpt": text_in[:2000],
            **out,
        }
        try:
            feat_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Saved features: %s", feat_path)
        except OSError as e:
            logger.warning("Could not save features JSON: %s", e)

    return out


def load_features_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    from .transcribe import transcribe_wav

    ensure_dirs()
    if len(sys.argv) > 1:
        wav = Path(sys.argv[1])
    else:
        from .utils import AUDIO_INPUTS_DIR

        wavs = sorted(AUDIO_INPUTS_DIR.glob("*.wav"))
        if not wavs:
            print(f"No .wav in {AUDIO_INPUTS_DIR}; pass a path.")
            sys.exit(1)
        wav = wavs[0]

    tr = transcribe_wav(wav, save=True)
    feats = extract_features(wav, tr["transcript"], duration_sec=tr["duration"], save=True)
    print(
        json.dumps(
            {k: feats[k] for k in ("speech_rate", "pause_ratio", "ttr", "filler_count", "avg_sentence_len", "semantic_drift")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
