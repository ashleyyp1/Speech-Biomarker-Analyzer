"""
Download a small slice of public English speech data and build feature JSONs for training.

Primary source: Mozilla Common Voice (English, validated split) via Hugging Face `datasets`.
Fallback: tiny LibriSpeech dummy set (no gate, for CI / offline smoke tests).

Labels remain heuristic (same as `dataset.py`) — this script only supplies diverse real audio.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import whisper

from .features import extract_features
from .transcribe import transcribe_wav
from .utils import DATA_DIR, FEATURES_DIR, PUBLIC_DATA_DIR, ensure_dirs, safe_stem

logger = logging.getLogger(__name__)

MANIFEST_NAME = "public_prepare_manifest.jsonl"


def _write_wav_from_array(arr: np.ndarray, sr: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    sf.write(str(dest), x, int(sr))


def _hf_datasets_cache_root() -> Path:
    base = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return Path(base) / "datasets"


def _resolve_hf_audio_path(p: str, dataset: Any) -> str | None:
    if not p:
        return None
    if os.path.isfile(p):
        return p
    name = Path(p).name
    roots: list[Path] = []
    for cf in getattr(dataset, "cache_files", None) or []:
        fn = cf.get("filename") if isinstance(cf, dict) else getattr(cf, "filename", None)
        if fn:
            roots.append(Path(fn).parent)
    for root in roots:
        cand = root / p
        if cand.is_file():
            return str(cand)
        if (root / name).is_file():
            return str(root / name)
        for sub in root.rglob(name):
            if sub.is_file():
                return str(sub)
    # Last resort: search under HF datasets cache (extracted shards live deep in tree)
    cache = _hf_datasets_cache_root()
    if cache.is_dir():
        for sub in cache.rglob(name):
            if sub.is_file():
                return str(sub)
    return None


def _read_audio_dict(audio: Any, dataset: Any | None = None) -> tuple[np.ndarray, int] | None:
    """HF Audio: in-memory bytes, decoded array, or path (resolved against cache)."""
    if not audio:
        return None
    if isinstance(audio, dict):
        if audio.get("array") is not None and audio.get("sampling_rate"):
            return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
        raw = audio.get("bytes")
        if raw:
            try:
                data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception as e:
                logger.warning("soundfile could not decode bytes: %s", e)
                return None
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            return np.asarray(data, dtype=np.float32), int(sr)
        p = audio.get("path")
        if p:
            resolved = _resolve_hf_audio_path(str(p), dataset) if dataset is not None else None
            path = resolved or (p if os.path.isfile(p) else None)
            if not path:
                return None
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            return np.asarray(data, dtype=np.float32), int(sr)
    return None


def _load_common_voice(max_samples: int, seed: int) -> list[dict[str, Any]]:
    from datasets import Audio, load_dataset

    split = f"validated[:{max_samples}]"
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split=split,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed)
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for i in range(min(max_samples, len(ds))):
        row = ds[i]
        audio = row.get("audio")
        got = _read_audio_dict(audio, ds)
        if got is None:
            continue
        arr, sr = got
        rows.append(
            {
                "id": f"cv_{i:05d}",
                "array": arr,
                "sr": sr,
                "meta": {"source": "common_voice_17_en", "original": row.get("path")},
            }
        )
    return rows


def _load_librispeech_dummy(max_samples: int | None = None) -> list[dict[str, Any]]:
    from datasets import Audio, load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
    )
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row = ds[i]
        got = _read_audio_dict(row.get("audio"), ds)
        if got is None:
            continue
        arr, sr = got
        rows.append(
            {
                "id": f"ls_{i:05d}",
                "array": arr,
                "sr": sr,
                "meta": {"source": "librispeech_dummy"},
            }
        )
    return rows


def prepare_public_features(
    *,
    source: str = "common_voice",
    max_samples: int = 200,
    seed: int = 42,
    whisper_model: str = "tiny",
    clear_audio_dir: bool = False,
) -> Path:
    """
    Write WAVs under data/public_audio/ and feature JSONs under features/ with ids cv_* or ls_*.
    Returns path to manifest JSONL.
    """
    ensure_dirs()
    if clear_audio_dir and PUBLIC_DATA_DIR.is_dir():
        shutil.rmtree(PUBLIC_DATA_DIR)
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if source == "common_voice":
        try:
            items = _load_common_voice(max_samples, seed)
        except Exception as e:
            logger.warning("Common Voice load failed (%s). Falling back to librispeech_dummy.", e)
            items = _load_librispeech_dummy(max_samples=max(1, max_samples // 4))
    elif source == "librispeech_dummy":
        items = _load_librispeech_dummy(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown source: {source}")

    if not items:
        raise RuntimeError("No audio items loaded from public data source")

    asr = whisper.load_model(whisper_model)
    manifest = DATA_DIR / MANIFEST_NAME
    n_ok = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for item in items:
            wav_path = PUBLIC_DATA_DIR / f"{item['id']}.wav"
            try:
                _write_wav_from_array(item["array"], item["sr"], wav_path)
            except Exception as e:
                rec = {"id": item["id"], "stage": "write_wav", "ok": False, "error": str(e)}
                mf.write(json.dumps(rec) + "\n")
                logger.warning("Skip %s: %s", item["id"], e)
                continue

            tr: dict[str, Any] = {}
            try:
                tr = transcribe_wav(
                    wav_path,
                    model=asr,
                    model_name=whisper_model,
                    save=True,
                )
            except Exception as e:
                logger.warning("Transcribe failed for %s: %s", wav_path, e)
                tr = {
                    "transcript": "",
                    "segments": [],
                    "duration": float(len(item["array"]) / item["sr"]),
                    "transcription_meta": {"fatal": str(e)},
                }

            try:
                extract_features(
                    wav_path,
                    tr.get("transcript") or "",
                    duration_sec=tr.get("duration"),
                    save=True,
                )
            except Exception as e:
                rec = {
                    "id": item["id"],
                    "stage": "features",
                    "ok": False,
                    "error": str(e),
                }
                mf.write(json.dumps(rec) + "\n")
                logger.warning("Features failed for %s: %s", wav_path, e)
                continue

            stem = safe_stem(wav_path)
            rec = {
                "id": item["id"],
                "wav": str(wav_path),
                "feature_json": str(FEATURES_DIR / f"{stem}.json"),
                "transcript_len": len(tr.get("transcript") or ""),
                "duration": tr.get("duration"),
                "asr_issues": tr.get("transcription_meta", {}),
                "ok": True,
                **(item.get("meta") or {}),
            }
            mf.write(json.dumps(rec) + "\n")
            n_ok += 1

    logger.info("Prepared %d public samples; manifest %s", n_ok, manifest)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Build features from public speech data")
    p.add_argument(
        "--source",
        choices=("common_voice", "librispeech_dummy"),
        default="common_voice",
    )
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--whisper", default="tiny", help="Whisper size (tiny recommended for bulk)")
    p.add_argument(
        "--clear-audio",
        action="store_true",
        help="Delete data/public_audio/ before writing",
    )
    args = p.parse_args()

    manifest = prepare_public_features(
        source=args.source,
        max_samples=args.max_samples,
        seed=args.seed,
        whisper_model=args.whisper,
        clear_audio_dir=args.clear_audio,
    )
    prefix = "cv_" if args.source == "common_voice" else "ls_"
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "train_hint": f"python -m src.train --feature-prefix {prefix}",
                "note": "Common Voice may require HF auth; use --source librispeech_dummy for a small offline-friendly set.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
