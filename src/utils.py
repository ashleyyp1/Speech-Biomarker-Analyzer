"""Shared paths and helpers for the Speech Biomarker Analyzer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Project root: speech_biomarker_analyzer/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DATA_DIR = DATA_DIR / "public_audio"
AUDIO_INPUTS_DIR = PROJECT_ROOT / "audio_inputs"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        PUBLIC_DATA_DIR,
        AUDIO_INPUTS_DIR,
        TRANSCRIPTS_DIR,
        FEATURES_DIR,
        MODELS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def safe_stem(path: str | Path) -> str:
    """Filename without extension, safe for use as IDs."""
    return Path(path).stem


def finite_float(x: Any, default: float = 0.0) -> float:
    """Parse float; return default for None, NaN, inf, or bad types."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return default
