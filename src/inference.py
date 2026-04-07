"""
End-to-end: transcribe -> features -> risk score (cognitive signal, not diagnosis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .dataset import FEATURE_COLS
from .features import extract_features
from .transcribe import transcribe_wav
from .utils import MODELS_DIR, ensure_dirs, finite_float, safe_stem


def _sklearn_classes(model: Any) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("clf")
        if est is not None and hasattr(est, "classes_"):
            return np.asarray(est.classes_)
    return np.array([0], dtype=int)


def load_model_bundle(path: Path | None = None) -> dict[str, Any]:
    ensure_dirs()
    p = path or (MODELS_DIR / "model.pkl")
    if not p.is_file():
        raise FileNotFoundError(
            f"No model at {p}. Run: python -m src.train --demo "
            "(default: GBDT + calibration) or "
            "python -m src.public_data && python -m src.train --feature-prefix ls_"
        )
    return joblib.load(p)


def predict_risk_score(feature_vector: pd.DataFrame, bundle: dict[str, Any]) -> float:
    """
    Probability of positive class (label 1) when the model is binary.
    If the model was trained on a single class (only 0 or only 1), sklearn
    emits a single-column proba — handle without indexing errors.
    """
    model = bundle["model"]
    proba_full = model.predict_proba(feature_vector)
    if proba_full.shape[1] >= 2:
        return float(np.clip(proba_full[0, 1], 0.0, 1.0))
    cls = _sklearn_classes(model)
    if cls.size == 1:
        return 1.0 if int(cls[0]) == 1 else 0.0
    return float(np.clip(proba_full[0, 0], 0.0, 1.0))


def run_pipeline(
    wav_path: str | Path,
    *,
    whisper_model: str = "base",
    save_intermediate: bool = True,
    model_path: Path | None = None,
) -> dict[str, Any]:
    wav_path = Path(wav_path)
    tr = transcribe_wav(wav_path, model_name=whisper_model, save=save_intermediate)
    transcript = tr.get("transcript")
    if not isinstance(transcript, str):
        transcript = str(transcript or "")
    duration = finite_float(tr.get("duration"), 0.0)

    feats = extract_features(
        wav_path,
        transcript,
        duration_sec=duration,
        save=save_intermediate,
    )

    bundle = load_model_bundle(model_path)
    try:
        X = pd.DataFrame([[finite_float(feats.get(c), 0.0) for c in FEATURE_COLS]], columns=FEATURE_COLS)
        risk = predict_risk_score(X, bundle)
    except Exception as e:
        raise RuntimeError(
            "Model prediction failed (check that models/model.pkl matches FEATURE_COLS). "
            f"Original error: {e}"
        ) from e

    core_feats = {k: finite_float(feats.get(k), 0.0) for k in FEATURE_COLS}
    return {
        "risk_score": float(np.clip(risk, 0.0, 1.0)),
        "features": core_feats,
        "features_full": feats,
        "transcript": transcript,
        "segments": tr.get("segments") or [],
        "duration": duration,
        "audio_id": safe_stem(wav_path),
        "transcription_meta": tr.get("transcription_meta"),
        "feature_issues": feats.get("feature_issues") or [],
    }


def main() -> None:
    import json
    import sys

    from .utils import AUDIO_INPUTS_DIR

    ensure_dirs()
    if len(sys.argv) > 1:
        wav = Path(sys.argv[1])
    else:
        wavs = sorted(AUDIO_INPUTS_DIR.glob("*.wav"))
        if not wavs:
            print("Pass a .wav path or add one to audio_inputs/.")
            sys.exit(1)
        wav = wavs[0]

    out = run_pipeline(wav)
    print(json.dumps({k: out[k] for k in ("risk_score", "transcript", "duration")}, indent=2))
    print("features:", json.dumps(out["features"], indent=2))


if __name__ == "__main__":
    main()
