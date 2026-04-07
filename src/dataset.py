"""
Build a pandas DataFrame from feature JSON files and heuristic risk labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .utils import FEATURES_DIR, ensure_dirs

FEATURE_COLS = [
    "speech_rate",
    "pause_ratio",
    "ttr",
    "filler_count",
    "avg_sentence_len",
    "semantic_drift",
]


def _heuristic_risk_continuous(row: pd.Series) -> float:
    """
    Map features to a rough 0–1 risk score (not clinical).
    Higher = more 'at-risk' style signal under simple rules.
    """
    # Normalize components to [0, 1] with soft thresholds
    sr = float(row["speech_rate"])
    ttr = float(row["ttr"])
    fillers = float(row["filler_count"])
    pause = float(row["pause_ratio"])
    drift = float(row["semantic_drift"])
    slen = float(row["avg_sentence_len"])

    # Slow speech: assume < 100 WPM elevated
    slow = np.clip((120 - sr) / 80.0, 0.0, 1.0) if sr < 120 else 0.0
    # Low diversity
    low_div = np.clip(0.55 - ttr, 0.0, 0.55) / 0.55
    # Fillers (cap influence)
    filler_score = np.clip(fillers / 12.0, 0.0, 1.0)
    # Long pauses
    pause_score = np.clip((pause - 0.15) / 0.5, 0.0, 1.0)
    # Drift
    drift_score = np.clip(drift / 0.5, 0.0, 1.0)
    # Very long sentences sometimes correlate with disorganization (weak)
    long_sent = np.clip((slen - 22) / 30.0, 0.0, 1.0)

    raw = (
        0.22 * slow
        + 0.22 * low_div
        + 0.18 * filler_score
        + 0.15 * pause_score
        + 0.13 * drift_score
        + 0.10 * long_sent
    )
    return float(np.clip(raw, 0.0, 1.0))


def build_dataframe(
    features_dir: Path | None = None,
    *,
    label_mode: Literal["continuous", "binary"] = "binary",
    threshold: float = 0.45,
    name_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load all *.json from features_dir, return X (FEATURE_COLS) and y.
    Skips files missing required keys or non-finite values.
    """
    ensure_dirs()
    root = features_dir or FEATURES_DIR
    paths = sorted(root.glob("*.json"))
    if name_prefix:
        paths = [p for p in paths if p.name.startswith(name_prefix)]
    if not paths:
        raise FileNotFoundError(f"No matching feature JSON files in {root}")

    rows: list[dict] = []
    for p in paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skip unreadable {p.name}: {e}")
            continue
        try:
            row = {c: float(obj[c]) for c in FEATURE_COLS}
        except (KeyError, TypeError, ValueError) as e:
            print(f"Skip invalid features {p.name}: {e}")
            continue
        if not all(np.isfinite(row[c]) for c in FEATURE_COLS):
            print(f"Skip non-finite features {p.name}")
            continue
        row["_file"] = p.name
        rows.append(row)
    if not rows:
        raise ValueError("No valid feature rows after filtering")

    df = pd.DataFrame(rows)
    cont = df.apply(_heuristic_risk_continuous, axis=1)
    df["risk_continuous"] = cont

    if label_mode == "continuous":
        y = cont
    else:
        y = (cont >= threshold).astype(int)

    X = df[FEATURE_COLS].astype(float)
    return X, y


def main() -> None:
    X, y = build_dataframe()
    print("Samples:", len(X))
    print(X.describe())
    print("y (binary) value counts:\n", y.value_counts())


if __name__ == "__main__":
    main()
