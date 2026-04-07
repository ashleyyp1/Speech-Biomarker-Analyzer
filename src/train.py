"""
Train a discriminative tabular model on feature JSONs.

Default: gradient-boosted trees (LightGBM, or sklearn HistGradientBoosting if LightGBM
is unavailable) + probability calibration (Platt / isotonic via CalibratedClassifierCV).

Legacy options: random forest, logistic regression (no calibration unless requested).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import FEATURE_COLS, build_dataframe
from .utils import MODELS_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def _make_demo_feature_files(features_dir: Path, n: int = 24, seed: int = 42) -> None:
    """Write synthetic feature JSONs so training can run without real recordings."""
    rng = np.random.default_rng(seed)
    features_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        high = i % 3 == 0
        if high:
            row = {
                "speech_rate": float(rng.normal(95, 8)),
                "pause_ratio": float(rng.uniform(0.25, 0.45)),
                "ttr": float(rng.uniform(0.28, 0.42)),
                "filler_count": int(rng.integers(6, 18)),
                "avg_sentence_len": float(rng.uniform(18, 35)),
                "semantic_drift": float(rng.uniform(0.2, 0.55)),
            }
        else:
            row = {
                "speech_rate": float(rng.normal(135, 12)),
                "pause_ratio": float(rng.uniform(0.08, 0.22)),
                "ttr": float(rng.uniform(0.48, 0.72)),
                "filler_count": int(rng.integers(0, 5)),
                "avg_sentence_len": float(rng.uniform(10, 20)),
                "semantic_drift": float(rng.uniform(0.0, 0.22)),
            }
        path = features_dir / f"demo_sample_{i:03d}.json"
        path.write_text(
            json.dumps(
                {
                    "source_audio": "synthetic",
                    "transcript_excerpt": "",
                    **row,
                    "repetition_ratio": float(rng.uniform(0.05, 0.35)),
                    "duration_sec": float(rng.uniform(25, 90)),
                    "word_count": int(rng.integers(40, 200)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {n} demo feature files to {features_dir}")


def _gbdt_base_estimator(seed: int) -> tuple[Pipeline, str]:
    """Default strong tabular learner; LightGBM if usable, else sklearn HGBDT."""
    try:
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        backend = "lightgbm"
    except Exception as e:
        logger.info("LightGBM not used (%s); using sklearn HistGradientBoostingClassifier.", e)
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=200,
            learning_rate=0.05,
            random_state=seed,
            class_weight="balanced",
        )
        backend = "sklearn_histgradientboosting"

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipe, backend


def _build_base_pipeline(model_kind: str, seed: int) -> tuple[Pipeline, str | None]:
    if model_kind == "lr":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000, class_weight="balanced", random_state=seed
                        ),
                    ),
                ]
            ),
            "lr",
        )
    if model_kind == "rf":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=6,
                            random_state=seed,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "rf",
        )
    pipe, backend = _gbdt_base_estimator(seed)
    return pipe, backend


def _maybe_calibrate(
    base: Pipeline,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    calibrate: bool,
    min_train_for_calibration: int,
) -> tuple[Pipeline | CalibratedClassifierCV, dict]:
    """
    Wrap with CalibratedClassifierCV when there are two classes and enough rows.
    Uses sigmoid (Platt) on smaller sets, isotonic when n_train is larger.
    """
    meta: dict = {"calibrated": False, "method": None, "cv": None}
    n_train = len(X_train)
    if (
        not calibrate
        or y_train.nunique() < 2
        or n_train < min_train_for_calibration
    ):
        return base, meta

    # cv must be at least 2 and leave enough points per fold
    cv = min(5, max(2, n_train // 5))
    if cv < 2 or n_train < cv * 2:
        return base, meta

    method = "sigmoid" if n_train < 80 else "isotonic"
    wrapped = CalibratedClassifierCV(
        base,
        method=method,
        cv=cv,
    )
    meta = {"calibrated": True, "method": method, "cv": int(cv)}
    return wrapped, meta


def train(
    *,
    model_kind: str = "gbdt",
    test_size: float = 0.25,
    seed: int = 42,
    demo: bool = False,
    feature_prefix: str | None = None,
    calibrate: bool = True,
    min_train_for_calibration: int = 12,
) -> dict:
    ensure_dirs()
    from .utils import FEATURES_DIR

    if demo:
        _make_demo_feature_files(FEATURES_DIR)

    X, y = build_dataframe(label_mode="binary", name_prefix=feature_prefix)
    vc = y.value_counts()
    strat = (
        y
        if y.nunique() > 1 and int(vc.min()) >= 2 and len(y) >= 8
        else None
    )
    n = len(X)
    if n <= 1:
        X_train, y_train = X, y
        X_test, y_test = X.copy(), y.copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat
        )

    base, backend = _build_base_pipeline(model_kind, seed)

    clf, cal_meta = _maybe_calibrate(
        base,
        X_train=X_train,
        y_train=y_train,
        seed=seed,
        calibrate=calibrate and model_kind in ("gbdt", "lr", "rf"),
        min_train_for_calibration=min_train_for_calibration,
    )

    clf.fit(X_train, y_train)

    est_classes = getattr(clf, "classes_", None)
    if est_classes is None and hasattr(clf, "named_steps"):
        est_classes = clf.named_steps["clf"].classes_
    if est_classes is None and hasattr(clf, "estimator"):
        inner = clf.estimator
        est_classes = getattr(inner, "classes_", None)
        if est_classes is None and hasattr(inner, "named_steps"):
            est_classes = inner.named_steps["clf"].classes_

    proba_full = clf.predict_proba(X_test)
    if proba_full.shape[1] >= 2:
        proba = proba_full[:, 1]
    elif est_classes is not None and len(est_classes) == 1:
        only = int(est_classes[0])
        proba = np.full(len(X_test), 1.0 if only == 1 else 0.0)
    else:
        proba = proba_full[:, 0]
    pred = (proba >= 0.5).astype(int)

    metrics: dict = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "model_kind": model_kind,
        "gbdt_backend": backend if model_kind == "gbdt" else None,
    }
    metrics.update(cal_meta)

    try:
        v = float(roc_auc_score(y_test, proba))
        metrics["roc_auc"] = v if v == v else None
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["brier"] = float(brier_score_loss(y_test, proba))
    except ValueError:
        metrics["brier"] = None

    bundle = {
        "model": clf,
        "feature_names": FEATURE_COLS,
        "metrics": metrics,
        "label": "binary heuristic risk (1 = elevated signal); probabilities calibrated when enabled",
        "train_feature_prefix": feature_prefix,
        "calibration": cal_meta,
    }
    out_path = MODELS_DIR / "model.pkl"
    joblib.dump(bundle, out_path)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model bundle to {out_path}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train tabular risk model (default: GBDT + calibration)",
    )
    p.add_argument(
        "--model",
        choices=("gbdt", "rf", "lr"),
        default="gbdt",
        help="gbdt = LightGBM or sklearn HGBDT + optional calibration (default)",
    )
    p.add_argument("--demo", action="store_true", help="Create synthetic feature JSONs then train")
    p.add_argument(
        "--feature-prefix",
        default=None,
        metavar="PREFIX",
        help="Only use features/*.json files whose names start with PREFIX (e.g. cv_)",
    )
    p.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable probability calibration (raw estimator probabilities)",
    )
    p.add_argument(
        "--min-calibration-n",
        type=int,
        default=12,
        metavar="N",
        help="Minimum training rows to enable calibration (default 12)",
    )
    args = p.parse_args()
    train(
        model_kind=args.model,
        demo=args.demo,
        feature_prefix=args.feature_prefix,
        calibrate=not args.no_calibrate,
        min_train_for_calibration=args.min_calibration_n,
    )


if __name__ == "__main__":
    main()
