"""
Human-readable cognitive signal report (not medical advice).
"""

from __future__ import annotations

from typing import Any, Literal

from .utils import finite_float, safe_int


def _band(
    value: float,
    low: float,
    high: float,
) -> Literal["low", "moderate", "high"]:
    v = finite_float(value, 0.0)
    if v <= low:
        return "low"
    if v >= high:
        return "high"
    return "moderate"


def _speech_rate_band(wpm: float) -> str:
    w = finite_float(wpm, 0.0)
    if w < 100:
        return "low"
    if w > 160:
        return "high"
    return "moderate"


def _ttr_band(ttr: float) -> str:
    return _band(ttr, 0.42, 0.58)


def _filler_band(n: int) -> str:
    if n <= 3:
        return "low"
    if n >= 10:
        return "high"
    return "moderate"


def _drift_band(d: float) -> str:
    return _band(d, 0.15, 0.35)


def build_interpretation(features: dict[str, Any], risk_score: float) -> list[str]:
    bullets: list[str] = []
    rs = finite_float(features.get("speech_rate"), 0.0)
    ttr = finite_float(features.get("ttr"), 0.0)
    fillers = safe_int(features.get("filler_count"), 0)
    pause = finite_float(features.get("pause_ratio"), 0.0)
    drift = finite_float(features.get("semantic_drift"), 0.0)
    asl = finite_float(features.get("avg_sentence_len"), 0.0)
    risk = finite_float(risk_score, 0.0)

    if rs < 105:
        bullets.append("Slower speaking rate than typical conversational pace")
    if ttr < 0.45:
        bullets.append("Lower lexical variety (type–token ratio)")
    if fillers >= 8:
        bullets.append("Frequent hesitation markers (fillers)")
    if pause > 0.28:
        bullets.append("Higher estimated silent pause proportion")
    if drift > 0.3:
        bullets.append("Larger semantic shifts between consecutive sentences")
    if asl > 24:
        bullets.append("Longer average sentence length")

    if not bullets:
        bullets.append("No strong directional indicators in this snapshot")

    if risk >= 0.65:
        bullets.append("Overall pattern suggests elevated cognitive signal score on this heuristic")
    elif risk <= 0.35:
        bullets.append("Overall pattern suggests lower cognitive signal score on this heuristic")

    return bullets


def render_report(
    features: dict[str, Any],
    risk_score: float,
    *,
    title: str = "Speech Biomarker Report",
) -> str:
    sr = finite_float(features.get("speech_rate"), 0.0)
    ttr = finite_float(features.get("ttr"), 0.0)
    fillers = safe_int(features.get("filler_count"), 0)
    drift = finite_float(features.get("semantic_drift"), 0.0)
    pause = finite_float(features.get("pause_ratio"), 0.0)
    asl = finite_float(features.get("avg_sentence_len"), 0.0)
    risk_score = finite_float(risk_score, 0.0)

    lines = [
        f"=== {title} ===",
        "",
        f"Speech rate: {sr:.0f} WPM ({_speech_rate_band(sr)})",
        f"Pause / silence estimate: {pause:.2f} ({_band(pause, 0.15, 0.35)})",
        f"Lexical diversity (TTR): {ttr:.2f} ({_ttr_band(ttr)})",
        f"Filler words: {fillers} ({_filler_band(fillers)})",
        f"Avg sentence length: {asl:.1f} words",
        f"Semantic drift (adjacent sentences): {drift:.2f} ({_drift_band(drift)})",
        "",
        f"Final cognitive signal score: {risk_score:.2f}",
        "",
        "Interpretation:",
    ]
    for b in build_interpretation(features, risk_score):
        lines.append(f"- {b}")
    lines.append("")
    lines.append(
        "Note: This is a research-style signal score, not a diagnosis or medical assessment."
    )
    return "\n".join(lines)


def main() -> None:
    from .inference import run_pipeline
    from .utils import AUDIO_INPUTS_DIR

    wavs = sorted(AUDIO_INPUTS_DIR.glob("*.wav"))
    if not wavs:
        print("Add a .wav under audio_inputs/ to generate a report.")
        return
    out = run_pipeline(wavs[0])
    text = render_report(out["features"], out["risk_score"])
    print(text)


if __name__ == "__main__":
    main()
