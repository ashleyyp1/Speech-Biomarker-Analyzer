"""
Streamlit demo: upload audio, run pipeline, view cognitive signal report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import run_pipeline  # noqa: E402
from src.report import render_report  # noqa: E402
from src.utils import MODELS_DIR, finite_float  # noqa: E402

st.set_page_config(page_title="Speech Biomarker Analyzer", layout="wide")
st.title("Speech Biomarker Analyzer")
st.caption(
    "Research-style cognitive signal score from speech + text features — not a medical diagnosis."
)

if not (MODELS_DIR / "model.pkl").is_file():
    st.warning(
        "No trained model found. Run `python -m src.train --demo` or prepare public data "
        "then `python -m src.train --feature-prefix cv_`."
    )

uploaded = st.file_uploader("Upload a .wav file", type=["wav"])

whisper_name = st.selectbox("Whisper model size", ["tiny", "base", "small"], index=1)

if uploaded and st.button("Run analysis"):
    suffix = Path(uploaded.name).suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Transcribing and extracting features…"):
            out = run_pipeline(
                tmp_path,
                whisper_model=whisper_name,
                save_intermediate=False,
            )
        st.subheader("Report")
        st.text(render_report(out["features"], out["risk_score"]))

        with st.expander("Pipeline health (ASR / features)"):
            st.json(
                {
                    "transcription_meta": out.get("transcription_meta"),
                    "feature_issues": out.get("feature_issues"),
                }
            )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Feature breakdown")
            st.json(out["features"])
        with c2:
            st.subheader("Transcript")
            st.write(out["transcript"] or "(empty)")
            st.metric("Duration (s)", f"{finite_float(out.get('duration'), 0.0):.2f}")
            st.metric("Cognitive signal score", f"{out['risk_score']:.3f}")
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.exception(e)
    finally:
        tmp_path.unlink(missing_ok=True)
