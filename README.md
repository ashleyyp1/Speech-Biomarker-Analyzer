# Speech Biomarker Analyzer

A modular Python pipeline that processes short speech recordings: **transcription** (Whisper), **acoustic + linguistic + semantic features**, a **tabular risk / cognitive signal score** (trained model), and a **human-readable report**.

**Important:** This is a **research and prototyping** tool. Outputs are **signals for exploration**, not a medical diagnosis or clinical assessment.

## Features

- **Transcription** — OpenAI Whisper with preprocessing, long-audio chunking, and graceful handling of load/ASR failures.
- **Features** — Speech rate (WPM), pause ratio, type–token ratio, filler counts, average sentence length, semantic drift (sentence embeddings), plus metadata like `feature_issues` when something degrades.
- **Dataset** — Loads `features/*.json`, builds a pandas table, applies **heuristic** labels for training (replace with real labels when you have them).
- **Training** — Default **gradient-boosted trees** (LightGBM, or scikit-learn `HistGradientBoostingClassifier` if LightGBM is unavailable) with optional **probability calibration** (`CalibratedClassifierCV`). Alternatives: random forest, logistic regression.
- **Inference** — End-to-end WAV → transcript → features → risk score; safe behavior for edge cases (e.g. single-class models).
- **Report** — Text summary with qualitative bands and a non-clinical disclaimer.
- **Streamlit app** — Upload audio, run the pipeline, inspect report and pipeline health.
- **Public data prep** — Optional script to sample public corpora (e.g. LibriSpeech dummy, Common Voice) and write features for experimentation.

## Requirements

- Python 3.10+ recommended  
- See `requirements.txt` (includes `torch`, `openai-whisper`, `librosa`, `scikit-learn`, `sentence-transformers`, `streamlit`, `datasets`, etc.)

## Setup

```bash
cd speech_biomarker_analyzer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Whisper and embedding models download weights on first use (disk + network).

## Quick start

### 1. Train a model (synthetic demo features)

```bash
python -m src.train --demo
```

Writes demo feature JSONs under `features/`, trains, saves `models/model.pkl`.

### 2. Run inference on a WAV

```bash
python -m src.inference path/to/audio.wav
```

Or drop a file in `audio_inputs/` and:

```bash
python -m src.inference
```

### 3. Streamlit UI

```bash
streamlit run app.py
```

### 4. Optional — features from public audio

```bash
python -m src.public_data --source librispeech_dummy --max-samples 50 --whisper tiny
python -m src.train --feature-prefix ls_
```

Common Voice requires accepting dataset terms on Hugging Face; use `--source common_voice` when configured.

## Training CLI

| Command | Purpose |
|--------|---------|
| `python -m src.train --demo` | Generate synthetic features + train |
| `python -m src.train --feature-prefix ls_` | Train only on files like `ls_*.json` |
| `python -m src.train --no-calibrate` | Disable probability calibration |
| `python -m src.train --model rf` | Random forest instead of GBDT |
| `python -m src.train --help` | Full options |

## Project layout

```
speech_biomarker_analyzer/
├── app.py                 # Streamlit demo
├── requirements.txt
├── audio_inputs/          # Optional local WAVs
├── data/                  # Manifests, public_audio when using public_data
├── features/              # Feature JSON (many gitignored after generation)
├── models/                # model.pkl (gitignored)
├── transcripts/         # Transcript JSON (gitignored)
└── src/
    ├── transcribe.py
    ├── features.py
    ├── dataset.py
    ├── train.py
    ├── inference.py
    ├── report.py
    ├── public_data.py
    └── utils.py
```

Run modules from the **`speech_biomarker_analyzer`** directory so imports resolve (`python -m src.train`, etc.).

## Git and large files

`.gitignore` excludes generated artifacts (e.g. `models/*.pkl`, many `features/*.json`, WAVs). After cloning, run **`python -m src.train --demo`** (or your data prep) to recreate a local model.

## Limitations

- Training labels are **heuristic** unless you plug in real outcomes.
- **Calibration** needs enough data and two classes; small runs may skip calibration.
- **Performance metrics** on tiny splits are noisy; use rigorous CV and held-out data for serious evaluation.

## License

Add a `LICENSE` file in the repository if you want to specify terms for others.
