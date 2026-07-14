# VBCUA — Voice-Based Concept Understanding Analyser

An AI-powered web application that evaluates how effectively a user understands
and explains a conceptual topic through spoken communication. It combines
speech-to-text transcription, semantic similarity scoring, audio fluency
analysis, and PDF report generation in a single Streamlit dashboard.

## Features

- **Speech-to-text transcription** using OpenAI Whisper
- **Semantic similarity scoring** against reference concept explanations using Sentence-BERT
- **Key point coverage analysis** — detects which core ideas were mentioned vs missed
- **Speech fluency analysis** using Librosa — filler word detection, pause ratio, RMS energy, speaking rate
- **Waveform visualization**
- **Weighted final comprehension score** (semantic + coverage + fluency)
- **AI-generated structured feedback**
- **Downloadable PDF report**
Project Demo 
Working of the project video
https://drive.google.com/file/d/1rMGxhxHG_NIjDjdrAuU6Kl2C0na4Gl5K/view?usp=sharing
## Project Structure

```
VBCUA/
├── app/
│   └── app.py                 # Main Streamlit application (entry point)
├── modules/
│   ├── transcription.py       # Whisper speech-to-text
│   ├── semantic_analysis.py   # Sentence-BERT similarity + key point coverage
│   ├── audio_features.py      # Librosa fluency metrics + waveform plotting
│   ├── scoring.py             # Score combination + feedback generation
│   └── pdf_report.py          # PDF report builder (fpdf2)
├── reference_data/
│   └── concepts.json          # Reference explanations & key points per concept
├── reports/                   # (optional) saved PDF reports
├── assets/                    # (optional) static assets, logos, etc.
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment** (Python 3.10+ recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Note: `openai-whisper` requires `ffmpeg` to be installed on your system.
> - macOS: `brew install ffmpeg`
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - Windows: download from https://ffmpeg.org and add to PATH

3. **Run the app**

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

## How to Use

1. Select a reference concept from the sidebar (e.g., "Machine Learning").
2. Choose a Whisper model size (start with `base`).
3. Upload an audio recording (wav/mp3/m4a/ogg) of yourself explaining the concept.
4. Click **Analyze Explanation**.
5. Review the dashboard: transcription, semantic analysis, fluency analysis, and feedback tabs.
6. Click **Download PDF Report** to save a structured report.

## Adding New Concepts

Edit `reference_data/concepts.json` and add a new entry:

```json
"Your Concept Name": {
  "reference_explanation": "A clear, complete explanation of the concept...",
  "key_points": ["Key idea 1", "Key idea 2", "Key idea 3"]
}
```

It will automatically appear in the concept dropdown.

## Scoring Methodology

| Component             | Weight | Source                                  |
|------------------------|--------|------------------------------------------|
| Semantic Similarity    | 50%    | Sentence-BERT cosine similarity           |
| Key Point Coverage     | 25%    | Sentence-level matching against key points|
| Fluency Score          | 25%    | Filler words, pause ratio, speaking rate  |

Final score bands:
- **75–100** → Strong Understanding
- **50–74** → Moderate Understanding
- **0–49** → Poor Understanding

## Tech Stack

Python · Streamlit · OpenAI Whisper · Sentence-Transformers (SBERT) · PyTorch ·
Librosa · Matplotlib · fpdf2 · NLTK

## Notes for Beginners

- All core logic lives in `modules/` — each file does one job and is documented with comments.
- `app/app.py` is the only file you run; it imports and orchestrates the modules.
- If model downloads are slow on first run, that's expected — Whisper and SBERT models are downloaded once and cached locally afterward.
