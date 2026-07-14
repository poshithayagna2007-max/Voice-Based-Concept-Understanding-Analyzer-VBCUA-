"""
audio_features.py
-------------------
Extracts speech-fluency related audio features using Librosa:
pause ratio, RMS energy, speaking rate, and waveform data.
Also detects filler words from the transcribed text.
"""

import re
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

FILLER_WORDS = [
    "um", "umm", "uh", "uhh", "like", "you know", "actually",
    "basically", "literally", "so", "ah", "hmm", "i mean", "well",
]


def load_audio(file_path: str, sr: int = 16000):
    """Loads an audio file and returns the waveform and sample rate."""
    y, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return y, sample_rate


def compute_rms_energy(y: np.ndarray) -> dict:
    """Computes RMS (loudness/energy) statistics for the waveform."""
    rms = librosa.feature.rms(y=y)[0]
    return {
        "mean_rms": round(float(np.mean(rms)), 5),
        "std_rms": round(float(np.std(rms)), 5),
        "max_rms": round(float(np.max(rms)), 5),
    }


def compute_pause_ratio(y: np.ndarray, sr: int, top_db: int = 30) -> dict:
    """
    Computes the ratio of silence/pause duration to total duration
    using librosa's silence-split based on amplitude threshold.
    """
    total_duration = librosa.get_duration(y=y, sr=sr)
    if total_duration == 0:
        return {"pause_ratio": 0.0, "speaking_duration": 0.0, "total_duration": 0.0}

    intervals = librosa.effects.split(y, top_db=top_db)
    speaking_duration = sum((end - start) for start, end in intervals) / sr
    pause_duration = total_duration - speaking_duration
    pause_ratio = pause_duration / total_duration if total_duration > 0 else 0.0

    return {
        "pause_ratio": round(float(pause_ratio), 3),
        "speaking_duration": round(float(speaking_duration), 2),
        "total_duration": round(float(total_duration), 2),
    }


def detect_filler_words(text: str) -> dict:
    """
    Counts occurrences of common filler words in the transcribed text.
    """
    text_lower = text.lower()
    words = re.findall(r"\b[\w']+\b", text_lower)
    total_words = max(len(words), 1)

    counts = {}
    for filler in FILLER_WORDS:
        # Use word-boundary regex to also catch multi-word fillers like "you know"
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, text_lower)
        if matches:
            counts[filler] = len(matches)

    total_fillers = sum(counts.values())
    filler_ratio = round(total_fillers / total_words, 4)

    return {
        "filler_counts": counts,
        "total_fillers": total_fillers,
        "total_words": total_words,
        "filler_ratio": filler_ratio,
    }


def estimate_speaking_rate(text: str, speaking_duration_sec: float) -> float:
    """Estimates words per minute based on transcribed text and speaking duration."""
    word_count = len(re.findall(r"\b[\w']+\b", text))
    if speaking_duration_sec <= 0:
        return 0.0
    wpm = (word_count / speaking_duration_sec) * 60
    return round(wpm, 1)


def generate_waveform_plot(y: np.ndarray, sr: int) -> bytes:
    """
    Generates a waveform plot image and returns it as PNG bytes,
    suitable for displaying in Streamlit or embedding in a PDF report.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4F8BF9")
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def analyze_audio(file_path: str, transcribed_text: str) -> dict:
    """
    Master function that runs all audio-feature analyses and returns
    a combined dictionary of fluency metrics.
    """
    y, sr = load_audio(file_path)

    rms_stats = compute_rms_energy(y)
    pause_stats = compute_pause_ratio(y, sr)
    filler_stats = detect_filler_words(transcribed_text)
    speaking_rate = estimate_speaking_rate(transcribed_text, pause_stats["speaking_duration"])
    waveform_png = generate_waveform_plot(y, sr)

    return {
        "rms": rms_stats,
        "pause": pause_stats,
        "fillers": filler_stats,
        "speaking_rate_wpm": speaking_rate,
        "waveform_png": waveform_png,
    }
