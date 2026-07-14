"""
transcription.py
-----------------
Handles speech-to-text transcription using OpenAI Whisper.
"""

import whisper
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    """
    Loads and caches the Whisper model so it is only loaded once per session.

    model_size options: "tiny", "base", "small", "medium", "large"
    "base" is a good default tradeoff between speed and accuracy.
    """
    model = whisper.load_model(model_size)
    return model


def transcribe_audio(file_path: str, model_size: str = "base") -> dict:
    """
    Transcribes an audio file to text using Whisper.

    Args:
        file_path: path to the audio file (wav/mp3/m4a) on disk.
        model_size: which whisper model to use.

    Returns:
        dict with keys:
            text: full transcribed text
            segments: list of segment dicts (with start/end/text)
            language: detected language code
    """
    model = load_whisper_model(model_size)
    result = model.transcribe(file_path, fp16=False, verbose=False)

    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", "unknown"),
    }
