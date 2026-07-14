"""
semantic_analysis.py
---------------------
Performs semantic similarity scoring between a user's spoken explanation
and a reference concept explanation using Sentence-BERT embeddings.
Also performs key-point coverage analysis.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer, util


@st.cache_resource(show_spinner=False)
def load_sbert_model(model_name: str = "all-MiniLM-L6-v2"):
    """Loads and caches the Sentence-BERT model."""
    return SentenceTransformer(model_name)


def compute_semantic_similarity(user_text: str, reference_text: str) -> float:
    """
    Computes cosine similarity between the embeddings of the user's
    explanation and the reference explanation.

    Returns a float between 0 and 100 (percentage similarity).
    """
    model = load_sbert_model()
    embeddings = model.encode([user_text, reference_text], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    # cosine similarity ranges roughly -1 to 1; clamp and scale to 0-100
    similarity = max(0.0, min(1.0, similarity))
    return round(similarity * 100, 2)


def analyze_key_point_coverage(user_text: str, key_points: list, threshold: float = 0.45) -> dict:
    """
    Checks which key points from the reference concept were covered
    in the user's explanation, using sentence-level semantic similarity.

    Args:
        user_text: the transcribed user explanation
        key_points: list of key concept points to check coverage of
        threshold: cosine similarity threshold above which a point is
                   considered "covered"

    Returns:
        dict with:
            covered: list of key points covered
            missed: list of key points missed
            coverage_ratio: float 0-1
    """
    model = load_sbert_model()

    # Split user text into sentences (simple split, good enough for spoken text)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', user_text.strip())
    sentences = [s for s in sentences if len(s.strip()) > 0]

    if not sentences:
        return {"covered": [], "missed": key_points, "coverage_ratio": 0.0}

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    point_embeddings = model.encode(key_points, convert_to_tensor=True)

    covered = []
    missed = []

    for i, point in enumerate(key_points):
        sims = util.cos_sim(point_embeddings[i], sentence_embeddings)
        max_sim = sims.max().item()
        if max_sim >= threshold:
            covered.append(point)
        else:
            missed.append(point)

    coverage_ratio = len(covered) / len(key_points) if key_points else 0.0

    return {
        "covered": covered,
        "missed": missed,
        "coverage_ratio": round(coverage_ratio, 2),
    }


def get_understanding_label(similarity_score: float) -> str:
    """
    Maps a semantic similarity score (0-100) to a qualitative
    understanding label.
    """
    if similarity_score >= 75:
        return "Strong Understanding"
    elif similarity_score >= 50:
        return "Moderate Understanding"
    else:
        return "Poor Understanding"
