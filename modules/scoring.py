"""
scoring.py
-----------
Combines semantic similarity, key-point coverage, and fluency metrics
into a final weighted comprehension score and structured feedback.
"""


def compute_fluency_score(audio_features: dict) -> float:
    """
    Converts raw fluency metrics into a 0-100 fluency score.

    Penalizes:
        - high filler word ratio
        - high pause ratio (too much silence)
    Rewards:
        - moderate, consistent RMS energy
        - speaking rate within a natural range (110-160 wpm)
    """
    filler_ratio = audio_features["fillers"]["filler_ratio"]
    pause_ratio = audio_features["pause"]["pause_ratio"]
    wpm = audio_features["speaking_rate_wpm"]

    # Start at 100 and apply penalties
    score = 100.0

    # Filler word penalty: every 1% filler ratio costs ~4 points, capped at 30
    filler_penalty = min(filler_ratio * 100 * 4, 30)
    score -= filler_penalty

    # Pause penalty: ideal pause ratio ~10-20%; penalize beyond that
    if pause_ratio > 0.35:
        score -= min((pause_ratio - 0.35) * 100, 25)
    elif pause_ratio < 0.05:
        # Too few pauses can mean rushed, unnatural speech
        score -= 5

    # Speaking rate penalty: ideal range 110-160 wpm
    if wpm > 0:
        if wpm < 90:
            score -= min((90 - wpm) * 0.4, 15)
        elif wpm > 180:
            score -= min((wpm - 180) * 0.4, 15)

    return round(max(0.0, min(100.0, score)), 2)


def compute_final_score(semantic_score: float, coverage_ratio: float, fluency_score: float) -> float:
    """
    Computes the final weighted comprehension score.

    Weighting:
        50% semantic similarity (how close the explanation is in meaning)
        25% key point coverage (how many core ideas were mentioned)
        25% fluency (delivery quality)
    """
    coverage_score = coverage_ratio * 100
    final = (0.50 * semantic_score) + (0.25 * coverage_score) + (0.25 * fluency_score)
    return round(final, 2)


def get_final_label(final_score: float) -> str:
    """Maps the final score to a qualitative comprehension label."""
    if final_score >= 75:
        return "Strong Understanding"
    elif final_score >= 50:
        return "Moderate Understanding"
    else:
        return "Poor Understanding"


def generate_feedback(semantic_score, coverage_result, audio_features, fluency_score, final_score) -> list:
    """
    Generates a list of qualitative feedback strings based on all
    computed metrics. Used in both the dashboard and PDF report.
    """
    feedback = []

    # Semantic feedback
    if semantic_score >= 75:
        feedback.append("Your explanation closely matches the core meaning of the concept.")
    elif semantic_score >= 50:
        feedback.append("Your explanation captures the general idea but lacks some precision or depth.")
    else:
        feedback.append("Your explanation deviates significantly from the expected concept. Review the core definition again.")

    # Coverage feedback
    if coverage_result["missed"]:
        missed_str = ", ".join(coverage_result["missed"])
        feedback.append(f"You missed these key points: {missed_str}.")
    else:
        feedback.append("You covered all the essential key points of the concept.")

    # Filler word feedback
    filler_ratio = audio_features["fillers"]["filler_ratio"]
    if filler_ratio > 0.06:
        feedback.append("You used a high number of filler words (e.g., um, like, uh). Try pausing silently instead of using fillers.")
    elif filler_ratio > 0.02:
        feedback.append("You used a moderate number of filler words. Practicing slow, deliberate speech can help reduce these.")
    else:
        feedback.append("Great job keeping filler word usage low.")

    # Pause feedback
    pause_ratio = audio_features["pause"]["pause_ratio"]
    if pause_ratio > 0.35:
        feedback.append("There were long periods of silence in your explanation. Try to organize your thoughts beforehand to reduce hesitation.")
    elif pause_ratio < 0.05:
        feedback.append("Your speech had very few natural pauses, which may sound rushed. Brief pauses can improve clarity.")
    else:
        feedback.append("Your pause pattern was natural and well balanced.")

    # Speaking rate feedback
    wpm = audio_features["speaking_rate_wpm"]
    if wpm > 180:
        feedback.append(f"Your speaking rate was fast (~{wpm} words/min). Slowing down can improve listener comprehension.")
    elif 0 < wpm < 90:
        feedback.append(f"Your speaking rate was slow (~{wpm} words/min). A slightly faster, confident pace may help engagement.")

    # Overall
    feedback.append(f"Final Comprehension Score: {final_score}/100 - {get_final_label(final_score)}.")

    return feedback