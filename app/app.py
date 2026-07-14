"""
app.py
-------
Main Streamlit application for the Voice-Based Concept Understanding
Analyser (VBCUA). Run with: streamlit run app/app.py
"""

import os
import sys
import json
import tempfile

import streamlit as st
import pandas as pd

# Make the modules/ package importable regardless of working directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.transcription import transcribe_audio
from modules.semantic_analysis import (
    compute_semantic_similarity,
    analyze_key_point_coverage,
    get_understanding_label,
)
from modules.audio_features import analyze_audio
from modules.scoring import compute_fluency_score, compute_final_score, get_final_label, generate_feedback
from modules.pdf_report import generate_pdf_report


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VBCUA | Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
)

REFERENCE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference_data",
    "concepts.json",
)


@st.cache_data
def load_reference_concepts():
    with open(REFERENCE_DATA_PATH, "r") as f:
        return json.load(f)


def init_session_state():
    defaults = {
        "analysis_done": False,
        "transcribed_text": None,
        "semantic_score": None,
        "coverage_result": None,
        "audio_features": None,
        "fluency_score": None,
        "final_score": None,
        "final_label": None,
        "feedback_list": None,
        "concept_name": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()
concepts = load_reference_concepts()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎙️ VBCUA")
    st.caption("Voice-Based Concept Understanding Analyser")
    st.markdown("---")

    st.subheader("1. Select Concept")
    concept_name = st.selectbox("Choose a reference concept", list(concepts.keys()))

    st.subheader("2. Whisper Model")
    model_size = st.selectbox(
        "Transcription model size",
        ["tiny", "base", "small"],
        index=1,
        help="Smaller models are faster but less accurate. 'base' is recommended for most use cases.",
    )

    st.markdown("---")
    st.subheader("3. Provide Audio")
    input_mode = st.radio(
        "How would you like to provide your explanation?",
        ["🎤 Record with microphone", "📁 Upload audio file"],
        index=0,
    )

    recorded_audio = None
    uploaded_file = None

    if input_mode == "🎤 Record with microphone":
        recorded_audio = st.audio_input("Record your spoken explanation")
    else:
        uploaded_file = st.file_uploader(
            "Upload your spoken explanation",
            type=["wav", "mp3", "m4a", "ogg"],
        )

    analyze_clicked = st.button("🔍 Analyze Explanation", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Built with Streamlit, Whisper, Sentence-BERT & Librosa")
    st.info("""
💡 Tips

• Speak clearly

• Avoid background noise

• Explain every key concept

• Keep your explanation between 30–90 seconds
""")


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("Voice-Based Concept Understanding Analyser")
st.write(
    "Speak your explanation of a concept directly into your microphone (or "
    "upload an audio file). The system will transcribe your speech, evaluate "
    "how well you understood the concept, and assess your speaking fluency."
)

# Use whichever audio source is active
audio_source = recorded_audio if recorded_audio is not None else uploaded_file

if audio_source is not None:
    st.audio(audio_source)

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if audio_source is None:
        st.error("Please record or upload audio before analyzing.")
    else:
        with st.spinner("Saving audio..."):
            # st.audio_input returns a WAV-encoded UploadedFile; uploaded files
            # carry their own extension. Default to .wav for the mic recording.
            file_name = getattr(audio_source, "name", "recording.wav")
            suffix = os.path.splitext(file_name)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_source.read())
                tmp_path = tmp.name

        try:
            with st.spinner("Transcribing speech with Whisper..."):
                transcription_result = transcribe_audio(tmp_path, model_size=model_size)
                transcribed_text = transcription_result["text"]

            if not transcribed_text.strip():
                st.error("No speech could be detected in the uploaded audio. Please try a clearer recording.")
            else:
                with st.spinner("Comparing explanation with reference concept..."):
                    reference = concepts[concept_name]
                    semantic_score = compute_semantic_similarity(
                        transcribed_text, reference["reference_explanation"]
                    )
                    coverage_result = analyze_key_point_coverage(
                        transcribed_text, reference["key_points"]
                    )

                with st.spinner("Analyzing speech fluency and audio features..."):
                    audio_features = analyze_audio(tmp_path, transcribed_text)
                    fluency_score = compute_fluency_score(audio_features)

                final_score = compute_final_score(
                    semantic_score, coverage_result["coverage_ratio"], fluency_score
                )
                final_label = get_final_label(final_score)
                feedback_list = generate_feedback(
                    semantic_score, coverage_result, audio_features, fluency_score, final_score
                )

                # Persist results in session state
                st.session_state.update(
                    analysis_done=True,
                    transcribed_text=transcribed_text,
                    semantic_score=semantic_score,
                    coverage_result=coverage_result,
                    audio_features=audio_features,
                    fluency_score=fluency_score,
                    final_score=final_score,
                    final_label=final_label,
                    feedback_list=feedback_list,
                    concept_name=concept_name,
                )
                st.success("Analysis complete!")
        finally:
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------
if st.session_state.analysis_done:
    st.markdown("---")
    st.header("📊 Evaluation Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Score", f"{st.session_state.final_score}/100")
    col2.metric("Semantic Similarity", f"{st.session_state.semantic_score}%")
    col3.metric("Key Point Coverage", f"{int(st.session_state.coverage_result['coverage_ratio']*100)}%")
    col4.metric("Fluency Score", f"{st.session_state.fluency_score}/100")

    if st.session_state.final_score >= 90:
     st.success("🏆 Excellent Performance")

    elif st.session_state.final_score >= 75:
     st.info("👍 Good Performance")

    elif st.session_state.final_score >= 50:
     st.warning("🙂 Average Performance")

    else:
     st.error("❌ Needs Improvement")

    st.markdown("### 📈 Performance Overview")

    chart_data = pd.DataFrame(
     {
        "Score": [
            st.session_state.semantic_score,
            int(st.session_state.coverage_result["coverage_ratio"] * 100),
            st.session_state.fluency_score,
            st.session_state.final_score,
        ]
     },
     index=[
        "Understanding",
        "Coverage",
        "Fluency",
        "Overall",
     ],
    )

    st.bar_chart(chart_data)

    st.markdown("### 📊 Individual Scores")

    st.write("🧠 Semantic Understanding")
    st.progress(st.session_state.semantic_score / 100)

    st.write("📚 Concept Coverage")
    st.progress(st.session_state.coverage_result["coverage_ratio"])

    st.write("🎤 Fluency")
    st.progress(st.session_state.fluency_score / 100)

    st.write("🏆 Overall")
    st.progress(st.session_state.final_score / 100)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Transcription", "Semantic Analysis", "Fluency Analysis", "Feedback"]
    )

    with tab1:
        st.subheader("Transcribed Explanation")
        with st.expander("📝 View Transcript", expanded=True):
            st.write(st.session_state.transcribed_text)

    with tab2:
        st.subheader("Concept Coverage")
        with st.expander("📚 Concept Coverage", expanded=True):
            cov = st.session_state.coverage_result
            c1, c2 = st.columns(2)
            with c1:
             st.markdown("**✅ Covered Points**")
             if cov["covered"]:
                for p in cov["covered"]:
                    st.markdown(f"- {p}")
             else:
                st.write("None")
            with c2:
             st.markdown("**❌ Missed Points**")
             if cov["missed"]:
                for p in cov["missed"]:
                    st.markdown(f"- {p}")
             else:
                st.write("None — all points covered!")

            st.progress(cov["coverage_ratio"])

    with tab3:
        st.subheader("Audio Waveform")
        st.image(st.session_state.audio_features["waveform_png"])

        st.subheader("Fluency Metrics")
        af = st.session_state.audio_features
        m1, m2, m3 = st.columns(3)
        m1.metric("Filler Words", af["fillers"]["total_fillers"])
        m2.metric("Pause Ratio", f"{round(af['pause']['pause_ratio']*100, 1)}%")
        m3.metric("Speaking Rate", f"{af['speaking_rate_wpm']} wpm")

        if af["fillers"]["filler_counts"]:
            st.markdown("**Filler word breakdown:**")
            st.bar_chart(af["fillers"]["filler_counts"])

        st.markdown("**Timing:**")
        st.write(
            f"Total duration: {af['pause']['total_duration']}s | "
            f"Speaking duration: {af['pause']['speaking_duration']}s"
        )

    with tab4:
    
     st.subheader("✅ Strengths")

     for point in st.session_state.feedback_list:
        if any(word in point.lower() for word in ["good", "well", "strong", "excellent"]):
            st.success(point)

     st.subheader("⚠ Areas for Improvement")

     for point in st.session_state.feedback_list:
        if not any(word in point.lower() for word in ["good", "well", "strong", "excellent"]):
            st.warning(point)

    # --- PDF download ---
     st.markdown("---")
     pdf_bytes = generate_pdf_report(
        concept_name=st.session_state.concept_name,
        transcribed_text=st.session_state.transcribed_text,
        semantic_score=st.session_state.semantic_score,
        coverage_result=st.session_state.coverage_result,
        audio_features=st.session_state.audio_features,
        fluency_score=st.session_state.fluency_score,
        final_score=st.session_state.final_score,
        final_label=st.session_state.final_label,
        feedback_list=st.session_state.feedback_list,
    )

     st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"VBCUA_Report_{st.session_state.concept_name.replace(' ', '_')}.pdf",
        mime="application/pdf",
        type="primary",
    )