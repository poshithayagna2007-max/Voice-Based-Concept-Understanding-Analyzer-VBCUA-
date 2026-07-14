"""
pdf_report.py
--------------
Generates a structured PDF evaluation report containing the
transcription, scores, waveform image, and feedback.
"""

import io
import tempfile
import os
from datetime import datetime
from fpdf import FPDF


def _sanitize(text: str) -> str:
    """
    Replaces common Unicode characters (smart quotes, em/en dashes, ellipses)
    that the built-in Helvetica font can't render, with safe Latin-1 equivalents.
    Prevents FPDFUnicodeEncodingException crashes from transcribed speech or
    feedback text containing these characters.
    """
    replacements = {
        "\u2014": "-",   # em dash —
        "\u2013": "-",   # en dash –
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Final fallback: drop any remaining characters outside Latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


class VBCUAReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(40, 60, 120)
        self.cell(0, 10, "VBCUA - Voice-Based Concept Understanding Report", ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, datetime.now().strftime("Generated on %B %d, %Y at %H:%M"), ln=True, align="C")
        self.ln(4)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, _sanitize(title), ln=True)
        self.set_draw_color(220, 220, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 6, _sanitize(text))
        self.ln(2)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 11)
        self.cell(60, 7, f"{_sanitize(str(key))}:")
        self.set_font("Helvetica", "", 11)
        self.cell(0, 7, _sanitize(str(value)), ln=True)


def generate_pdf_report(
    concept_name: str,
    transcribed_text: str,
    semantic_score: float,
    coverage_result: dict,
    audio_features: dict,
    fluency_score: float,
    final_score: float,
    final_label: str,
    feedback_list: list,
) -> bytes:
    """
    Builds the full PDF report and returns it as raw bytes,
    ready to be served via Streamlit's download_button.
    """
    pdf = VBCUAReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- Summary section ---
    pdf.section_title("Evaluation Summary")
    pdf.key_value("Concept", concept_name)
    pdf.key_value("Final Score", f"{final_score} / 100")
    pdf.key_value("Comprehension Level", final_label)
    pdf.key_value("Semantic Similarity", f"{semantic_score}%")
    pdf.key_value("Key Point Coverage", f"{int(coverage_result['coverage_ratio']*100)}%")
    pdf.key_value("Fluency Score", f"{fluency_score} / 100")
    pdf.ln(4)

    # --- Transcription ---
    pdf.section_title("Transcribed Explanation")
    pdf.body_text(transcribed_text if transcribed_text.strip() else "(No speech detected)")
    pdf.ln(2)

    # --- Key point coverage ---
    pdf.section_title("Key Point Coverage")
    if coverage_result["covered"]:
        pdf.body_text("Covered: " + ", ".join(coverage_result["covered"]))
    if coverage_result["missed"]:
        pdf.body_text("Missed: " + ", ".join(coverage_result["missed"]))
    pdf.ln(2)

    # --- Fluency metrics ---
    pdf.section_title("Speech Fluency Metrics")
    pdf.key_value("Total Words", audio_features["fillers"]["total_words"])
    pdf.key_value("Filler Words Used", audio_features["fillers"]["total_fillers"])
    pdf.key_value("Filler Ratio", f"{round(audio_features['fillers']['filler_ratio']*100, 2)}%")
    pdf.key_value("Pause Ratio", f"{round(audio_features['pause']['pause_ratio']*100, 2)}%")
    pdf.key_value("Speaking Duration", f"{audio_features['pause']['speaking_duration']} sec")
    pdf.key_value("Total Duration", f"{audio_features['pause']['total_duration']} sec")
    pdf.key_value("Speaking Rate", f"{audio_features['speaking_rate_wpm']} words/min")
    pdf.key_value("Mean RMS Energy", audio_features["rms"]["mean_rms"])
    pdf.ln(2)

    # --- Waveform image ---
    if audio_features.get("waveform_png"):
        pdf.section_title("Waveform Visualization")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            tmp_img.write(audio_features["waveform_png"])
            tmp_img_path = tmp_img.name
        try:
            pdf.image(tmp_img_path, x=10, w=190)
        finally:
            os.remove(tmp_img_path)
        pdf.ln(4)

    # --- Feedback ---
    pdf.add_page()
    pdf.section_title("AI-Generated Feedback")
    for point in feedback_list:
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize(f"- {point}"))
        pdf.ln(1)

    # Output as bytes
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")
    return bytes(pdf_bytes)