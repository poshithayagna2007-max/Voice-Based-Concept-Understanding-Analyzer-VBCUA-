# 🎙️ Voice-Based Concept Understanding Analyzer (VBCUA)

An AI-powered web application that evaluates a user's conceptual understanding through voice responses. The system converts speech to text, compares the response with the expected answer using semantic similarity, analyzes audio features, and generates an overall performance score with feedback.

---

## 📌 Project Overview

Traditional assessments mainly focus on written examinations and often fail to evaluate a student's verbal conceptual understanding. VBCUA addresses this limitation by allowing users to answer questions verbally. The application analyzes both the spoken content and speech characteristics to provide an intelligent evaluation.

---

## ✨ Features

- 🎤 Voice Recording
- 🔊 Speech-to-Text Conversion
- 🧠 Semantic Similarity Analysis
- 📊 Audio Feature Extraction
- 📈 Performance Scoring
- 💬 AI-Based Feedback Generation
- 📄 Downloadable PDF Report
- 🌐 Interactive Streamlit Interface
- 📉 Waveform Visualization

---

## 🛠️ Technologies Used

### Programming Language
- Python 3.11

### Frontend
- Streamlit

### Backend
- Python

### AI & Machine Learning
- OpenAI Whisper
- Sentence-BERT (SBERT)
- Scikit-learn

### Audio Processing
- Librosa
- SoundDevice
- PyAudio
- SoundFile

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Plotly

### Report Generation
- ReportLab

---

## 📂 Project Structure

```
VBCUA/
│
├── app.py
├── requirements.txt
├── README.md
├── assets/
├── models/
├── reports/
├── recordings/
├── utils/
│   ├── audio_processing.py
│   ├── evaluation.py
│   ├── report_generator.py
│   └── helper.py
└── screenshots/
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/VBCUA.git
```

### Navigate to Project Directory

```bash
cd VBCUA
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🚀 How It Works

1. Select or enter a concept/topic.
2. Record your voice response.
3. Speech is converted into text using Whisper.
4. The response is compared with the expected answer using Sentence-BERT.
5. Audio characteristics such as fluency and speech quality are analyzed.
6. An overall score is calculated.
7. Personalized feedback is generated.
8. A detailed PDF report can be downloaded.

---

## 📊 Evaluation Parameters

- Semantic Similarity
- Concept Coverage
- Fluency
- Confidence
- Speech Clarity
- Response Completeness
- Overall Performance Score

```

---

## 📋 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Enhancements

- Multi-language support
- Real-time pronunciation evaluation
- AI-powered question generation
- User login and progress tracking
- Cloud deployment
- Advanced analytics dashboard
- Interview simulation mode

---

## 👩‍💻 Author

**Puripanda Poshitha Yagna**

B.Tech – Information Technology

Andhra University College of Engineering for Women

---

## 📄 License

This project is developed for educational and academic purposes.

---

## ⭐ Acknowledgements

- OpenAI Whisper
- Sentence Transformers
- Streamlit
- Scikit-learn
- Librosa
- ReportLab
- Python Community
