# 🧠 Smart Review Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-VADER%20Sentiment-306998?style=for-the-badge&logo=nltk&logoColor=white)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/surajsingh4507-del/Smart-Review-Analyzer)

An enterprise-ready, multi-layered **Natural Language Processing (NLP)** and **Machine Learning sentiment intelligence platform** designed to analyze customer reviews, detect aspect-level feedback, quantify opinion subjectivity, and evaluate classification algorithms with real-time analytics.

---

## 🌟 Key Highlights & Architecture

- **🔤 Comprehensive NLP Pipeline:** Tokenization, lowercasing, regex noise cleaning, dynamic stop-word filtering, and **WordNet morphological lemmatization**.
- **📊 Rule-Based Lexicon Scoring:** NLTK **VADER (Valence Aware Dictionary and sEntiment Reasoner)** computing compound polarity with adjustable decision thresholds.
- **🔍 Opinion Mining & Subjectivity:** **TextBlob integration** separating objective factual statements from emotional opinions (0.0 to 1.0).
- **🤖 Dual Machine Learning Benchmarking:** **TF-IDF N-gram feature vectorizer** paired with **Multinomial Naive Bayes** and **Logistic Regression**, tracked via a **3x3 Confusion Matrix**.
- **🏷️ Aspect-Based Sentiment Analysis (ABSA):** Detects domain-specific nouns (*Quality*, *Shipping*, *Customer Support*, *Pricing*) and extracts keyword clouds.
- **🎙️ Voice Review Ingestion:** Integrated **HTML5 Web Speech API** enabling direct microphone voice dictation.
- **📁 Bulk Batch Processing & CSV Ingest:** Multi-line text analysis and `.csv` / `.txt` dataset uploads with aggregate percentage distribution charts.
- **💾 Local Zero-Config Relational Store:** Embedded SQLite database logging historical interactions with one-click **CSV report export**.
- **🌐 RESTful Architecture:** Fully typed endpoints powered by **FastAPI** with auto-generated interactive OpenAPI/Swagger docs (`/docs`).

---

## 🛠️ Tech Stack & Libraries

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Backend & API** | Python 3.11, FastAPI, Uvicorn, Starlette, Pydantic v2 |
| **Natural Language Processing** | NLTK (VADER, WordNet Lemmatizer, Stopwords), TextBlob |
| **Machine Learning** | Scikit-Learn (TF-IDF Vectorizer, MultinomialNB, Logistic Regression) |
| **Data Processing & Evaluation** | Pandas, NumPy, Confusion Matrix, Classification Report |
| **Database & Persistence** | SQLite3, Joblib, CSV Engine |
| **Frontend & Visualization** | HTML5, CSS3 Modern Light Theme, Chart.js, Web Speech API |
| **Testing** | Pytest Unit Test Suite |

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/surajsingh4507-del/Smart-Review-Analyzer.git
cd Smart-Review-Analyzer
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
uvicorn main:app --reload --port 8000
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000/`**

---

## 🧪 Automated Testing & Verification
The platform includes an automated unit test suite verifying tokenization, edge cases, polarity scores, and aspect tagging:
```bash
pytest test_pipeline.py
```
*Result: 5 passed in < 7 seconds with 100% verification.*

---

## 📊 Model Generalization & Benchmark

Trained and evaluated on stratified multi-class customer review benchmarks:

| Model / Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (TF-IDF)** | **High** | Balanced | Robust | Evaluated |
| **Multinomial Naive Bayes** | Robust | Fast Inference | Low Latency | Stable |

Includes a **3x3 Interactive Confusion Matrix** visualizer directly within the dashboard to inspect true vs predicted diagonal values.

---

## 📁 Repository Structure

```text
Smart-Review-Analyzer/
├── api/
│   └── index.py            # Vercel serverless entry point
├── models/
│   ├── dataset_benchmark.csv # Stratified benchmark dataset
│   ├── tfidf_vectorizer.pkl  # Trained TF-IDF vectorizer artifact
│   ├── sentiment_lr_model.pkl# Serialized Logistic Regression model
│   └── sentiment_nb_model.pkl# Serialized Naive Bayes model
├── main.py                 # FastAPI application, REST endpoints & dashboard GUI
├── nlp_engine.py           # NLP preprocessing, VADER, TextBlob & aspect pipeline
├── ml_trainer.py           # Scikit-Learn training, validation & confusion matrix
├── test_pipeline.py        # Pytest test suite
├── requirements.txt        # Production dependency specifications
├── vercel.json             # Vercel serverless deployment rewrite configuration
├── LICENSE                 # MIT Open Source License
└── README.md               # Project documentation & reference
```

---

## 🎓 Technical Reference & Concepts

1. **VADER Compound Formula:**
   $$\text{Compound Score } (c) = \frac{x}{\sqrt{x^2 + \alpha}}$$
   *(where $x$ is the sum of valence scores and $\alpha \approx 15$)*.
2. **TextBlob Subjectivity:**
   Quantifies emotional orientation from **0.0 (Objective / Factual)** to **1.0 (Subjective / Opinionated)**.
3. **Lemmatization vs Stemming:**
   Lemmatization maps words to morphological dictionary roots (*better* $\rightarrow$ *good*) rather than crudely truncating character suffixes.

---

## 📜 License
Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for more details.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/surajsingh4507-del/Smart-Review-Analyzer/issues).
If you find this project useful, consider giving it a ⭐️ on GitHub!

---
Built with ❤️ for College NLP Lab
