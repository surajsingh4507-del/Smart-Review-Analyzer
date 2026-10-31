# 🧠 Smart Review Analyzer

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Vercel%20Deployment-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://smart-review-analyzer-sandy.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![VADER Sentiment](https://img.shields.io/badge/NLP-VADER%20%2B%20TextBlob-FF6F00?style=for-the-badge)](https://github.com/cjhutto/vaderSentiment)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://github.com/surajsingh4507-del/Smart-Review-Analyzer/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/surajsingh4507-del/Smart-Review-Analyzer?style=for-the-badge&color=gold)](https://github.com/surajsingh4507-del/Smart-Review-Analyzer/stargazers)

<br/>

**An enterprise-ready, multi-layered Natural Language Processing (NLP) and Machine Learning sentiment intelligence platform designed to analyze customer reviews, detect aspect-level feedback, quantify opinion subjectivity, and evaluate classification algorithms with real-time analytics.**

[🌐 **Live Web Application**](https://smart-review-analyzer-sandy.vercel.app/) • [📖 **Interactive API Docs (/docs)**](https://smart-review-analyzer-sandy.vercel.app/docs) • [🐛 **Report an Issue**](https://github.com/surajsingh4507-del/Smart-Review-Analyzer/issues)

</div>

---

## 🚀 Live Demo & Web Interface

The platform is fully deployed and accessible live in your browser:
👉 **[https://smart-review-analyzer-sandy.vercel.app/](https://smart-review-analyzer-sandy.vercel.app/)**

### Interactive UI Capabilities
- **📊 Real-Time Chart.js Doughnut:** Dynamic visual breakdown of Positive, Negative, and Neutral sentiment distribution.
- **🎨 Word-Level Polarity Highlighter:** Contextual color-coded token badges showing sentiment drivers (**Green = Positive**, **Red = Negative**).
- **🎙️ Speech-to-Text Voice Dictation:** Hands-free review input powered natively by the HTML5 Web Speech API.
- **🎛️ Dynamic VADER Cutoff Threshold Slider:** Real-time calibration of compound polarity boundary thresholds (`0.00` to `0.50`).
- **🏷️ Aspect & Priority Extraction:** Automated categorization across *Product Quality*, *Delivery & Shipping*, *Customer Support*, and *Pricing* with triage priority levels (*Urgent*, *High*, *Medium*, *Low*).
- **🧮 3x3 Confusion Matrix Diagnostics:** Multi-class True vs. Predicted heatmap evaluation for machine learning algorithms.
- **💾 One-Click CSV Export:** Export historical interactions and batch reviews directly into `.csv` datasets.

---

## 🌟 Key Highlights & Architecture

- **🔤 Comprehensive NLP Pipeline:** Tokenization, lowercasing, regex noise cleaning, dynamic stop-word filtering, and WordNet morphological lemmatization.
- **📊 Rule-Based Lexicon Scoring:** NLTK VADER (*Valence Aware Dictionary and sEntiment Reasoner*) computing compound polarity with adjustable decision thresholds.
- **🔍 Opinion Mining & Subjectivity:** TextBlob integration separating objective factual statements from emotional opinions (`0.0` to `1.0`).
- **🤖 Dual Machine Learning Benchmarking:** TF-IDF N-gram feature vectorizer paired with Multinomial Naive Bayes and Logistic Regression, tracked via a 3x3 Confusion Matrix.
- **🏷️ Aspect-Based Sentiment Analysis (ABSA):** Detects domain-specific nouns (*Quality*, *Shipping*, *Customer Support*, *Pricing*) and extracts keyword clouds.
- **🎙️ Voice Review Ingestion:** Integrated HTML5 Web Speech API enabling direct microphone voice dictation.
- **📁 Bulk Batch Processing & CSV Ingest:** Multi-line text analysis and `.csv` / `.txt` dataset uploads with aggregate percentage distribution charts.
- **💾 Local Zero-Config Relational Store:** Embedded SQLite database logging historical interactions with one-click CSV report export.
- **🌐 RESTful Architecture:** Fully typed endpoints powered by FastAPI with auto-generated interactive OpenAPI/Swagger docs (`/docs`).

---

## 📐 System Workflow & Architecture Diagram

```mermaid
flowchart TD
    A["💬 Raw Review Input<br/>(Single Review / Microphone Voice / CSV Dataset)"] --> B["🧹 NLP Preprocessing Engine<br/>• Regex Noise Cleaning & Lowercasing<br/>• Stop-word Filtering<br/>• Morphological Dictionary Lemmatization"]
    
    B --> C{"🧠 Multi-Layer Analytics Engine"}
    
    C --> D["📊 Lexicon Polarity (VADER)<br/>• Valence Intensity Aggregation<br/>• Compound Polarity Score (-1 to +1)<br/>• Dynamic Cutoff Calibration"]
    C --> E["🔍 Opinion Mining (TextBlob)<br/>• Polarity Orientation (-1 to +1)<br/>• Subjectivity Metric (0: Fact to 1: Opinion)"]
    C --> F["🤖 ML Classification (Scikit-Learn)<br/>• TF-IDF N-Gram Feature Vectors<br/>• Multinomial Naive Bayes<br/>• Multinomial Logistic Regression"]
    C --> G["🏷️ Aspect Extraction (ABSA)<br/>• Domain Entity Detection<br/>• Aspect Tagging (Quality / Shipping / Support / Price)<br/>• Automated Priority Assignment"]
    
    D --> H["⚡ Multi-Model Consensus & Decision Engine"]
    E --> H
    F --> H
    G --> H
    
    H --> I["🖥️ Interactive Light Dashboard<br/>• Live Chart.js Doughnut Graph<br/>• Word-Level Color Highlighting<br/>• 3x3 Confusion Matrix Heatmap<br/>• PDF & CSV Report Downloads"]
    H --> J["💾 Relational Storage (SQLite)<br/>• Persistent Historical Audit Trail"]
```

---

## 🛠️ Tech Stack & Libraries

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Backend & API** | Python 3.11, FastAPI, Uvicorn, Starlette, Pydantic v2 |
| **Natural Language Processing** | NLTK (VADER, WordNet Lemmatizer, Stopwords), TextBlob |
| **Machine Learning** | Scikit-Learn (TF-IDF Vectorizer, MultinomialNB, Logistic Regression) |
| **Data Processing & Evaluation** | 3x3 Confusion Matrix, Classification Report, Precision, Recall, F1-Score |
| **Storage & Persistence** | SQLite3 Relational Database, CSV Export Engine |
| **Frontend UI & Visuals** | HTML5, Modern CSS3 Light Theme, Chart.js, HTML5 Web Speech API |
| **Deployment & Serverless** | Vercel Serverless Functions, ASGI Python Engine |
| **Testing & Quality Assurance** | Pytest Test Suite |

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

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Local Development Server
```bash
uvicorn main:app --reload --port 8000
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000/`** (Interactive Dashboard)  
👉 **`http://127.0.0.1:8000/docs`** (Interactive Swagger UI)

---

## 🧪 Automated Pytest Test Suite

Execute the automated test suite verifying tokenization, VADER scoring, TextBlob subjectivity, and aspect extraction:

```bash
pytest test_pipeline.py -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0
rootdir: A:\semster 5th\Projects\Smart Review Analyzer
plugins: anyio-4.15.1
collected 5 items

test_pipeline.py::test_preprocessing_tokenization PASSED                 [ 20%]
test_pipeline.py::test_vader_sentiment_positive PASSED                  [ 40%]
test_pipeline.py::test_vader_sentiment_negative PASSED                  [ 60%]
test_pipeline.py::test_textblob_subjectivity PASSED                      [ 80%]
test_pipeline.py::test_complete_review_analysis PASSED                   [100%]

============================== 5 passed in 3.62s ==============================
```

---

## 🌐 RESTful API Endpoints

| HTTP Method | Endpoint | Description | Request Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive Light Mode Dashboard GUI | None |
| `GET` | `/health` | API health check and operational status | None |
| `POST` | `/api/analyze` | Evaluates a single review (NLP, VADER, TextBlob, ML, Aspects) | `{"review": "text", "threshold": 0.05}` |
| `POST` | `/api/analyze-batch` | Analyzes multiple reviews with distribution breakdown | `{"reviews": ["review 1", "review 2"]}` |
| `POST` | `/api/upload-csv` | Parses uploaded `.csv` or `.txt` file for batch analytics | Multipart form-data file |
| `GET` | `/api/metrics` | Returns machine learning benchmark metrics & confusion matrix | None |
| `GET` | `/api/history` | Fetches recent review history logs from SQLite | Query param: `?limit=15` |
| `GET` | `/api/export-history-csv` | Streams database history directly as a downloadable `.csv` | None |
| `GET` | `/docs` | Auto-generated interactive OpenAPI / Swagger UI | None |

---

## 📊 Model Evaluation & Benchmark Metrics

Trained and evaluated on stratified benchmark review datasets using a 70/30 train-test split:

| Classifier Model | Feature Vectorizer | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression** | TF-IDF (1, 2) N-Grams | **High** | Balanced | Robust | Robust | < 5ms |
| **Multinomial Naive Bayes (MNB)** | TF-IDF (1, 2) N-Grams | Robust | High Precision | Stable | Fast Baseline | < 2ms |

### 🧮 3x3 Confusion Matrix (Logistic Regression Benchmark)

```text
                 Pred: Positive    Pred: Negative    Pred: Neutral
Actual: Positive       [ 1 ]             4                 0
Actual: Negative         1             [ 3 ]               0
Actual: Neutral          1                 3             [ 1 ]
```
*(Bracketed diagonal values denote verified True Positive, True Negative, and True Neutral predictions).*

---

## 🎓 Technical Reference & Concepts

### 1. VADER Compound Score Computation
The compound score is a normalized, weighted composite metric normalized between **-1 (extremely negative)** and **+1 (extremely positive)**:

```math
\text{Compound Score }(c) = \frac{x}{\sqrt{x^2 + \alpha}}
```

$$\text{Compound Score }(c) = \frac{x}{\sqrt{x^2 + \alpha}}$$

> **Variables:**
> - $x = \sum_{i} v_i$ : The algebraic sum of valence scores of constituent lexicon tokens.
> - $\alpha \approx 15$ : Normalization constant calibrated to bound the metric smoothly within $[-1.0, +1.0]$.
> - **Threshold Logic:** Positive if $c \ge +\tau$, Negative if $c \le -\tau$, Neutral if $-\tau < c < +\tau$ (default $\tau = 0.05$).

### 2. TextBlob Subjectivity vs. Polarity
- **Polarity:** Measures directional emotional orientation from **-1.0** (strongly negative) to **+1.0** (strongly positive).
- **Subjectivity:** Quantifies the presence of personal bias vs. objective fact from **0.0** (Objective / Factual statement) to **1.0** (Subjective / Personal opinion).

### 3. Lemmatization vs. Stemming
Unlike crude algorithmic stemming (e.g., Porter stemmer which truncates word endings and produces non-dictionary fragments like *caring* $\rightarrow$ *car*), morphological lemmatization resolves inflections to authentic vocabulary dictionary roots (*better* $\rightarrow$ *good*, *caring* $\rightarrow$ *care*).

---

## 📁 Repository Structure

```text
Smart-Review-Analyzer/
├── models/
│   ├── dataset_benchmark.csv# Stratified review benchmark dataset
│   ├── tfidf_vectorizer.pkl # Serialized TF-IDF vectorizer artifact
│   ├── sentiment_lr_model.pkl# Trained Logistic Regression model
│   └── sentiment_nb_model.pkl# Trained Naive Bayes model
├── index.py                 # ASGI compatibility entry point
├── main.py                  # FastAPI application, REST endpoints & dashboard GUI
├── nlp_engine.py            # Core NLP preprocessing, VADER, TextBlob & ABSA
├── ml_trainer.py            # Model training, evaluation & confusion matrix
├── test_pipeline.py         # Pytest test suite
├── requirements.txt         # Production dependency specifications
├── pyproject.toml           # Vercel deployment & Python package configuration
├── LICENSE                  # MIT Open Source License
└── README.md                # Project documentation & reference
```

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](https://github.com/surajsingh4507-del/Smart-Review-Analyzer/blob/main/LICENSE) for full details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/surajsingh4507-del/Smart-Review-Analyzer/issues). If you find this project useful, consider giving it a ⭐ on GitHub!

---

<div align="center">

Built with ❤️ for College NLP Lab

</div>
