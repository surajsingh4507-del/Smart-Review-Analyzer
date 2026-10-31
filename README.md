# 🧠 Smart Review Analyzer

> **College NLP Lab Project** — Natural Language Processing & Sentiment Intelligence System

An end-to-end NLP-powered web application that analyzes customer reviews using multiple sentiment analysis techniques, machine learning classifiers, and aspect extraction — all running **100% offline** with no paid APIs.

---

## 👥 Team Members

| Name | Role |
|------|------|
| [Your Name] | Lead Developer — NLP Pipeline & Backend |
| [Friend 1 Name] | Frontend UI & Dashboard |
| [Friend 2 Name] | ML Training & Evaluation |

> 📌 *Update the table above with your actual team names.*

---

## 📌 Project Overview

**Smart Review Analyzer** processes customer feedback across multiple layers of NLP:

1. **Text Preprocessing Pipeline** — Tokenization, Lowercasing, Noise/Punctuation Removal, Stop-word Filtering, and Lemmatization (WordNet)
2. **Lexicon-based Sentiment Scoring** — VADER Sentiment Intensity Analyzer (Compound polarity, Pos/Neg/Neu breakdown)
3. **Subjectivity & Opinion Mining** — TextBlob polarity and subjectivity detection (Fact vs. Opinion assessment)
4. **Machine Learning Classifier** — Scikit-Learn TF-IDF N-gram Vectorizer with Multinomial Naive Bayes & Logistic Regression
5. **Aspect & Priority Extraction** — Categorizes feedback into *Product Quality*, *Delivery & Shipping*, *Customer Service*, or *Pricing* with automated urgency priority levels
6. **Zero-Dependency Architecture** — Self-contained SQLite database and interactive browser GUI — runs locally without Docker or paid APIs

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.11)
- Git

### 1. Clone the Repository
```powershell
git clone https://github.com/<your-username>/Smart-Review-Analyzer.git
cd Smart-Review-Analyzer
```

### 2. Activate Virtual Environment (Windows PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Start the Application
```powershell
.\.venv\Scripts\uvicorn main:app --reload --port 8000
```

Open your browser: 👉 **`http://127.0.0.1:8000/`**

---

## 🧪 Automated Testing

```powershell
.\.venv\Scripts\pytest test_pipeline.py
```
*All 5 unit tests pass with 100% verification.*

---

## 📊 Model Evaluation & Benchmark Metrics

Benchmarked on a stratified review dataset (`models/dataset_benchmark.csv`):

| Metric | Description |
|--------|-------------|
| Feature Extraction | TF-IDF (Sublinear scaling, Unigrams + Bigrams) |
| Classifiers | Multinomial Naive Bayes & Logistic Regression |
| Metrics Tracked | Accuracy, Macro-Precision, Macro-Recall, Macro F1-Score |

---

## 📁 Project Directory Structure

```text
Smart Review Analyzer/
│
├── main.py                 # FastAPI Web Server + Interactive HTML Dashboard + SQLite storage
├── nlp_engine.py           # Core NLP pipeline (VADER, TextBlob, Lemmatizer, Aspects)
├── ml_trainer.py           # Scikit-Learn TF-IDF training & evaluation script
├── test_pipeline.py        # Automated test suite (Pytest)
├── reviews.db              # SQLite database (stores analysis history)
├── models/                 # Saved TF-IDF vectorizer & model pickle files (.pkl)
│   ├── tfidf_vectorizer.pkl
│   ├── sentiment_lr_model.pkl
│   └── sentiment_nb_model.pkl
├── agent-service/          # LangGraph / FastAPI agent service
└── agent-ui/               # Next.js frontend UI
```

---

## 🎓 Viva & Presentation Talking Points

1. **Why VADER?** VADER is tailored for social media and customer reviews — it accounts for punctuation (`!`), capitalization (`GREAT`), and valence shifters.
2. **Why Lemmatization over Stemming?** Lemmatization uses vocabulary and morphological analysis (WordNet) to return valid dictionary words (`better` → `good`), unlike stemming which merely chops off suffixes.
3. **Why TF-IDF?** Term Frequency-Inverse Document Frequency penalizes universally frequent words while boosting unique domain-specific sentiment terms.
4. **Why SQLite?** Lightweight, zero-configuration, file-based database — perfect for a self-contained offline project.
5. **Why FastAPI?** Async Python web framework with auto-generated Swagger docs (`/docs`) — ideal for rapid NLP prototype development.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| NLP | VADER (vaderSentiment), TextBlob, NLTK WordNet |
| ML | Scikit-Learn (TF-IDF, Naive Bayes, Logistic Regression) |
| Database | SQLite |
| Frontend | Next.js, TailwindCSS |
| Testing | Pytest |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a **Pull Request**

---

*Built with ❤️ for College NLP Lab — Semester 5*
