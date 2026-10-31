import os
import sqlite3
import json
import csv
import io
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import joblib

from nlp_engine import complete_review_analysis, clean_and_tokenize
from ml_trainer import train_and_evaluate_models

app = FastAPI(
    title="Smart Review Analyzer API",
    description="Multi-layer NLP Sentiment & Review Intelligence Engine",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SQLITE_DB_PATH", "/tmp/reviews.db" if os.environ.get("VERCEL") else os.path.join(BASE_DIR, "reviews.db"))

def init_sqlite_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                categories TEXT NOT NULL,
                priority TEXT NOT NULL,
                vader_compound REAL NOT NULL,
                subjectivity REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"SQLite Init Warning: {e}")

init_sqlite_db()

MODELS_DIR = os.path.join(BASE_DIR, "models")
try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_lr_model.pkl"))
    nb_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_nb_model.pkl"))
except Exception:
    vectorizer, lr_model, nb_model = None, None, None

class SingleReviewRequest(BaseModel):
    review: str
    threshold: Optional[float] = 0.05

class BatchReviewRequest(BaseModel):
    reviews: List[str]
    threshold: Optional[float] = 0.05

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Smart Review Analyzer",
        "version": "3.0.0",
        "nlp_engine": "VADER + TextBlob + TF-IDF (N-gram) + Word Tokenizer",
        "database": "SQLite"
    }

@app.get("/api/docs", include_in_schema=False)
def api_docs_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.post("/api/analyze")
@app.post("/analyze")
def analyze_single_review(payload: SingleReviewRequest):
    text = payload.review.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    threshold = payload.threshold if payload.threshold is not None else 0.05
    result = complete_review_analysis(text, threshold)
    
    ml_predictions = {}
    if vectorizer and lr_model and nb_model:
        try:
            vec = vectorizer.transform([text])
            lr_pred = lr_model.predict(vec)[0]
            nb_pred = nb_model.predict(vec)[0]
            ml_predictions = {
                "logistic_regression": lr_pred,
                "naive_bayes": nb_pred
            }
        except Exception as e:
            ml_predictions = {"error": str(e)}
            
    result["ml_model_predictions"] = ml_predictions
    
    vader_sent = result["sentiment"]
    lr_sent = ml_predictions.get("logistic_regression")
    tb_sent = result["textblob_metrics"].get("tb_sentiment")
    
    disagreements = []
    if lr_sent and lr_sent != vader_sent:
        disagreements.append(f"VADER ({vader_sent}) vs Logistic Regression ({lr_sent})")
    if tb_sent and tb_sent != vader_sent:
        disagreements.append(f"VADER ({vader_sent}) vs TextBlob ({tb_sent})")
        
    result["model_comparison"] = {
        "vader": vader_sent,
        "logistic_regression": lr_sent or "N/A",
        "textblob": tb_sent or "N/A",
        "has_disagreement": len(disagreements) > 0,
        "disagreement_details": disagreements
    }
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO reviews_history (review, sentiment, confidence, categories, priority, vader_compound, subjectivity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            text,
            result["sentiment"],
            result["confidence_percentage"],
            ", ".join(result["categories"]),
            result["priority"],
            result["vader_scores"]["compound_score"],
            result["textblob_metrics"]["subjectivity"]
        ))
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"DB Error: {db_err}")
        
    return result

@app.post("/api/analyze-batch")
@app.post("/analyze-batch")
def analyze_batch_reviews(payload: BatchReviewRequest):
    valid_reviews = [r.strip() for r in payload.reviews if r.strip()]
    if not valid_reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")
        
    threshold = payload.threshold if payload.threshold is not None else 0.05
    results = [complete_review_analysis(r, threshold) for r in valid_reviews]
    
    total = len(results)
    pos_count = sum(1 for r in results if r["sentiment"] == "Positive")
    neg_count = sum(1 for r in results if r["sentiment"] == "Negative")
    neu_count = sum(1 for r in results if r["sentiment"] == "Neutral")
    
    return {
        "total_reviews": total,
        "sentiment_distribution": {
            "Positive": pos_count,
            "Negative": neg_count,
            "Neutral": neu_count,
            "Positive_Percentage": round((pos_count / total) * 100, 1),
            "Negative_Percentage": round((neg_count / total) * 100, 1),
            "Neutral_Percentage": round((neu_count / total) * 100, 1),
        },
        "detailed_results": results
    }

@app.post("/api/upload-csv")
@app.post("/upload-csv")
async def upload_csv_batch(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        text = contents.decode('utf-8', errors='ignore')
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded file.")
        
    lines = []
    if file.filename.endswith('.csv'):
        reader = csv.reader(io.StringIO(text))
        for row in reader:
            if row and row[0].strip() and not row[0].lower().startswith("review"):
                lines.append(row[0].strip())
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
    if not lines:
        raise HTTPException(status_code=400, detail="No readable reviews found in file.")
        
    return analyze_batch_reviews(BatchReviewRequest(reviews=lines[:100]))

@app.get("/api/export-history-csv")
@app.get("/export-history-csv")
def export_history_csv():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, review, sentiment, confidence, categories, priority, vader_compound, subjectivity, created_at FROM reviews_history ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Review", "Sentiment", "Confidence (%)", "Categories", "Priority", "VADER Compound", "Subjectivity", "Timestamp"])
    for row in rows:
        writer.writerow(row)
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=review_analysis_history.csv"}
    )

@app.get("/api/metrics")
@app.get("/metrics")
def get_evaluation_metrics():
    return train_and_evaluate_models()

@app.get("/api/history")
@app.get("/history")
def get_analysis_history(limit: int = 15):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM reviews_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return {"count": len(rows), "history": rows}

@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def get_interactive_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Review Analyzer - AI & NLP Intelligence Platform</title>
    <!-- Chart.js for Visual Graphs -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --primary-light: #e0e7ff;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --card-border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --accent-pos: #059669;
            --accent-neg: #dc2626;
            --accent-neu: #d97706;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }

        .container {
            max-width: 1220px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 20px;
        }

        .header-title h1 {
            font-size: 2.1rem;
            color: #312e81;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .header-title p {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 4px;
            font-weight: 500;
        }

        .header-actions {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        .btn-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #ffffff;
            color: #4f46e5;
            text-decoration: none;
            padding: 9px 15px;
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            font-size: 13.5px;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .btn-link:hover {
            background: #f1f5f9;
            border-color: #94a3b8;
        }

        /* Clean System Architecture Bar */
        .info-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #4f46e5;
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }

        .info-card-left {
            display: flex;
            gap: 18px;
            align-items: center;
            font-size: 13.5px;
            color: #334155;
            font-weight: 500;
        }

        .info-badge {
            background: #e0e7ff;
            color: #3730a3;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 12px;
            border: 1px solid #c7d2fe;
        }

        .tab-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 16px;
            border-bottom: 2px solid var(--card-border);
            padding-bottom: 4px;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 15px;
            font-weight: 600;
            padding: 8px 18px;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s;
        }

        .tab-btn.active {
            color: #4f46e5;
            background: #ffffff;
            border: 1px solid var(--card-border);
            border-bottom: 2px solid #ffffff;
            margin-bottom: -6px;
            font-weight: 700;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1.05fr 1fr;
            gap: 24px;
        }

        @media (max-width: 900px) {
            .grid-2 { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            margin-bottom: 24px;
        }

        .card h2 {
            font-size: 1.25rem;
            color: #1e293b;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        label {
            font-size: 13px;
            color: #475569;
            font-weight: 600;
            display: block;
            margin-bottom: 6px;
        }

        select, textarea {
            width: 100%;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            color: #0f172a;
            padding: 12px;
            font-size: 14.5px;
            margin-bottom: 14px;
            outline: none;
            transition: all 0.2s;
        }

        select:focus, textarea:focus {
            background: #ffffff;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
        }

        textarea {
            height: 125px;
            resize: vertical;
        }

        .controls-row {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .slider-group {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }

        .slider-group input[type="range"] {
            flex: 1;
            accent-color: var(--primary);
            cursor: pointer;
        }

        .btn-primary {
            width: 100%;
            background: var(--primary);
            color: #fff;
            border: none;
            padding: 13px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2);
            transition: all 0.2s;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
        }

        .btn-secondary {
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-secondary:hover {
            background: #e2e8f0;
            color: #0f172a;
        }

        .btn-mic {
            background: #ffffff;
            color: #4f46e5;
            border: 1px solid #c7d2fe;
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }
        .btn-mic:hover {
            background: #e0e7ff;
        }
        .btn-mic.listening {
            background: #fee2e2;
            color: #dc2626;
            border-color: #fca5a5;
            animation: pulse 1.2s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-pos { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
        .badge-neg { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
        .badge-neu { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }

        .word-highlight-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px;
            font-size: 15px;
            line-height: 2.1;
            margin-top: 8px;
            color: #1e293b;
        }

        .word-pos {
            background: #dcfce7;
            color: #166534;
            padding: 3px 8px;
            border-radius: 4px;
            border-bottom: 2px solid #22c55e;
            font-weight: 700;
        }

        .word-neg {
            background: #fee2e2;
            color: #991b1b;
            padding: 3px 8px;
            border-radius: 4px;
            border-bottom: 2px solid #ef4444;
            font-weight: 700;
        }

        .word-stop {
            opacity: 0.45;
            text-decoration: line-through;
            font-style: italic;
        }

        .metric-tile {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
        }

        .metric-title {
            font-size: 11.5px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 17px;
            font-weight: 800;
            margin-top: 4px;
            color: #0f172a;
        }

        .tag {
            background: #eef2ff;
            color: #4338ca;
            padding: 3px 9px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid #c7d2fe;
            display: inline-block;
            margin-right: 5px;
        }

        .recommendation-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 14px;
            font-size: 13.5px;
            color: #92400e;
            font-weight: 600;
        }

        .disagreement-box {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-left: 4px solid #ef4444;
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 10px;
            font-size: 13px;
            color: #991b1b;
            font-weight: 600;
        }

        .bubble-cloud {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        .bubble-tag {
            background: #e0f2fe;
            color: #0369a1;
            border: 1px solid #bae6fd;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12.5px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            overflow-x: auto;
            padding: 14px 4px;
            margin-top: 6px;
        }

        .pipeline-step {
            flex: 1;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 10px;
            text-align: center;
            min-width: 130px;
            transition: transform 0.2s;
        }
        .pipeline-step:hover {
            transform: translateY(-2px);
            border-color: #cbd5e1;
        }

        .pipeline-step h4 {
            font-size: 13px;
            color: #3730a3;
            margin-bottom: 4px;
            font-weight: 700;
        }

        .pipeline-step p {
            font-size: 11.5px;
            color: var(--text-muted);
        }

        .pipeline-arrow {
            color: #94a3b8;
            font-weight: bold;
            font-size: 18px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
            margin-top: 10px;
        }

        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }

        th {
            background: #f8fafc;
            color: #475569;
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 700;
        }

        tbody tr:hover {
            background: #f8fafc;
        }

        .cm-table {
            width: auto;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 10px;
        }
        .cm-table th, .cm-table td {
            text-align: center;
            padding: 12px 18px;
            font-weight: 700;
        }
        .cm-diag {
            background: #dcfce7 !important;
            color: #15803d;
            font-size: 16px;
        }
        .cm-off {
            background: #fee2e2 !important;
            color: #b91c1c;
            font-size: 15px;
        }

        /* Documentation Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 23, 42, 0.5);
            backdrop-filter: blur(4px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .modal-box {
            background: #ffffff;
            border-radius: 14px;
            max-width: 750px;
            width: 100%;
            max-height: 85vh;
            overflow-y: auto;
            padding: 28px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            position: relative;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 14px;
            margin-bottom: 16px;
        }
        .modal-header h3 {
            font-size: 1.3rem;
            color: #1e293b;
        }
        .close-modal-btn {
            background: #f1f5f9;
            border: none;
            font-size: 18px;
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }
        .close-modal-btn:hover { background: #e2e8f0; }

        @media print {
            body { background: #fff !important; color: #000 !important; padding: 0; }
            .btn-primary, .btn-secondary, .btn-link, select, .tab-bar, .header-actions, #batchModeSection, .controls-row, .modal-overlay {
                display: none !important;
            }
            .card { border: 1px solid #cbd5e1 !important; box-shadow: none !important; page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="header-title">
                <h1>🧠 Smart Review Analyzer <span style="font-size: 13px; background: #e0e7ff; color: #4338ca; padding: 4px 8px; border-radius: 6px; font-weight: 700;">v3.0</span></h1>
                <p>Natural Language Processing & Sentiment Intelligence Platform</p>
            </div>
            <div class="header-actions">
                <button onclick="openDocModal()" class="btn-link" title="Open Technical Reference Documentation">
                    📖 Technical Reference
                </button>
                <a href="/docs" target="_blank" class="btn-link" title="Open Interactive Swagger REST API Docs">
                    🌐 API Docs (/docs)
                </a>
                <a href="/api/export-history-csv" class="btn-link" title="Download Entire History in CSV format">
                    📥 Download CSV
                </a>
                <button onclick="window.print()" class="btn-link" title="Export Current Analysis Report to PDF">
                    🖨️ Export PDF
                </button>
            </div>
        </header>

        <!-- System Architecture Bar -->
        <div class="info-card">
            <div class="info-card-left">
                <div>⚡ <strong>System Architecture:</strong> Hybrid Rule-based + ML Vectorizer Pipeline</div>
                <div>💾 <strong>Database:</strong> SQLite Relational Store</div>
            </div>
            <div>
                <span class="info-badge">NLTK VADER</span>
                <span class="info-badge">TextBlob</span>
                <span class="info-badge">TF-IDF + Scikit-Learn</span>
                <span class="info-badge">SQLite</span>
            </div>
        </div>

        <!-- Mode Tabs -->
        <div class="tab-bar">
            <button class="tab-btn active" id="singleTabBtn" onclick="switchTab('single')">📝 Single Review Analysis</button>
            <button class="tab-btn" id="batchTabBtn" onclick="switchTab('batch')">📋 Batch & CSV Bulk Processor</button>
        </div>

        <!-- Single Review Mode -->
        <div id="singleModeSection">
            <div class="grid-2">
                <!-- Input Column -->
                <div class="card">
                    <h2>
                        <span>✍️ Review Input</span>
                        <button class="btn-mic" id="micBtn" onclick="toggleSpeechRecognition()" title="Voice Review Input via Microphone">
                            🎙️ <span id="micText">Speak Review</span>
                        </button>
                    </h2>
                    
                    <label for="sampleDropdown">📌 Quick Test Preset (Select Any Domain):</label>
                    <select id="sampleDropdown" onchange="handleSampleSelect(this.value)">
                        <option value="">-- Choose a pre-written test review --</option>
                        <option value="tech_pos">📱 Smart Phone (Positive) - Battery & Screen</option>
                        <option value="tech_neg">💻 Laptop Overheating (Negative Defect)</option>
                        <option value="food_pos">🍕 Restaurant Delivery (Positive Experience)</option>
                        <option value="ecom_neg">📦 E-Commerce Shipping Delay (Complaint)</option>
                        <option value="spec_neu">📏 Neutral Specification & Dimensions</option>
                        <option value="support_neg">📞 Customer Support (Negative Rude Behavior)</option>
                    </select>

                    <label for="reviewInput">Customer Review Text:</label>
                    <textarea id="reviewInput" placeholder="Type, paste, or click 'Speak Review' to talk... e.g., 'The sound quality is exceptional and battery lasts 20 hours! Highly recommended.'"></textarea>

                    <!-- Interactive Controls: Threshold Slider & Stopwords Filter -->
                    <div class="controls-row">
                        <div class="slider-group">
                            <label style="margin:0; font-size:12.5px;">🎚️ VADER Threshold Cutoff: <strong id="thresholdDisplay" style="color:var(--primary);">+0.05</strong></label>
                            <input type="range" id="thresholdSlider" min="0.0" max="0.5" step="0.05" value="0.05" oninput="updateThreshold(this.value)">
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <label style="margin:0; font-size:12.5px; display:flex; align-items:center; gap:6px; cursor:pointer;">
                                <input type="checkbox" id="stopwordToggle" onchange="toggleStopwordsDisplay(this.checked)" checked>
                                🔤 Dim / Cross-out Filtered Stop-Words
                            </label>
                            <span style="font-size:11.5px; color:var(--text-muted);">(Real-time Preprocessing Control)</span>
                        </div>
                    </div>

                    <button class="btn-primary" id="analyzeBtn" onclick="triggerAnalyze()">
                        <div class="spinner" id="analyzeSpinner"></div>
                        <span id="analyzeBtnText">⚡ Analyze Review with NLP Pipeline</span>
                    </button>
                </div>

                <!-- Diagnostics Column -->
                <div class="card" id="resultsCard">
                    <h2>
                        <span>📊 Sentiment Diagnostics</span>
                        <span id="statusIndicator" style="font-size: 12px; color: var(--text-muted); font-weight: 600;">Ready</span>
                    </h2>

                    <div id="emptyPlaceholder" style="text-align: center; padding: 45px 0; color: var(--text-muted);">
                        <div style="font-size: 34px; margin-bottom: 8px;">🔍</div>
                        <p style="font-size: 14.5px;">Select a sample review or speak, then click <strong>Analyze</strong>.</p>
                    </div>

                    <div id="resultContent" style="display: none;">
                        <!-- Top Summary Metric Badges -->
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                            <div>
                                <div class="metric-title">Primary Sentiment</div>
                                <div id="sentimentBadgeContainer" style="margin-top: 4px;"></div>
                            </div>
                            <div>
                                <div class="metric-title">Model Confidence</div>
                                <div id="confidenceVal" style="font-size: 1.4rem; font-weight: 800; color: #4f46e5;">0%</div>
                            </div>
                            <div>
                                <div class="metric-title">Urgency Priority</div>
                                <div id="priorityVal" style="font-size: 1.1rem; font-weight: 700;">Normal</div>
                            </div>
                        </div>

                        <!-- Side-by-Side Model Comparison -->
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin-bottom:12px; font-size:12.5px;">
                            <div class="metric-title" style="margin-bottom:4px;">⚖️ Multi-Model Consensus (Comparison):</div>
                            <div style="display:flex; justify-content:space-between; font-weight:700;">
                                <span>VADER: <span id="cmpVader" style="color:#4f46e5;">-</span></span>
                                <span>LogReg (TF-IDF): <span id="cmpLR" style="color:#059669;">-</span></span>
                                <span>TextBlob: <span id="cmpTB" style="color:#d97706;">-</span></span>
                            </div>
                            <div id="disagreementAlert" style="display:none;" class="disagreement-box"></div>
                        </div>

                        <!-- Automated Business Action Recommendation -->
                        <div id="recommendationAlert" class="recommendation-box"></div>

                        <!-- Color-coded Word Highlighting -->
                        <div style="margin-top:12px;">
                            <label>🎨 Lexicon Word-Level Polarity Highlighting:</label>
                            <div class="word-highlight-box" id="highlightedWordsBox"></div>
                            <div style="display: flex; gap: 14px; font-size: 11.5px; margin-top: 6px; color: var(--text-muted); font-weight: 600;">
                                <span><span style="color: #166534; font-weight: 900;">■</span> Green: Positive</span>
                                <span><span style="color: #991b1b; font-weight: 900;">■</span> Red: Negative</span>
                                <span><span style="opacity: 0.5; font-style: italic;">Strikethrough</span>: Filtered Stopword</span>
                            </div>
                        </div>

                        <!-- Live Sentiment Doughnut Chart -->
                        <div style="margin-top: 14px; display: grid; grid-template-columns: 140px 1fr; gap: 14px; align-items: center;">
                            <div style="height: 140px; width: 140px; position: relative;">
                                <canvas id="sentimentPieChart"></canvas>
                            </div>
                            <div style="display: flex; flex-direction: column; gap: 8px;">
                                <div class="metric-tile">
                                    <div class="metric-title">VADER Compound Score</div>
                                    <div class="metric-value" id="compoundVal">0.000</div>
                                </div>
                                <div class="metric-tile">
                                    <div class="metric-title">TextBlob Subjectivity (Fact vs Opinion)</div>
                                    <div class="metric-value" id="subjectivityVal" style="font-size: 13.5px;">0.00 (Fact)</div>
                                </div>
                            </div>
                        </div>

                        <!-- Aspect Keywords Bubble Cloud -->
                        <div style="margin-top: 14px;">
                            <div class="metric-title">☁️ Extracted Aspect Keywords (ABSA):</div>
                            <div class="bubble-cloud" id="bubbleCloudContainer"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Batch & File Upload Mode Section -->
        <div id="batchModeSection" style="display: none;">
            <div class="card">
                <h2>📋 Bulk / Batch Reviews Processor & CSV Upload</h2>
                
                <div style="display: flex; gap: 14px; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
                    <div style="flex: 1;">
                        <label>📁 Option A: Upload .CSV or .TXT Review File (Max 100 rows):</label>
                        <input type="file" id="csvFileInput" accept=".csv, .txt" style="padding: 8px; font-size: 13px;">
                    </div>
                    <div>
                        <button class="btn-secondary" style="padding: 10px 16px; margin-top: 18px;" onclick="uploadFileBatch()">
                            ⬆️ Upload & Process File
                        </button>
                    </div>
                </div>

                <label>✍️ Option B: Paste Multiple Reviews (One per line):</label>
                <textarea id="batchInput" style="height: 110px;" placeholder="The phone display is stunning and battery is awesome!&#10;Worst delivery ever, package arrived broken.&#10;Normal item, matches specs."></textarea>
                
                <button class="btn-primary" onclick="triggerBatchAnalyze()">
                    <div class="spinner" id="batchSpinner"></div>
                    <span>⚡ Process All Reviews in Bulk</span>
                </button>

                <div id="batchResultsContainer" style="display: none; margin-top: 20px;">
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">
                        <div class="metric-tile">
                            <div class="metric-title">Total Processed</div>
                            <div class="metric-value" id="batchTotalCount">0</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-title" style="color: var(--accent-pos);">Positive Count</div>
                            <div class="metric-value" id="batchPosCount" style="color: var(--accent-pos);">0</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-title" style="color: var(--accent-neg);">Negative Count</div>
                            <div class="metric-value" id="batchNegCount" style="color: var(--accent-neg);">0</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-title" style="color: var(--accent-neu);">Neutral Count</div>
                            <div class="metric-value" id="batchNeuCount" style="color: var(--accent-neu);">0</div>
                        </div>
                    </div>
                    <div style="overflow-x: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Review Snippet</th>
                                    <th>Sentiment</th>
                                    <th>Confidence</th>
                                    <th>VADER Compound</th>
                                    <th>Priority</th>
                                </tr>
                            </thead>
                            <tbody id="batchTableBody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- 6-Stage NLP Pipeline Flow -->
        <div class="card">
            <h2>🧪 System Architecture: The 6-Stage NLP Pipeline</h2>
            <div class="pipeline-container">
                <div class="pipeline-step">
                    <h4>1. Raw Input</h4>
                    <p>User voice or text review string</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>2. Clean & Tokenize</h4>
                    <p>Lowercasing, Regex & Noise removal</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>3. Lemmatize</h4>
                    <p>WordNet morphological root forms</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>4. VADER & Lexicon</h4>
                    <p>Word polarity & valence scoring</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>5. TextBlob & TF-IDF</h4>
                    <p>Subjectivity & ML classification</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>6. SQLite & Dashboard</h4>
                    <p>Historical logging & Action items</p>
                </div>
            </div>
        </div>

        <!-- History Trend Graph & Log -->
        <div class="card">
            <h2>
                <span>📈 Historical Sentiment Trends (Recent Analyses)</span>
                <div style="display:flex; gap:10px;">
                    <a href="/api/export-history-csv" class="btn-secondary" style="text-decoration:none; display:inline-flex; align-items:center; gap:4px;">
                        📥 Export to CSV
                    </a>
                    <button class="btn-secondary" onclick="loadHistory()">🔄 Refresh History</button>
                </div>
            </h2>
            <div style="height: 190px; width: 100%; margin-bottom: 20px;">
                <canvas id="trendLineChart"></canvas>
            </div>

            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Review Snippet</th>
                            <th>Sentiment</th>
                            <th>Confidence</th>
                            <th>Compound</th>
                            <th>Priority</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody id="historyTableBody">
                        <tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Loading database records...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Confusion Matrix & ML Benchmark -->
        <div class="card">
            <h2>🎓 Scikit-Learn TF-IDF Benchmark & Confusion Matrix</h2>
            <p style="color:var(--text-muted); font-size:13.5px; margin-bottom:14px;">
                Demonstrates machine learning generalization metrics alongside a 3x3 classification confusion matrix.
            </p>
            <div style="overflow-x: auto; margin-bottom: 20px;">
                <table>
                    <thead>
                        <tr>
                            <th>Model / Classifier</th>
                            <th>Accuracy</th>
                            <th>Macro Precision</th>
                            <th>Macro Recall</th>
                            <th>Macro F1-Score</th>
                        </tr>
                    </thead>
                    <tbody id="metricsTableBody">
                        <tr><td colspan="5" style="text-align:center;">Loading model metrics...</td></tr>
                    </tbody>
                </table>
            </div>

            <h3 style="font-size:15px; color:#1e293b; margin-top:16px;">🧮 Interactive Confusion Matrix (Logistic Regression on Test Split):</h3>
            <div style="display:flex; justify-content:center; margin-top:10px;">
                <table class="cm-table" id="confusionMatrixTable">
                    <thead>
                        <tr>
                            <th>Actual \\ Predicted</th>
                            <th>Pred: Positive</th>
                            <th>Pred: Negative</th>
                            <th>Pred: Neutral</th>
                        </tr>
                    </thead>
                    <tbody id="cmBody"></tbody>
                </table>
            </div>
            <div style="text-align:center; font-size:12px; color:var(--text-muted); margin-top:6px;">
                Green cells = True Positives / Negatives / Neutrals (Diagonal) | Red cells = Classification Errors
            </div>
        </div>
    </div>

    <!-- Technical Reference Modal -->
    <div class="modal-overlay" id="docModal">
        <div class="modal-box">
            <div class="modal-header">
                <h3>📖 System Technical Reference</h3>
                <button class="close-modal-btn" onclick="closeDocModal()">✕</button>
            </div>
            <div style="font-size:14px; line-height:1.7; color:#334155;">
                <h4 style="color:#4f46e5; margin-bottom:4px;">1. VADER Compound Score Computation</h4>
                <p>The compound score is a normalized metric between <strong>-1 (extremely negative)</strong> and <strong>+1 (extremely positive)</strong>.</p>
                <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:10px 14px; margin:8px 0; font-family:'Courier New', monospace; font-size:14px; font-weight:600; color:#1e293b;">
                    Compound Score (c) = x / √(x² + α)
                </div>
                <p style="font-size:13px; color:#64748b; margin-top:2px;">Where: <strong>x</strong> = sum of valence scores of lexicon tokens, and <strong>α (alpha)</strong> = normalization parameter (default ≈ 15).</p>
                <br>
                <h4 style="color:#4f46e5; margin-bottom:4px;">2. TextBlob Polarity vs Subjectivity</h4>
                <p><strong>Polarity:</strong> Quantifies emotional direction (-1.0 to +1.0).<br>
                <strong>Subjectivity:</strong> Distinguishes objective factual statements from opinionated expressions (0.0 = Objective Fact, 1.0 = Subjective Emotion).</p>
                <br>
                <h4 style="color:#4f46e5; margin-bottom:4px;">3. Morphological Lemmatization (WordNet)</h4>
                <p>Unlike rule-based stemming (which crudely truncates suffixes), lemmatization references the WordNet lexical database to resolve inflected words back to their genuine grammatical roots (e.g., <em>caring -> care, better -> good</em>).</p>
                <br>
                <h4 style="color:#4f46e5; margin-bottom:4px;">4. Multi-class Confusion Matrix</h4>
                <p>Maps real ground-truth labels against predicted outputs across a 3x3 grid. The diagonal represents true correct classifications, while off-diagonal elements isolate specific model confusion patterns.</p>
            </div>
        </div>
    </div>

    <script>
        let pieChartInstance = null;
        let trendChartInstance = null;
        let speechRecognition = null;
        let isListening = false;
        let currentTokens = [];

        const PRESETS = {
            "tech_pos": "The battery life is phenomenal and the screen resolution is gorgeous! Best gadget I bought this entire year.",
            "tech_neg": "Do not buy this laptop! It overheats within 15 minutes of usage, constantly crashes, and has cheap plastic build.",
            "food_pos": "Outstanding food quality, delivery arrived in just 20 minutes hot and fresh. Loved the customer service!",
            "ecom_neg": "Horrible delivery delay! Package took 3 weeks to arrive and arrived completely broken inside with missing items.",
            "spec_neu": "The item dimensions are 10cm by 15cm. Package includes one standard USB-C cable and user manual.",
            "support_neg": "Customer support is completely unresponsive and extremely rude. Refused to issue a refund for defective unit."
        };

        function handleSampleSelect(val) {
            if (val && PRESETS[val]) {
                document.getElementById('reviewInput').value = PRESETS[val];
            }
        }

        function switchTab(mode) {
            const singleSec = document.getElementById('singleModeSection');
            const batchSec = document.getElementById('batchModeSection');
            const singleBtn = document.getElementById('singleTabBtn');
            const batchBtn = document.getElementById('batchTabBtn');

            if (mode === 'single') {
                singleSec.style.display = 'block';
                batchSec.style.display = 'none';
                singleBtn.classList.add('active');
                batchBtn.classList.remove('active');
            } else {
                singleSec.style.display = 'none';
                batchSec.style.display = 'block';
                singleBtn.classList.remove('active');
                batchBtn.classList.add('active');
            }
        }

        function updateThreshold(val) {
            document.getElementById('thresholdDisplay').innerText = (val >= 0 ? '+' : '') + parseFloat(val).toFixed(2);
        }

        function toggleStopwordsDisplay(checked) {
            renderWordHighlights(checked);
        }

        function openDocModal() {
            document.getElementById('docModal').style.display = 'flex';
        }

        function closeDocModal() {
            document.getElementById('docModal').style.display = 'none';
        }

        function toggleSpeechRecognition() {
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRec) {
                alert('Speech recognition is not supported in this browser. Please use Google Chrome or Edge.');
                return;
            }

            const micBtn = document.getElementById('micBtn');
            const micText = document.getElementById('micText');

            if (!speechRecognition) {
                speechRecognition = new SpeechRec();
                speechRecognition.continuous = false;
                speechRecognition.interimResults = false;
                speechRecognition.lang = 'en-US';

                speechRecognition.onstart = () => {
                    isListening = true;
                    micBtn.classList.add('listening');
                    micText.innerText = 'Listening...';
                };

                speechRecognition.onresult = (e) => {
                    const transcript = e.results[0][0].transcript;
                    document.getElementById('reviewInput').value = transcript;
                };

                speechRecognition.onerror = (e) => {
                    console.error('Speech error:', e);
                    stopListening();
                };

                speechRecognition.onend = () => {
                    stopListening();
                };
            }

            if (!isListening) {
                speechRecognition.start();
            } else {
                speechRecognition.stop();
            }
        }

        function stopListening() {
            isListening = false;
            const micBtn = document.getElementById('micBtn');
            const micText = document.getElementById('micText');
            micBtn.classList.remove('listening');
            micText.innerText = 'Speak Review';
        }

        function setLoading(isLoading) {
            const spinner = document.getElementById('analyzeSpinner');
            const btnText = document.getElementById('analyzeBtnText');
            const btn = document.getElementById('analyzeBtn');

            if (isLoading) {
                spinner.style.display = 'inline-block';
                btnText.innerText = 'Running NLP Analysis...';
                btn.disabled = true;
            } else {
                spinner.style.display = 'none';
                btnText.innerText = '⚡ Analyze Review with NLP Pipeline';
                btn.disabled = false;
            }
        }

        async function triggerAnalyze() {
            const text = document.getElementById('reviewInput').value.trim();
            if (!text) {
                alert('Please enter or select a review to analyze!');
                return;
            }

            const threshold = parseFloat(document.getElementById('thresholdSlider').value);
            setLoading(true);

            try {
                await new Promise(r => setTimeout(r, 180));

                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ review: text, threshold: threshold })
                });

                if (!res.ok) throw new Error('Failed to analyze');
                const data = await res.json();

                renderSingleResults(data);
                loadHistory();
            } catch (err) {
                alert('Analysis Error: ' + err.message);
            } finally {
                setLoading(false);
            }
        }

        function renderSingleResults(data) {
            document.getElementById('emptyPlaceholder').style.display = 'none';
            document.getElementById('resultContent').style.display = 'block';
            document.getElementById('statusIndicator').innerText = 'Analysis Complete';

            // Sentiment Badge
            const badgeContainer = document.getElementById('sentimentBadgeContainer');
            let badgeClass = 'badge-neu';
            if (data.sentiment === 'Positive') badgeClass = 'badge-pos';
            if (data.sentiment === 'Negative') badgeClass = 'badge-neg';
            badgeContainer.innerHTML = `<span class="badge ${badgeClass}">${data.sentiment}</span>`;

            // Confidence & Priority
            document.getElementById('confidenceVal').innerText = data.confidence_percentage + '%';
            const prio = document.getElementById('priorityVal');
            prio.innerText = data.priority;
            prio.style.color = data.priority === 'High' ? '#dc2626' : (data.priority === 'Medium' ? '#d97706' : '#059669');

            // Model Comparison
            const mc = data.model_comparison;
            document.getElementById('cmpVader').innerText = mc.vader;
            document.getElementById('cmpLR').innerText = mc.logistic_regression;
            document.getElementById('cmpTB').innerText = mc.textblob;

            const disAlert = document.getElementById('disagreementAlert');
            if (mc.has_disagreement) {
                disAlert.style.display = 'block';
                disAlert.innerHTML = `⚠️ <strong>Model Disagreement:</strong> ${mc.disagreement_details.join(' | ')}`;
            } else {
                disAlert.style.display = 'none';
            }

            // Recommendation
            const recAlert = document.getElementById('recommendationAlert');
            recAlert.innerHTML = `<strong>System Recommendation:</strong> ${data.recommendation}`;

            // Aspect Cloud
            const cloudContainer = document.getElementById('bubbleCloudContainer');
            if (data.aspect_keywords && data.aspect_keywords.length) {
                cloudContainer.innerHTML = data.aspect_keywords.map(a => `
                    <div class="bubble-tag">
                        <span>🏷️ ${a.keyword}</span>
                        <span style="font-size:10px; background:#bae6fd; padding:1px 6px; border-radius:10px;">${a.aspect}</span>
                    </div>
                `).join('');
            } else {
                cloudContainer.innerHTML = '<span style="font-size:13px; color:var(--text-muted);">No domain-specific aspect terms identified.</span>';
            }

            currentTokens = data.word_tokens || [];
            renderWordHighlights(document.getElementById('stopwordToggle').checked);

            document.getElementById('compoundVal').innerText = data.vader_scores.compound_score;
            document.getElementById('subjectivityVal').innerText = `${data.textblob_metrics.subjectivity} (${data.textblob_metrics.assessment})`;

            renderSentimentDoughnut(
                data.vader_scores.positive_score,
                data.vader_scores.negative_score,
                data.vader_scores.neutral_score
            );
        }

        function renderWordHighlights(dimStopwords) {
            const highlightBox = document.getElementById('highlightedWordsBox');
            if (!currentTokens.length) return;

            highlightBox.innerHTML = currentTokens.map(item => {
                let stopClass = (dimStopwords && item.is_stopword) ? 'word-stop' : '';
                if (item.sentiment === 'positive') {
                    return `<span class="word-pos ${stopClass}" title="VADER: +${item.score}">${item.word}</span> `;
                } else if (item.sentiment === 'negative') {
                    return `<span class="word-neg ${stopClass}" title="VADER: ${item.score}">${item.word}</span> `;
                } else {
                    return `<span class="${stopClass}">${item.word}</span> `;
                }
            }).join('');
        }

        function renderSentimentDoughnut(pos, neg, neu) {
            const ctx = document.getElementById('sentimentPieChart').getContext('2d');
            if (pieChartInstance) {
                pieChartInstance.destroy();
            }

            pieChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{
                        data: [pos || 0.01, neg || 0.01, neu || 0.01],
                        backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    cutout: '65%'
                }
            });
        }

        async function uploadFileBatch() {
            const fileInput = document.getElementById('csvFileInput');
            if (!fileInput.files.length) {
                alert('Please select a .csv or .txt file first!');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const res = await fetch('/api/upload-csv', {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) throw new Error('File upload failed');
                const data = await res.json();
                renderBatchResults(data);
            } catch (e) {
                alert('Error uploading file: ' + e.message);
            }
        }

        async function triggerBatchAnalyze() {
            const raw = document.getElementById('batchInput').value.trim();
            if (!raw) {
                alert('Please enter at least one review per line!');
                return;
            }

            const lines = raw.split('\\n').map(l => l.trim()).filter(l => l.length > 0);
            const spinner = document.getElementById('batchSpinner');
            spinner.style.display = 'inline-block';

            try {
                const res = await fetch('/api/analyze-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ reviews: lines })
                });

                if (!res.ok) throw new Error('Batch analysis failed');
                const data = await res.json();
                renderBatchResults(data);
            } catch (err) {
                alert(err.message);
            } finally {
                spinner.style.display = 'none';
            }
        }

        function renderBatchResults(data) {
            document.getElementById('batchResultsContainer').style.display = 'block';
            document.getElementById('batchTotalCount').innerText = data.total_reviews;
            document.getElementById('batchPosCount').innerText = data.sentiment_distribution.Positive;
            document.getElementById('batchNegCount').innerText = data.sentiment_distribution.Negative;
            document.getElementById('batchNeuCount').innerText = data.sentiment_distribution.Neutral;

            const tbody = document.getElementById('batchTableBody');
            tbody.innerHTML = data.detailed_results.map((r, i) => `
                <tr>
                    <td>${i + 1}</td>
                    <td>${r.original_review.substring(0, 50)}...</td>
                    <td><strong>${r.sentiment}</strong></td>
                    <td>${r.confidence_percentage}%</td>
                    <td>${r.vader_scores.compound_score}</td>
                    <td>${r.priority}</td>
                </tr>
            `).join('');
        }

        async function loadHistory() {
            try {
                const res = await fetch('/api/history?limit=10');
                const data = await res.json();
                const history = data.history || [];

                const tbody = document.getElementById('historyTableBody');
                if (!history.length) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No history recorded yet.</td></tr>';
                    return;
                }

                tbody.innerHTML = history.map(h => `
                    <tr>
                        <td>#${h.id}</td>
                        <td>${h.review.substring(0, 45)}...</td>
                        <td><strong>${h.sentiment}</strong></td>
                        <td>${h.confidence}%</td>
                        <td>${h.vader_compound}</td>
                        <td>${h.priority}</td>
                        <td style="color: var(--text-muted); font-size: 12px;">${h.created_at}</td>
                    </tr>
                `).join('');

                const chronological = [...history].reverse();
                const labels = chronological.map((_, idx) => `#${idx + 1}`);
                const compounds = chronological.map(h => h.vader_compound);

                const ctx = document.getElementById('trendLineChart').getContext('2d');
                if (trendChartInstance) {
                    trendChartInstance.destroy();
                }

                trendChartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'VADER Sentiment Compound (-1 to +1)',
                            data: compounds,
                            borderColor: '#4f46e5',
                            backgroundColor: 'rgba(79, 70, 229, 0.08)',
                            fill: true,
                            tension: 0.35,
                            pointBackgroundColor: compounds.map(c => c > 0.05 ? '#16a34a' : (c < -0.05 ? '#dc2626' : '#d97706')),
                            pointRadius: 6,
                            pointHoverRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                min: -1,
                                max: 1,
                                grid: { color: '#f1f5f9' },
                                ticks: { color: '#64748b', font: { weight: '600' } }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: '#64748b', font: { weight: '600' } }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#1e293b', font: { weight: '600' } }
                            }
                        }
                    }
                });
            } catch (err) {
                console.error('History load error:', err);
            }
        }

        async function loadMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();
                
                const tbody = document.getElementById('metricsTableBody');
                tbody.innerHTML = `
                    <tr>
                        <td><strong>${data.logistic_regression.model_name}</strong></td>
                        <td>${(data.logistic_regression.accuracy * 100).toFixed(1)}%</td>
                        <td>${(data.logistic_regression.precision_macro * 100).toFixed(1)}%</td>
                        <td>${(data.logistic_regression.recall_macro * 100).toFixed(1)}%</td>
                        <td>${(data.logistic_regression.f1_macro * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                        <td><strong>${data.naive_bayes.model_name}</strong></td>
                        <td>${(data.naive_bayes.accuracy * 100).toFixed(1)}%</td>
                        <td>${(data.naive_bayes.precision_macro * 100).toFixed(1)}%</td>
                        <td>${(data.naive_bayes.recall_macro * 100).toFixed(1)}%</td>
                        <td>${(data.naive_bayes.f1_macro * 100).toFixed(1)}%</td>
                    </tr>
                `;

                const cm = data.logistic_regression.confusion_matrix;
                const labels = data.labels;
                const cmBody = document.getElementById('cmBody');
                
                cmBody.innerHTML = cm.map((row, rIdx) => `
                    <tr>
                        <th style="background:#f8fafc; font-weight:700;">Actual: ${labels[rIdx]}</th>
                        ${row.map((val, cIdx) => `
                            <td class="${rIdx === cIdx ? 'cm-diag' : (val > 0 ? 'cm-off' : '')}">
                                ${val}
                            </td>
                        `).join('')}
                    </tr>
                `).join('');

            } catch (err) {
                console.error(err);
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            loadHistory();
            loadMetrics();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
