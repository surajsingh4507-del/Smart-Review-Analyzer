import os
import sqlite3
import json
import csv
import io
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
import joblib

from nlp_engine import complete_review_analysis, clean_and_tokenize
from ml_trainer import train_and_evaluate_models
from document_parser import validate_and_extract_document, split_into_reviews, DocumentParsingError
from pdf_exporter import generate_batch_pdf_report

app = FastAPI(
    title="Smart Review Analyzer API",
    description="Multi-layer NLP Sentiment & Review Intelligence Engine with Multi-Format Document Support",
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
except Exception as e:
    print(f"ML Models Load Warning: {e}")
    vectorizer, lr_model, nb_model = None, None, None


class SingleReviewRequest(BaseModel):
    review: str
    threshold: Optional[float] = 0.05


class BatchReviewRequest(BaseModel):
    reviews: Optional[List[str]] = None
    text: Optional[str] = None
    threshold: Optional[float] = 0.05
    mode: Optional[str] = "auto"


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Smart Review Analyzer",
        "version": "3.0.0",
        "nlp_engine": "VADER + TextBlob + TF-IDF (N-gram) + Word Tokenizer + Scikit-Learn",
        "ml_models_loaded": vectorizer is not None and lr_model is not None and nb_model is not None,
        "supported_formats": [".docx", ".doc", ".pdf", ".csv", ".xlsx", ".xls", ".txt"],
        "database": "SQLite"
    }


@app.get("/api/docs", include_in_schema=False)
def api_docs_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.post("/api/analyze")
@app.post("/analyze")
def analyze_single_review(payload: SingleReviewRequest):
    """
    Dedicated exclusively to single text review analysis.
    Returns predicted sentiment, confidence, VADER scores, TextBlob metrics, ML predictions,
    and a clear sentiment explanation.
    """
    text = payload.review.strip() if payload.review else ""
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    threshold = payload.threshold if payload.threshold is not None else 0.05
    result = complete_review_analysis(text, threshold, vectorizer, lr_model, nb_model)
    
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
    """
    Process pasted batch reviews (one per line, numbered, or paragraphs).
    """
    review_items = []
    
    if payload.reviews:
        for idx, r in enumerate(payload.reviews):
            cleaned = r.strip()
            if cleaned and cleaned.lower().replace("_", " ").strip() not in ["review text", "review", "reviews", "customer review"]:
                review_items.append({"id": f"#{len(review_items) + 1}", "text": cleaned})
    elif payload.text:
        text_content = payload.text.strip()
        if text_content:
            raw_lines = [p.strip() for p in text_content.splitlines() if p.strip()]
            if raw_lines and raw_lines[0].lower().replace("_", " ").strip() in [
                "review text", "review", "reviews", "customer review", "feedback", "comment", "text"
            ]:
                raw_lines = raw_lines[1:]
                text_content = "\n".join(raw_lines)
            extracted = {
                "text": text_content,
                "paragraphs": raw_lines,
                "elements": [{"type": "text_block", "text": line, "source_id": f"Line {i+1}"} for i, line in enumerate(raw_lines)],
                "tables": [],
                "metadata": {"format": "txt", "detected_encoding": "utf-8"},
                "diagnostics": {
                    "filename": "pasted_reviews.txt",
                    "file_extension": ".txt",
                    "character_count": len(text_content),
                    "word_count": len(text_content.split())
                }
            }
            review_items = split_into_reviews(extracted, mode=payload.mode or "auto")

    if not review_items:
        raise HTTPException(status_code=400, detail="Review list or pasted text cannot be empty.")
        
    threshold = payload.threshold if payload.threshold is not None else 0.05
    results = []
    for item in review_items:
        r_text = item["text"] if isinstance(item, dict) else str(item)
        source_id = item.get("id", f"#{len(results) + 1}") if isinstance(item, dict) else f"#{len(results) + 1}"
        r_analysis = complete_review_analysis(r_text, threshold, vectorizer, lr_model, nb_model)
        r_analysis["source_id"] = source_id
        results.append(r_analysis)
    
    total = len(results)
    pos_count = sum(1 for r in results if r["sentiment"] == "Positive")
    neg_count = sum(1 for r in results if r["sentiment"] == "Negative")
    neu_count = sum(1 for r in results if r["sentiment"] == "Neutral")

    # Document/Batch-wide sentence-level distribution
    all_sentences = []
    for r in results:
        s_data = r.get("sentence_analysis", {})
        for s in s_data.get("sentences", []):
            all_sentences.append(s)

    tot_sentences = len(all_sentences)
    pos_sents = sum(1 for s in all_sentences if s.get("sentiment") == "Positive")
    neg_sents = sum(1 for s in all_sentences if s.get("sentiment") == "Negative")
    neu_sents = sum(1 for s in all_sentences if s.get("sentiment") == "Neutral")

    pos_s_pct = round((pos_sents / tot_sentences) * 100, 1) if tot_sentences > 0 else 0.0
    neg_s_pct = round((neg_sents / tot_sentences) * 100, 1) if tot_sentences > 0 else 0.0
    neu_s_pct = round((neu_sents / tot_sentences) * 100, 1) if tot_sentences > 0 else 0.0

    valid_compounds = [s["vader_compound"] for s in all_sentences if "vader_compound" in s]
    doc_mean_compound = round(sum(valid_compounds) / len(valid_compounds), 4) if valid_compounds else 0.0

    if doc_mean_compound > threshold:
        doc_overall_sent = "Positive"
    elif doc_mean_compound < -threshold:
        doc_overall_sent = "Negative"
    else:
        doc_overall_sent = "Neutral"

    sentence_distribution = {
        "total_sentences": tot_sentences,
        "Positive": pos_sents,
        "Negative": neg_sents,
        "Neutral": neu_sents,
        "positive_sentences": pos_sents,
        "negative_sentences": neg_sents,
        "neutral_sentences": neu_sents,
        "Positive_Percentage": pos_s_pct,
        "Negative_Percentage": neg_s_pct,
        "Neutral_Percentage": neu_s_pct,
        "positive_percentage": pos_s_pct,
        "negative_percentage": neg_s_pct,
        "neutral_percentage": neu_s_pct,
        "mean_compound_score": doc_mean_compound,
        "overall_sentiment": doc_overall_sent
    }

    aspect_counts = {}
    for r in results:
        for cat in r.get("categories", []):
            aspect_counts[cat] = aspect_counts.get(cat, 0) + 1
    
    diagnostics = {
        "filename": "Pasted Reviews",
        "file_extension": ".txt",
        "file_size_formatted": f"{sum(len(r['original_review']) for r in results)} chars",
        "character_count": sum(len(r["original_review"]) for r in results),
        "word_count": sum(len(r["original_review"].split()) for r in results),
        "reviews_identified": total,
        "sentences_identified": tot_sentences,
        "successfully_analyzed": total,
        "failed_records": 0,
        "processing_mode": payload.mode or "auto",
        "preview": results[0]["original_review"][:100] + "..." if results else ""
    }

    return {
        "total_reviews": total,
        "reviews_processed": total,
        "reviews_failed": 0,
        "sentiment_distribution": {
            "Positive": pos_count,
            "Negative": neg_count,
            "Neutral": neu_count,
            "Positive_Percentage": round((pos_count / total) * 100, 1) if total > 0 else 0.0,
            "Negative_Percentage": round((neg_count / total) * 100, 1) if total > 0 else 0.0,
            "Neutral_Percentage": round((neu_count / total) * 100, 1) if total > 0 else 0.0,
        },
        "sentence_distribution": sentence_distribution,
        "aspect_distribution": aspect_counts,
        "diagnostics": diagnostics,
        "detailed_results": results
    }


@app.post("/api/upload-document")
@app.post("/upload-document")
@app.post("/api/upload-file")
@app.post("/api/upload-csv")
@app.post("/upload-csv")
def upload_document_endpoint(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.05,
    mode: Optional[str] = "auto",
    column_name: Optional[str] = None
):
    """
    Multi-Format Document Upload & Batch Analysis.
    Supported formats: .docx, .doc, .pdf, .csv, .xlsx, .xls, .txt.
    Provides complete diagnostics, review segmentation, and multi-model consensus.
    """
    if hasattr(file, 'file') and hasattr(file.file, 'read'):
        contents = file.file.read()
    else:
        contents = file.read()
        
    try:
        extracted = validate_and_extract_document(file.filename, contents, column_name=column_name)
    except DocumentParsingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing document: {str(e)}")

    valid_mode = mode if mode in ["auto", "single", "multiple", "paragraph"] else "auto"
    review_items = split_into_reviews(extracted, mode=valid_mode, column_name=column_name)
    
    if not review_items:
        raise HTTPException(
            status_code=400,
            detail="No readable review records could be identified in the uploaded document."
        )

    thresh = threshold if threshold is not None else 0.05
    results = []
    failed_records = 0

    for item in review_items:
        try:
            r_text = item["text"] if isinstance(item, dict) else str(item)
            source_id = item.get("id", f"#{len(results) + 1}") if isinstance(item, dict) else f"#{len(results) + 1}"
            
            if not r_text.strip():
                failed_records += 1
                continue
                
            r_analysis = complete_review_analysis(r_text, thresh, vectorizer, lr_model, nb_model)
            r_analysis["source_id"] = source_id
            results.append(r_analysis)
        except Exception as item_err:
            print(f"Item analysis warning: {item_err}")
            failed_records += 1

    total = len(results)
    if total == 0:
        raise HTTPException(
            status_code=400,
            detail="Document text was extracted, but no valid non-empty review content could be analyzed."
        )

    pos_count = sum(1 for r in results if r["sentiment"] == "Positive")
    neg_count = sum(1 for r in results if r["sentiment"] == "Negative")
    neu_count = sum(1 for r in results if r["sentiment"] == "Neutral")

    # Document-wide sentence-level distribution across all reviews/records
    all_sentences = []
    for r in results:
        s_data = r.get("sentence_analysis", {})
        for s in s_data.get("sentences", []):
            all_sentences.append(s)

    tot_sentences = len(all_sentences)
    pos_sents = sum(1 for s in all_sentences if s.get("sentiment") == "Positive")
    neg_sents = sum(1 for s in all_sentences if s.get("sentiment") == "Negative")
    neu_sents = sum(1 for s in all_sentences if s.get("sentiment") == "Neutral")

    pos_s_pct = round((pos_sents / tot_sentences) * 100, 1) if tot_sentences > 0 else 0.0
    neg_s_pct = round((neg_sents / tot_sentences) * 100, 1) if tot_sentences > 0 else 0.0
    neu_s_pct = round((neu_sents / tot_sentences) * 100, 1) if tot_sentences > 0 else 0.0

    valid_compounds = [s["vader_compound"] for s in all_sentences if "vader_compound" in s]
    doc_mean_compound = round(sum(valid_compounds) / len(valid_compounds), 4) if valid_compounds else 0.0

    if doc_mean_compound > thresh:
        doc_overall_sent = "Positive"
    elif doc_mean_compound < -thresh:
        doc_overall_sent = "Negative"
    else:
        doc_overall_sent = "Neutral"

    sentence_distribution = {
        "total_sentences": tot_sentences,
        "Positive": pos_sents,
        "Negative": neg_sents,
        "Neutral": neu_sents,
        "positive_sentences": pos_sents,
        "negative_sentences": neg_sents,
        "neutral_sentences": neu_sents,
        "Positive_Percentage": pos_s_pct,
        "Negative_Percentage": neg_s_pct,
        "Neutral_Percentage": neu_s_pct,
        "positive_percentage": pos_s_pct,
        "negative_percentage": neg_s_pct,
        "neutral_percentage": neu_s_pct,
        "mean_compound_score": doc_mean_compound,
        "overall_sentiment": doc_overall_sent
    }

    aspect_counts = {}
    for r in results:
        for cat in r.get("categories", []):
            aspect_counts[cat] = aspect_counts.get(cat, 0) + 1

    diagnostics = {
        **extracted["diagnostics"],
        "processing_mode": valid_mode,
        "reviews_identified": len(review_items),
        "sentences_identified": tot_sentences,
        "successfully_analyzed": total,
        "failed_records": failed_records
    }

    # Store a sample of analyzed batch reviews into SQLite history
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for r in results[:20]:
            cur.execute("""
                INSERT INTO reviews_history (review, sentiment, confidence, categories, priority, vader_compound, subjectivity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                r["original_review"][:500],
                r["sentiment"],
                r["confidence_percentage"],
                ", ".join(r["categories"]),
                r["priority"],
                r["vader_scores"]["compound_score"],
                r["textblob_metrics"]["subjectivity"]
            ))
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"DB Batch Insert Warning: {db_err}")

    return {
        "total_reviews": total,
        "reviews_processed": total,
        "reviews_failed": failed_records,
        "sentiment_distribution": {
            "Positive": pos_count,
            "Negative": neg_count,
            "Neutral": neu_count,
            "Positive_Percentage": round((pos_count / total) * 100, 1) if total > 0 else 0.0,
            "Negative_Percentage": round((neg_count / total) * 100, 1) if total > 0 else 0.0,
            "Neutral_Percentage": round((neu_count / total) * 100, 1) if total > 0 else 0.0,
        },
        "sentence_distribution": sentence_distribution,
        "aspect_distribution": aspect_counts,
        "diagnostics": diagnostics,
        "detailed_results": results
    }


@app.post("/api/export-batch-csv")
def export_batch_csv(data: Dict[str, Any]):
    """
    Exports batch sentiment analysis results as a standardized CSV file.
    Filename: sentiment_results_YYYYMMDD_HHMMSS.csv
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Source ID / Row",
        "Original Review",
        "Sentiment",
        "ML Confidence (LR %)",
        "VADER Compound",
        "TextBlob Polarity",
        "ML Prediction (LR)",
        "Aspect Categories",
        "Priority",
        "Sentiment Explanation"
    ])

    results = data.get("detailed_results", [])
    for r in results:
        source_id = r.get("source_id", "")
        review_text = r.get("original_review", "")
        sentiment = r.get("sentiment", "Neutral")
        conf_val = r.get("confidence_percentage")
        confidence_str = f"{conf_val}%" if conf_val is not None else "N/A"
        vader_comp = r.get("overall_compound_score") if r.get("overall_compound_score") is not None else r.get("vader_scores", {}).get("compound_score", 0.0)
        tb_pol = r.get("textblob_metrics", {}).get("polarity", 0.0)
        ml_pred = r.get("ml_model_predictions", {}).get("logistic_regression", "N/A")
        cats = ", ".join(r.get("categories", [])) or "General"
        priority = r.get("priority", "Low")
        explanation = r.get("explanation", "")

        writer.writerow([
            source_id, review_text, sentiment, confidence_str,
            vader_comp, tb_pol, ml_pred, cats, priority, explanation
        ])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_results_{timestamp}.csv"
    output.seek(0)
    
    return Response(
        content=output.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


@app.post("/api/export-batch-pdf")
def export_batch_pdf(data: Dict[str, Any]):
    """
    Generates and downloads a paginated, formatted PDF report of batch sentiment analysis.
    Filename: sentiment_results_YYYYMMDD_HHMMSS.pdf
    """
    try:
        pdf_bytes = generate_batch_pdf_report(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_results_{timestamp}.pdf"
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


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
        iter([output.getvalue().encode("utf-8")]),
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
    <title>Smart Review Analyzer v3.0 - AI Sentiment & Multi-Format Intelligence</title>
    <!-- Chart.js for Visualizations -->
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
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }

        .container {
            max-width: 1240px;
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

        /* Architecture bar */
        .info-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #4f46e5;
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 20px;
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
            font-size: 11.5px;
            border: 1px solid #c7d2fe;
        }

        /* Tabs */
        .tab-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--card-border);
            padding-bottom: 4px;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 15px;
            font-weight: 600;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .tab-btn.active {
            color: #4f46e5;
            background: #ffffff;
            border: 1px solid var(--card-border);
            border-bottom: 2px solid #ffffff;
            margin-bottom: -6px;
            font-weight: 700;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.03);
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

        select, textarea, input[type="text"] {
            width: 100%;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            color: #0f172a;
            padding: 11px 13px;
            font-size: 14px;
            margin-bottom: 14px;
            outline: none;
            transition: all 0.2s;
        }

        select:focus, textarea:focus, input[type="text"]:focus {
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

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
            padding: 7px 14px;
            border-radius: 6px;
            font-size: 12.5px;
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
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12.5px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }
        .btn-mic:hover { background: #e0e7ff; }
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

        @keyframes spin { to { transform: rotate(360deg); } }

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
            font-size: 14.5px;
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
            padding: 12px 14px;
        }

        .metric-title {
            font-size: 11.5px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 18px;
            font-weight: 800;
            margin-top: 4px;
            color: #0f172a;
        }

        .explanation-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 12px;
            font-size: 13.5px;
            color: #1e40af;
            font-weight: 500;
            line-height: 1.5;
        }

        .recommendation-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 10px;
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
            font-size: 12px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Batch styling */
        .batch-option-box {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 20px;
        }
        .batch-option-box.active-box {
            border-color: #4f46e5;
            box-shadow: 0 0 0 1px #4f46e5;
        }

        .filter-btn {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12.5px;
            font-weight: 600;
            cursor: pointer;
            color: #475569;
            transition: all 0.2s;
        }
        .filter-btn.active {
            background: #4f46e5;
            color: #ffffff;
            border-color: #4f46e5;
        }

        .sent-row-pos {
            background-color: #f0fdf4 !important;
            border-left: 4px solid #22c55e !important;
        }
        .sent-row-neg {
            background-color: #fef2f2 !important;
            border-left: 4px solid #ef4444 !important;
        }
        .sent-row-neu {
            background-color: #fffbeb !important;
            border-left: 4px solid #f59e0b !important;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 10px;
        }

        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }

        th {
            background: #f8fafc;
            color: #475569;
            font-size: 11.5px;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        tbody tr:hover { background: #f8fafc; }

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
            padding: 12px 10px;
            text-align: center;
            min-width: 130px;
        }

        .pipeline-step h4 {
            font-size: 12.5px;
            color: #3730a3;
            margin-bottom: 4px;
            font-weight: 700;
        }

        .pipeline-step p {
            font-size: 11px;
            color: var(--text-muted);
        }

        .pipeline-arrow {
            color: #94a3b8;
            font-weight: bold;
            font-size: 16px;
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
            padding: 10px 16px;
            font-weight: 700;
        }
        .cm-diag {
            background: #dcfce7 !important;
            color: #15803d;
            font-size: 15px;
        }
        .cm-off {
            background: #fee2e2 !important;
            color: #b91c1c;
            font-size: 14px;
        }

        /* Modal */
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
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 14px;
            margin-bottom: 16px;
        }
        .modal-header h3 { font-size: 1.25rem; color: #1e293b; }
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
            .btn-primary, .btn-secondary, .btn-link, select, .tab-bar, .header-actions, .controls-row, .modal-overlay, #batchModeSection {
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
                <p>Natural Language Processing, Machine Learning & Multi-Format Document Intelligence</p>
            </div>
            <div class="header-actions">
                <button onclick="openDocModal()" class="btn-link" title="Open Technical Reference Documentation">
                    📖 Technical Reference
                </button>
                <a href="/docs" target="_blank" class="btn-link" title="Open Interactive Swagger REST API Docs">
                    🌐 API Docs (/docs)
                </a>
                <a href="/api/export-history-csv" class="btn-link" title="Download Historical Analyses in CSV format">
                    📥 History CSV
                </a>
                <button onclick="window.print()" class="btn-link" title="Print Current View">
                    🖨️ Print
                </button>
            </div>
        </header>

        <!-- System Architecture Bar -->
        <div class="info-card">
            <div class="info-card-left">
                <div>⚡ <strong>Architecture:</strong> Multi-Engine Consensus (VADER + TextBlob + TF-IDF ML)</div>
                <div>📁 <strong>Supported Formats:</strong> .DOCX, .DOC, .PDF, .CSV, .XLSX, .XLS, .TXT</div>
            </div>
            <div>
                <span class="info-badge">NLTK VADER</span>
                <span class="info-badge">TextBlob</span>
                <span class="info-badge">Scikit-Learn (LR + MNB)</span>
                <span class="info-badge">SQLite</span>
            </div>
        </div>

        <!-- Mode Tabs -->
        <div class="tab-bar">
            <button class="tab-btn active" id="singleTabBtn" onclick="switchTab('single')">
                📝 Single Review Analysis
            </button>
            <button class="tab-btn" id="batchTabBtn" onclick="switchTab('batch')">
                📋 Batch Review Analysis & Multi-Format File Upload
            </button>
        </div>

        <!-- ========================================== -->
        <!-- SECTION A: SINGLE REVIEW ANALYSIS (Text-Only) -->
        <!-- ========================================== -->
        <div id="singleModeSection">
            <div class="grid-2">
                <!-- Input Column -->
                <div class="card">
                    <h2>
                        <span>✍️ Single Review Input</span>
                        <button class="btn-mic" id="micBtn" onclick="toggleSpeechRecognition()" title="Voice Review Input via Microphone">
                            🎙️ <span id="micText">Speak Review</span>
                        </button>
                    </h2>
                    
                    <label for="sampleDropdown">📌 Preset Test Sample (Optional):</label>
                    <select id="sampleDropdown" onchange="handleSampleSelect(this.value)">
                        <option value="">-- Choose a pre-written test review --</option>
                        <option value="samsung_a55">📱 Samsung Galaxy A55 (5-Sentence Long Paragraph)</option>
                        <option value="tech_pos">📱 Smart Phone (Positive) - Battery & Screen</option>
                        <option value="tech_neg">💻 Laptop Overheating (Negative Defect)</option>
                        <option value="food_pos">🍕 Restaurant Delivery (Positive Experience)</option>
                        <option value="ecom_neg">📦 E-Commerce Shipping Delay (Complaint)</option>
                        <option value="spec_neu">📏 Neutral Specification & Dimensions</option>
                        <option value="support_neg">📞 Customer Support (Negative Rude Behavior)</option>
                        <option value="negation_pos">💡 Negation Handled ("Not bad at all, actually quite good!")</option>
                        <option value="negation_neg">⚠️ Negation Handled ("The camera is not good.")</option>
                    </select>

                    <label for="reviewInput">Customer Review Text:</label>
                    <textarea id="reviewInput" placeholder="Paste or type a product review here..."></textarea>

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

                    <div id="singleErrorAlert" style="display:none; padding:10px 14px; margin-bottom:14px; background:#fef2f2; border:1px solid #fecaca; border-radius:8px; color:#b91c1c; font-size:13px;"></div>

                    <button class="btn-primary" id="analyzeBtn" onclick="triggerAnalyze()">
                        <div class="spinner" id="analyzeSpinner"></div>
                        <span id="analyzeBtnText">⚡ Analyze Sentiment</span>
                    </button>
                </div>

                <!-- Single Diagnostics / Results Column -->
                <div class="card" id="resultsCard">
                    <h2>
                        <span>📊 Sentiment Diagnostics</span>
                        <span id="statusIndicator" style="font-size: 12px; color: var(--text-muted); font-weight: 600;">Ready</span>
                    </h2>

                    <div id="emptyPlaceholder" style="text-align: center; padding: 45px 0; color: var(--text-muted);">
                        <div style="font-size: 34px; margin-bottom: 8px;">🔍</div>
                        <p style="font-size: 14px;">Enter text above and click <strong>Analyze Sentiment</strong>.</p>
                    </div>

                    <div id="resultContent" style="display: none;">
                        <!-- 1. Overall Sentiment Card -->
                        <div style="background:#ffffff; border:1px solid var(--card-border); border-left:5px solid #4f46e5; border-radius:10px; padding:16px; margin-bottom:16px; box-shadow:0 2px 4px rgba(0,0,0,0.03);">
                            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                                <div>
                                    <div class="metric-title">Final Overall Sentiment</div>
                                    <div id="sentimentBadgeContainer" style="margin-top:4px;"></div>
                                </div>
                                <div>
                                    <div class="metric-title">Mean VADER Compound</div>
                                    <div id="meanCompoundVal" style="font-size:1.35rem; font-weight:800; color:#4f46e5;">+0.0000</div>
                                </div>
                                <div>
                                    <div class="metric-title">Selected Threshold (T)</div>
                                    <div id="selectedThresholdVal" style="font-size:1.1rem; font-weight:700; color:#334155;">±0.05</div>
                                </div>
                                <div>
                                    <div class="metric-title">Urgency Priority</div>
                                    <div id="priorityVal" style="font-size:1.1rem; font-weight:700;">Normal</div>
                                </div>
                                <div id="singleConfidenceCard">
                                    <div class="metric-title">ML Confidence (LR)</div>
                                    <div id="singleConfidenceVal" style="font-size:1.1rem; font-weight:700; color:#4f46e5;">N/A</div>
                                </div>
                            </div>
                            <div id="singleExplanationCard" class="explanation-box" style="margin-top:12px;">
                                <div id="overallCalcExplanation"></div>
                            </div>
                        </div>

                        <!-- 2. Sentence Summary Cards (4 Cards) -->
                        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin-bottom:16px;">
                            <div class="metric-tile">
                                <div class="metric-title">Total Sentences</div>
                                <div class="metric-value" id="totalSentencesVal">0</div>
                            </div>
                            <div class="metric-tile" style="border-left:3px solid var(--accent-pos);">
                                <div class="metric-title" style="color:var(--accent-pos);">Positive Sentences</div>
                                <div class="metric-value" id="posSentencesVal" style="color:var(--accent-pos); font-size:15px;">0 (0%)</div>
                            </div>
                            <div class="metric-tile" style="border-left:3px solid var(--accent-neg);">
                                <div class="metric-title" style="color:var(--accent-neg);">Negative Sentences</div>
                                <div class="metric-value" id="negSentencesVal" style="color:var(--accent-neg); font-size:15px;">0 (0%)</div>
                            </div>
                            <div class="metric-tile" style="border-left:3px solid var(--accent-neu);">
                                <div class="metric-title" style="color:var(--accent-neu);">Neutral Sentences</div>
                                <div class="metric-value" id="neuSentencesVal" style="color:var(--accent-neu); font-size:15px;">0 (0%)</div>
                            </div>
                        </div>

                        <!-- 3. Sentence Sentiment Distribution Chart & Model Consensus -->
                        <div style="display:grid; grid-template-columns:160px 1fr; gap:16px; align-items:center; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px; margin-bottom:16px;">
                            <div style="height:150px; width:150px; position:relative; margin:0 auto;">
                                <canvas id="sentimentPieChart"></canvas>
                            </div>
                            <div>
                                <div class="metric-title" style="margin-bottom:6px;">⚖️ Multi-Model Consensus (Review Level):</div>
                                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px; font-weight:600;">
                                    <div>VADER: <span id="cmpVader" style="color:#4f46e5; font-weight:700;">-</span></div>
                                    <div>LogReg (TF-IDF): <span id="cmpLR" style="color:#059669; font-weight:700;">-</span></div>
                                    <div>Naive Bayes: <span id="cmpNB" style="color:#6366f1; font-weight:700;">-</span></div>
                                    <div>TextBlob: <span id="cmpTB" style="color:#d97706; font-weight:700;">-</span></div>
                                </div>
                                <div id="disagreementAlert" style="display:none;" class="disagreement-box"></div>
                                <div id="recommendationAlert" class="recommendation-box" style="margin-top:8px; padding:8px 12px; font-size:12px;"></div>
                            </div>
                        </div>

                        <!-- 4. Sentence-Level Results Table with Interactive Filters -->
                        <div style="margin-top:16px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
                                <div class="metric-title" style="font-size:12.5px;">📑 Sentence-Level Results Table:</div>
                                <div style="display:flex; gap:6px; align-items:center;">
                                    <span style="font-size:11.5px; color:var(--text-muted); font-weight:600;">Filter:</span>
                                    <button class="filter-btn active" id="sentFilterAll" onclick="setSentenceTableFilter('All')">All</button>
                                    <button class="filter-btn" id="sentFilterPos" onclick="setSentenceTableFilter('Positive')">Positive</button>
                                    <button class="filter-btn" id="sentFilterNeg" onclick="setSentenceTableFilter('Negative')">Negative</button>
                                    <button class="filter-btn" id="sentFilterNeu" onclick="setSentenceTableFilter('Neutral')">Neutral</button>
                                </div>
                            </div>

                            <div style="overflow-x:auto; max-height:360px; overflow-y:auto; border:1px solid var(--card-border); border-radius:8px;">
                                <table>
                                    <thead style="position:sticky; top:0; background:#f8fafc; z-index:2;">
                                        <tr>
                                            <th style="width:40px;">#</th>
                                            <th>Sentence Text</th>
                                            <th style="width:85px;">Sentiment</th>
                                            <th style="width:80px;">VADER</th>
                                            <th style="width:80px;">TextBlob</th>
                                            <th style="width:80px;">ML (LR)</th>
                                        </tr>
                                    </thead>
                                    <tbody id="sentenceTableBody"></tbody>
                                </table>
                            </div>
                        </div>

                        <!-- 5. Word-Level Lexicon Polarity Highlighting & Aspect Cloud -->
                        <div style="margin-top:16px;">
                            <label>🎨 Lexicon Word-Level Polarity Highlighting:</label>
                            <div class="word-highlight-box" id="highlightedWordsBox"></div>
                            <div style="display: flex; gap: 14px; font-size: 11.5px; margin-top: 6px; color: var(--text-muted); font-weight: 600;">
                                <span><span style="color: #166534; font-weight: 900;">■</span> Green: Positive</span>
                                <span><span style="color: #991b1b; font-weight: 900;">■</span> Red: Negative</span>
                                <span><span style="opacity: 0.5; font-style: italic;">Strikethrough</span>: Filtered Stopword</span>
                            </div>
                        </div>

                        <div style="margin-top: 14px;">
                            <div class="metric-title">☁️ Extracted Aspect Keywords:</div>
                            <div class="bubble-cloud" id="bubbleCloudContainer"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- SECTION B: BATCH REVIEW ANALYSIS & MULTI-FORMAT FILE UPLOAD    -->
        <!-- ============================================================== -->
        <div id="batchModeSection" style="display: none;">
            <div class="card">
                <h2>
                    <span>📋 Batch Review Analysis & Multi-Format File Upload</span>
                    <span style="font-size:12px; color:var(--text-muted); font-weight:500;">Processes files & bulk reviews without manual copy-pasting</span>
                </h2>

                <!-- Option 1: Upload a File -->
                <div class="batch-option-box" id="boxUploadOption">
                    <h3 style="font-size:14px; color:#1e293b; margin-bottom:10px; display:flex; align-items:center; gap:8px;">
                        <span>📁 Option 1: Upload a File</span>
                        <span style="font-size:11px; background:#e0e7ff; color:#3730a3; padding:2px 8px; border-radius:10px; font-weight:700;">
                            CSV, TXT, PDF, DOCX, DOC, XLSX, XLS
                        </span>
                    </h3>
                    
                    <div style="display:grid; grid-template-columns: 1.5fr 1fr 1fr; gap:12px; align-items:end;">
                        <div>
                            <label for="multiFileInput">Select Document / Spreadsheet:</label>
                            <input type="file" id="multiFileInput" accept=".docx,.doc,.pdf,.csv,.xlsx,.xls,.txt" style="margin-bottom:0; background:#fff;">
                        </div>
                        <div>
                            <label for="docModeSelect">Review Boundary Detection:</label>
                            <select id="docModeSelect" style="margin-bottom:0; background:#fff;">
                                <option value="auto">🔍 Auto-Detect (Recommended)</option>
                                <option value="multiple">📑 Multiple Reviews (Numbered, Rows, Bulleted)</option>
                                <option value="single">📄 Single Review (Entire Document as 1)</option>
                                <option value="paragraph">¶ Paragraphs (Split by Blank Lines)</option>
                            </select>
                        </div>
                        <div>
                            <label for="csvColumnInput">Review Column Name (Optional):</label>
                            <input type="text" id="csvColumnInput" placeholder="Auto-detect (e.g. review, comment)" style="margin-bottom:0; background:#fff;">
                        </div>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:14px; flex-wrap:wrap; gap:10px;">
                        <span style="font-size:12px; color:var(--text-muted);">
                            Supports multi-page PDFs, tables in Word/DOCX, worksheets in Excel, and automatic column detection.
                        </span>
                        <button class="btn-primary" style="width:auto; padding:10px 22px;" id="uploadFileBtn" onclick="uploadMultiFormatFile()">
                            <div class="spinner" id="batchUploadSpinner" style="display:none;"></div>
                            <span id="uploadBtnText">📁 Upload & Process File</span>
                        </button>
                    </div>
                </div>

                <!-- Option 2: Paste Multiple Reviews -->
                <div class="batch-option-box" id="boxPasteOption">
                    <h3 style="font-size:14px; color:#1e293b; margin-bottom:10px;">
                        ✍️ Option 2: Paste Multiple Reviews
                    </h3>
                    <label for="batchInput">Paste multiple reviews below (one review per line, numbered 1. 2. 3., or separated by empty lines):</label>
                    <textarea id="batchInput" style="height: 110px;" placeholder="1. The product quality is outstanding and arrived quickly!&#10;2. Very disappointing battery life, stopped working after two days.&#10;3. The customer support agent was helpful and solved my issue.&#10;4. The device dimensions are 15cm by 10cm."></textarea>
                    
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div style="display:flex; gap:10px; align-items:center;">
                            <label style="margin:0; font-size:12.5px;">Split Strategy:</label>
                            <select id="pastedModeSelect" style="width:auto; margin:0; padding:6px 10px; font-size:12.5px; background:#fff;">
                                <option value="auto">Auto-detect (Numbered / Line / Paragraph)</option>
                                <option value="multiple">Line-by-Line</option>
                                <option value="paragraph">Paragraphs (Double Newline)</option>
                                <option value="single">Treat as Single Review</option>
                            </select>
                        </div>
                        <button class="btn-primary" style="width:auto; padding:10px 22px;" id="processPastedBtn" onclick="triggerPastedBatchAnalyze()">
                            <div class="spinner" id="batchPasteSpinner" style="display:none;"></div>
                            <span id="processPastedBtnText">⚡ Process All Reviews in Bulk</span>
                        </button>
                    </div>
                </div>

                <!-- Error Alert Box -->
                <div id="batchErrorAlert" style="display: none; padding: 12px 16px; margin-bottom: 16px; background: #fef2f2; border: 1px solid #f87171; border-radius: 8px; color: #b91c1c; font-size: 13.5px;"></div>

                <!-- Diagnostics & Extraction Summary Banner -->
                <div id="batchDiagnosticsCard" style="display: none; margin-bottom: 20px; padding: 16px 20px; background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <h4 style="color: #166534; font-size: 14px; display: flex; align-items: center; gap: 8px;">
                            <span>🔍 Document Extraction Diagnostics</span>
                            <span id="diagFormatBadge" style="background: #bbf7d0; color: #166534; font-size: 11px; padding: 2px 10px; border-radius: 10px; font-weight: 700;">DOCX</span>
                        </h4>
                        <span id="diagStatusBadge" style="font-size:11.5px; color:#15803d; font-weight:700;">Extraction Verified ✓</span>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; font-size: 12.5px; color: #1e293b;">
                        <div><strong>File Name:</strong> <span id="diagFilename">-</span></div>
                        <div><strong>File Size:</strong> <span id="diagSize">-</span></div>
                        <div><strong>Characters:</strong> <span id="diagChars">0</span></div>
                        <div><strong>Words:</strong> <span id="diagWords">0</span></div>
                        <div><strong>Identified Reviews:</strong> <span id="diagIdentified" style="color: #15803d; font-weight: 700;">0</span></div>
                        <div><strong>Processing Mode:</strong> <span id="diagMode">auto</span></div>
                    </div>

                    <div id="diagWarningAlert" style="display:none; margin-top:10px; padding:8px 12px; background:#fffbeb; border:1px solid #fde68a; border-radius:6px; color:#b45309; font-size:12px;"></div>

                    <div style="margin-top: 10px; font-size: 12px; color: #475569; background: #ffffff; padding: 8px 12px; border-radius: 6px; border: 1px solid #bbf7d0;">
                        <strong>Text Preview:</strong> <span id="diagPreview">-</span>
                    </div>
                </div>

                <!-- Batch Results Container -->
                <div id="batchResultsContainer" style="display: none;">
                    <!-- Document-Wide Sentence-Level Distribution Banner -->
                    <div id="batchSentenceSummaryBanner" style="display: none; background: #ffffff; border: 1px solid #cbd5e1; border-left: 5px solid #3b82f6; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div style="font-weight: 700; color: #1e40af; font-size: 13.5px; display: flex; align-items: center; gap: 6px;">
                                <span>📑 Document-Wide Sentence Distribution:</span>
                                <span id="batchSentTotalCount" style="background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 10px; font-size: 11.5px;">0 sentences</span>
                            </div>
                            <div style="display: flex; gap: 16px; font-size: 12.5px; font-weight: 600; flex-wrap: wrap;">
                                <span style="color: var(--accent-pos);">Positive: <strong id="batchSentPosCount">0 (0%)</strong></span>
                                <span style="color: var(--accent-neg);">Negative: <strong id="batchSentNegCount">0 (0%)</strong></span>
                                <span style="color: var(--accent-neu);">Neutral: <strong id="batchSentNeuCount">0 (0%)</strong></span>
                                <span style="color: #4f46e5;">Mean Compound: <strong id="batchSentMeanScore">+0.0000</strong></span>
                            </div>
                        </div>
                    </div>

                    <!-- 5 Metric Scorecards (Review Level) -->
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 20px;">
                        <div class="metric-tile">
                            <div class="metric-title">Total Analyzed</div>
                            <div class="metric-value" id="batchTotalCount">0</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-title" style="color: var(--accent-pos);">Positive Reviews</div>
                            <div class="metric-value" id="batchPosCount" style="color: var(--accent-pos);">0</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-title" style="color: var(--accent-neg);">Negative Reviews</div>
                            <div class="metric-value" id="batchNegCount" style="color: var(--accent-neg);">0</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-title" style="color: var(--accent-neu);">Neutral Reviews</div>
                            <div class="metric-value" id="batchNeuCount" style="color: var(--accent-neu);">0</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-title" style="color: #64748b;">Failed / Skipped</div>
                            <div class="metric-value" id="batchFailedCount" style="color: #64748b;">0</div>
                        </div>
                    </div>

                    <!-- Visual Charts Row: Doughnut + Bar Chart -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 20px;">
                        <div>
                            <div class="metric-title" style="margin-bottom: 8px; text-align:center;">📊 Sentiment Ratio (Doughnut)</div>
                            <div style="height: 180px; position: relative;">
                                <canvas id="batchPieChart"></canvas>
                            </div>
                        </div>
                        <div>
                            <div class="metric-title" style="margin-bottom: 8px; text-align:center;">📈 Review Volume by Category (Bar Chart)</div>
                            <div style="height: 180px; position: relative;">
                                <canvas id="batchBarChart"></canvas>
                            </div>
                        </div>
                    </div>

                    <!-- Aspect Breakdown -->
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px; margin-bottom:20px;">
                        <div class="metric-title" style="margin-bottom: 6px;">🏷️ Extracted Aspect Category Distribution:</div>
                        <div class="bubble-cloud" id="batchAspectTags" style="gap: 8px;"></div>
                    </div>

                    <!-- Filter & Export Action Toolbar -->
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px;">
                        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                            <input type="text" id="batchSearchInput" placeholder="🔍 Search reviews..." oninput="filterBatchTable()" style="width:220px; margin-bottom:0; padding:6px 12px; font-size:12.5px; background:#fff;">
                            <span style="font-size:12px; color:var(--text-muted); font-weight:600;">Filter:</span>
                            <button class="filter-btn active" id="filterAll" onclick="setSentimentFilter('All')">All</button>
                            <button class="filter-btn" id="filterPos" onclick="setSentimentFilter('Positive')">Positive</button>
                            <button class="filter-btn" id="filterNeg" onclick="setSentimentFilter('Negative')">Negative</button>
                            <button class="filter-btn" id="filterNeu" onclick="setSentimentFilter('Neutral')">Neutral</button>
                        </div>
                        <div style="display:flex; gap:10px;">
                            <button class="btn-link" onclick="downloadBatchCSV()" title="Download Results CSV">
                                📥 Download CSV
                            </button>
                            <button class="btn-link" onclick="downloadBatchPDF()" title="Download PDF Report">
                                📄 Export PDF Report
                            </button>
                        </div>
                    </div>

                    <!-- Batch Results Table -->
                    <div style="overflow-x: auto; background:#fff; border:1px solid var(--card-border); border-radius:10px;">
                        <table>
                            <thead>
                                <tr>
                                    <th style="width:60px;"># / ID</th>
                                    <th>Review Text Snippet</th>
                                    <th style="width:90px;">Sentiment</th>
                                    <th style="width:105px;">ML Conf (LR)</th>
                                    <th style="width:100px;">VADER Compound</th>
                                    <th style="width:80px;">TextBlob</th>
                                    <th style="width:90px;">ML (LR)</th>
                                    <th style="width:110px;">Aspects</th>
                                    <th style="width:80px;">Priority</th>
                                    <th style="width:60px;">Details</th>
                                </tr>
                            </thead>
                            <tbody id="batchTableBody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- 6-Stage NLP Pipeline Architecture Diagram -->
        <div class="card">
            <h2>🧪 System Architecture: The 6-Stage NLP & ML Pipeline</h2>
            <div class="pipeline-container">
                <div class="pipeline-step">
                    <h4>1. Input Ingestion</h4>
                    <p>Single text, .DOCX, .PDF, .CSV, .XLSX</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>2. Clean & Tokenize</h4>
                    <p>Lowercasing, Negation-preserving regex</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>3. Morphological Lemmas</h4>
                    <p>WordNet grammatical root resolution</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>4. Lexicon Polarity</h4>
                    <p>NLTK VADER valence scoring</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>5. TF-IDF & ML Inference</h4>
                    <p>Logistic Regression & Naive Bayes</p>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <h4>6. Consensus & Action</h4>
                    <p>Resolved Sentiment, ABSA & Exports</p>
                </div>
            </div>
        </div>

        <!-- History Trends -->
        <div class="card">
            <h2>
                <span>📈 Historical Sentiment Trends (Recent Analyses)</span>
                <div style="display:flex; gap:10px;">
                    <a href="/api/export-history-csv" class="btn-secondary" style="text-decoration:none; display:inline-flex; align-items:center; gap:4px;">
                        📥 Export CSV
                    </a>
                    <button class="btn-secondary" onclick="loadHistory()">🔄 Refresh History</button>
                </div>
            </h2>
            <div style="height: 180px; width: 100%; margin-bottom: 20px;">
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

        <!-- Confusion Matrix & Benchmark -->
        <div class="card">
            <h2>🎓 Scikit-Learn TF-IDF Benchmark & Confusion Matrix</h2>
            <p style="color:var(--text-muted); font-size:13.5px; margin-bottom:14px;">
                Trained on balanced n-gram TF-IDF vectors with negation preserving tokenization.
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

            <h3 style="font-size:14px; color:#1e293b; margin-top:16px;">🧮 3x3 Multi-Class Confusion Matrix (Logistic Regression on Test Split):</h3>
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
        </div>
    </div>

    <!-- Review Detail Modal -->
    <div class="modal-overlay" id="reviewDetailModal">
        <div class="modal-box">
            <div class="modal-header">
                <h3 id="modalReviewTitle">Review Details</h3>
                <button class="close-modal-btn" onclick="closeReviewModal()">✕</button>
            </div>
            <div id="modalReviewBody" style="font-size:13.5px; line-height:1.7; color:#334155;"></div>
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
                <h4 style="color:#4f46e5; margin-bottom:4px;">1. Multi-Engine Sentiment Consensus</h4>
                <p>The system resolves primary sentiment using a consensus engine combining VADER lexicon scores, TextBlob subjectivity/polarity, and supervised TF-IDF machine learning models. High-confidence indicators take precedence over neutral defaults.</p>
                <br>
                <h4 style="color:#4f46e5; margin-bottom:4px;">2. Multi-Format Document Ingestion</h4>
                <p>Supports DOCX, DOC, PDF, CSV, XLSX, XLS, and TXT. Multi-page PDFs, Word document tables, and Excel sheets are parsed systematically in natural reading order without truncating documents.</p>
                <br>
                <h4 style="color:#4f46e5; margin-bottom:4px;">3. VADER Compound Score Computation</h4>
                <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:10px 14px; margin:8px 0; font-family:'Courier New', monospace; font-size:13px; font-weight:600;">
                    Compound Score (c) = x / √(x² + α)
                </div>
                <p style="font-size:12.5px; color:#64748b;">Where x = sum of valence scores of lexicon tokens, and α = normalization parameter (default ≈ 15).</p>
            </div>
        </div>
    </div>

    <script>
        let pieChartInstance = null;
        let trendChartInstance = null;
        let batchPieChartInstance = null;
        let batchBarChartInstance = null;
        let speechRecognition = null;
        let isListening = false;
        let currentTokens = [];
        let currentBatchData = null;
        let activeSentimentFilter = 'All';

        const PRESETS = {
            "samsung_a55": "The Samsung Galaxy A55 has a beautiful display and the camera takes excellent photos. The battery lasts all day during normal usage. However, the phone heats up during gaming and the charging speed is disappointing. The design is premium and the phone feels comfortable to hold. The device comes with a 6.5-inch display and 128 GB of storage.",
            "tech_pos": "The battery life is phenomenal and the screen resolution is gorgeous! Best gadget I bought this entire year.",
            "tech_neg": "Do not buy this laptop! It overheats within 15 minutes of usage, constantly crashes, and has cheap plastic build.",
            "food_pos": "Outstanding food quality, delivery arrived in just 20 minutes hot and fresh. Loved the customer service!",
            "ecom_neg": "Horrible delivery delay! Package took 3 weeks to arrive and arrived completely broken inside with missing items.",
            "spec_neu": "The item dimensions are 10cm by 15cm. Package includes one standard USB-C cable and user manual.",
            "support_neg": "Customer support is completely unresponsive and extremely rude. Refused to issue a refund for defective unit.",
            "negation_pos": "Not bad at all, actually quite good and durable for the price!",
            "negation_neg": "The camera is not good and battery is not worth the price."
        };

        let currentSingleSentences = [];
        let activeSingleSentenceFilter = 'All';

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

        function closeReviewModal() {
            document.getElementById('reviewDetailModal').style.display = 'none';
        }

        function openReviewModal(idx) {
            if (!currentBatchData || !currentBatchData.detailed_results || !currentBatchData.detailed_results[idx]) return;
            const r = currentBatchData.detailed_results[idx];
            document.getElementById('modalReviewTitle').innerText = `Review Details (${r.source_id || '#' + (idx + 1)})`;
            
            const sentColor = r.sentiment === 'Positive' ? '#15803d' : (r.sentiment === 'Negative' ? '#b91c1c' : '#b45309');
            
            let sentencesHtml = '';
            if (r.sentence_analysis && r.sentence_analysis.sentences && r.sentence_analysis.sentences.length > 0) {
                const sents = r.sentence_analysis.sentences;
                sentencesHtml = `
                    <div style="margin-top:14px;">
                        <strong style="color:#1e293b;">Sentence-by-Sentence Sentiment (${sents.length} sentences):</strong>
                        <div style="max-height:200px; overflow-y:auto; border:1px solid #e2e8f0; border-radius:6px; margin-top:6px;">
                            <table style="margin:0; font-size:12px;">
                                <thead>
                                    <tr>
                                        <th style="width:30px;">#</th>
                                        <th>Sentence Text</th>
                                        <th style="width:75px;">Sentiment</th>
                                        <th style="width:65px;">Compound</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${sents.map((s, si) => `
                                        <tr class="${s.sentiment === 'Positive' ? 'sent-row-pos' : (s.sentiment === 'Negative' ? 'sent-row-neg' : 'sent-row-neu')}">
                                            <td>${si + 1}</td>
                                            <td>${s.text}</td>
                                            <td><span class="badge ${s.sentiment === 'Positive' ? 'badge-pos' : (s.sentiment === 'Negative' ? 'badge-neg' : 'badge-neu')}" style="font-size:10px; padding:2px 8px;">${s.sentiment}</span></td>
                                            <td style="font-family:monospace;">${s.vader_compound !== undefined ? (s.vader_compound >= 0 ? '+' : '') + s.vader_compound.toFixed(4) : '0.0000'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }

            const confDisplay = (r.confidence_percentage !== null && r.confidence_percentage !== undefined)
                ? `${r.confidence_percentage}%`
                : 'N/A';
            const vaderCompVal = (r.overall_compound_score !== undefined)
                ? `${r.overall_compound_score >= 0 ? '+' : ''}${Number(r.overall_compound_score).toFixed(4)}`
                : ((r.vader_scores && r.vader_scores.compound_score !== undefined)
                    ? `${r.vader_scores.compound_score >= 0 ? '+' : ''}${Number(r.vader_scores.compound_score).toFixed(4)}`
                    : 'N/A');

            document.getElementById('modalReviewBody').innerHTML = `
                <div style="margin-bottom:12px;">
                    <strong>Full Review Text:</strong>
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; margin-top:6px; font-size:13.5px;">
                        ${r.original_review}
                    </div>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
                    <div><strong>Resolved Sentiment:</strong> <span style="color:${sentColor}; font-weight:700;">${r.sentiment}</span></div>
                    <div><strong>ML Model Confidence (LR):</strong> <strong style="color:#4f46e5;">${confDisplay}</strong></div>
                    <div><strong>Urgency Priority:</strong> <span style="font-weight:700; color:${r.priority === 'High' ? '#dc2626' : (r.priority === 'Medium' ? '#d97706' : '#059669')}">${r.priority}</span></div>
                    <div><strong>VADER Compound:</strong> <span style="font-family:monospace; font-weight:700;">${vaderCompVal}</span></div>
                    <div><strong>TextBlob Polarity:</strong> ${r.textblob_metrics ? r.textblob_metrics.polarity : 'N/A'}</div>
                    <div><strong>ML (Logistic Regression):</strong> ${r.ml_model_predictions ? r.ml_model_predictions.logistic_regression : 'N/A'}</div>
                    <div><strong>ML (Naive Bayes):</strong> ${r.ml_model_predictions ? r.ml_model_predictions.naive_bayes : 'N/A'}</div>
                </div>
                <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:10px 12px; margin-bottom:12px; font-size:12.5px; color:#1e40af;">
                    <strong>Explanation:</strong> ${r.explanation || 'N/A'}
                </div>
                <div><strong>Aspects:</strong> ${(r.categories && r.categories.length) ? r.categories.join(', ') : 'General'}</div>
                ${sentencesHtml}
            `;
            document.getElementById('reviewDetailModal').style.display = 'flex';
        }

        function toggleSpeechRecognition() {
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRec) {
                alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
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

        // ==========================================
        // SINGLE REVIEW HANDLER & SENTENCE ANALYTICS
        // ==========================================
        async function triggerAnalyze() {
            const text = document.getElementById('reviewInput').value.trim();
            const errAlert = document.getElementById('singleErrorAlert');
            if (errAlert) errAlert.style.display = 'none';

            if (!text) {
                if (errAlert) {
                    errAlert.style.display = 'block';
                    errAlert.innerText = 'Please enter or paste a review before analyzing.';
                } else {
                    alert('Please enter or paste a review before analyzing.');
                }
                return;
            }

            const threshold = parseFloat(document.getElementById('thresholdSlider').value);
            const spinner = document.getElementById('analyzeSpinner');
            const btnText = document.getElementById('analyzeBtnText');
            const btn = document.getElementById('analyzeBtn');

            spinner.style.display = 'inline-block';
            btnText.innerText = 'Analyzing Sentences...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ review: text, threshold: threshold })
                });

                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || 'Analysis failed');
                }

                renderSingleResults(data);
                loadHistory();
            } catch (err) {
                if (errAlert) {
                    errAlert.style.display = 'block';
                    errAlert.innerText = 'Analysis Error: ' + err.message;
                } else {
                    alert('Analysis Error: ' + err.message);
                }
            } finally {
                spinner.style.display = 'none';
                btnText.innerText = '⚡ Analyze Sentiment';
                btn.disabled = false;
            }
        }

        function renderSingleResults(data) {
            document.getElementById('emptyPlaceholder').style.display = 'none';
            document.getElementById('resultContent').style.display = 'block';
            document.getElementById('statusIndicator').innerText = 'Analysis Complete';

            const sa = data.sentence_analysis || {};
            const dist = sa.sentiment_distribution || {};
            const overallSent = data.sentiment || sa.overall_sentiment || 'Neutral';
            const meanComp = sa.overall_compound_score !== undefined ? sa.overall_compound_score : (data.vader_scores ? data.vader_scores.compound_score : 0.0);
            const thresh = sa.threshold !== undefined ? sa.threshold : (data.vader_scores && data.vader_scores.threshold !== undefined ? data.vader_scores.threshold : 0.05);

            // 1. Overall Sentiment Card
            const badgeContainer = document.getElementById('sentimentBadgeContainer');
            let badgeClass = 'badge-neu';
            if (overallSent === 'Positive') badgeClass = 'badge-pos';
            if (overallSent === 'Negative') badgeClass = 'badge-neg';
            badgeContainer.innerHTML = `<span class="badge ${badgeClass}">${overallSent}</span>`;

            document.getElementById('meanCompoundVal').innerText = (meanComp >= 0 ? '+' : '') + meanComp.toFixed(4);
            document.getElementById('selectedThresholdVal').innerText = '±' + thresh.toFixed(2);
            
            const prio = document.getElementById('priorityVal');
            prio.innerText = data.priority;
            prio.style.color = data.priority === 'High' ? '#dc2626' : (data.priority === 'Medium' ? '#d97706' : '#059669');

            const confEl = document.getElementById('singleConfidenceVal');
            if (confEl) {
                confEl.innerText = (data.confidence_percentage !== null && data.confidence_percentage !== undefined)
                    ? `${data.confidence_percentage}%`
                    : 'N/A';
            }

            document.getElementById('overallCalcExplanation').innerHTML = `<strong>💡 Overall Sentiment Calculation:</strong> ${sa.explanation || data.explanation}`;

            // 2. Sentence Summary Cards
            const totalSents = sa.total_sentences || 1;
            const posC = dist.Positive !== undefined ? dist.Positive : (overallSent === 'Positive' ? 1 : 0);
            const negC = dist.Negative !== undefined ? dist.Negative : (overallSent === 'Negative' ? 1 : 0);
            const neuC = dist.Neutral !== undefined ? dist.Neutral : (overallSent === 'Neutral' ? 1 : 0);
            const posP = dist.Positive_Percentage !== undefined ? dist.Positive_Percentage : (overallSent === 'Positive' ? 100.0 : 0.0);
            const negP = dist.Negative_Percentage !== undefined ? dist.Negative_Percentage : (overallSent === 'Negative' ? 100.0 : 0.0);
            const neuP = dist.Neutral_Percentage !== undefined ? dist.Neutral_Percentage : (overallSent === 'Neutral' ? 100.0 : 0.0);

            document.getElementById('totalSentencesVal').innerText = totalSents;
            document.getElementById('posSentencesVal').innerText = `${posC} (${posP}%)`;
            document.getElementById('negSentencesVal').innerText = `${negC} (${negP}%)`;
            document.getElementById('neuSentencesVal').innerText = `${neuC} (${neuP}%)`;

            // 3. Sentence Distribution Doughnut Chart
            renderSentimentDoughnut(posC, negC, neuC);

            // 4. Model Comparison
            const mc = data.model_comparison || {};
            document.getElementById('cmpVader').innerText = mc.vader || 'N/A';
            document.getElementById('cmpLR').innerText = mc.logistic_regression || 'N/A';
            document.getElementById('cmpNB').innerText = (data.ml_model_predictions && data.ml_model_predictions.naive_bayes) ? data.ml_model_predictions.naive_bayes : 'N/A';
            document.getElementById('cmpTB').innerText = mc.textblob || 'N/A';

            const disAlert = document.getElementById('disagreementAlert');
            if (mc.has_disagreement) {
                disAlert.style.display = 'block';
                disAlert.innerHTML = `⚠️ <strong>Model Disagreement:</strong> ${mc.disagreement_details ? mc.disagreement_details.join(' | ') : 'Minor variation between rule-based & ML.'}`;
            } else {
                disAlert.style.display = 'none';
            }

            const recAlert = document.getElementById('recommendationAlert');
            recAlert.innerHTML = `<strong>Action Recommendation:</strong> ${data.recommendation}`;

            // 5. Populate Sentence-Level Results Table
            currentSingleSentences = sa.sentences || [];
            if (!currentSingleSentences.length) {
                currentSingleSentences = [{
                    id: 1,
                    sentence_id: 'S1',
                    text: data.original_review,
                    sentiment: overallSent,
                    vader_compound: meanComp,
                    textblob_polarity: data.textblob_metrics ? data.textblob_metrics.polarity : 0.0,
                    ml_prediction: (data.ml_model_predictions && data.ml_model_predictions.logistic_regression) ? data.ml_model_predictions.logistic_regression : 'N/A'
                }];
            }
            populateSingleSentenceTable(currentSingleSentences);

            // 6. Word Polarity Highlighting & Aspect Cloud
            currentTokens = data.word_tokens || [];
            renderWordHighlights(document.getElementById('stopwordToggle').checked);

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
        }

        function setSentenceTableFilter(filter) {
            activeSingleSentenceFilter = filter;
            ['sentFilterAll', 'sentFilterPos', 'sentFilterNeg', 'sentFilterNeu'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active');
            });
            if (filter === 'All' && document.getElementById('sentFilterAll')) document.getElementById('sentFilterAll').classList.add('active');
            if (filter === 'Positive' && document.getElementById('sentFilterPos')) document.getElementById('sentFilterPos').classList.add('active');
            if (filter === 'Negative' && document.getElementById('sentFilterNeg')) document.getElementById('sentFilterNeg').classList.add('active');
            if (filter === 'Neutral' && document.getElementById('sentFilterNeu')) document.getElementById('sentFilterNeu').classList.add('active');
            filterSingleSentenceTable();
        }

        function filterSingleSentenceTable() {
            const rows = document.querySelectorAll('#sentenceTableBody tr');
            rows.forEach(row => {
                const s = row.getAttribute('data-sentiment');
                row.style.display = (activeSingleSentenceFilter === 'All' || s === activeSingleSentenceFilter) ? '' : 'none';
            });
        }

        function populateSingleSentenceTable(sentences) {
            const tbody = document.getElementById('sentenceTableBody');
            if (!sentences || !sentences.length) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted);">No individual sentences to display.</td></tr>';
                return;
            }
            tbody.innerHTML = sentences.map((s, idx) => {
                let rowClass = 'sent-row-neu';
                let badgeClass = 'badge-neu';
                if (s.sentiment === 'Positive') { rowClass = 'sent-row-pos'; badgeClass = 'badge-pos'; }
                if (s.sentiment === 'Negative') { rowClass = 'sent-row-neg'; badgeClass = 'badge-neg'; }

                const comp = s.vader_compound !== undefined ? (s.vader_compound >= 0 ? '+' : '') + s.vader_compound.toFixed(4) : (s.vader_scores && s.vader_scores.compound !== undefined ? s.vader_scores.compound : '0.0000');
                const tbPol = s.textblob_polarity !== undefined ? (s.textblob_polarity >= 0 ? '+' : '') + s.textblob_polarity.toFixed(2) : '0.00';
                const ml = s.ml_prediction || (s.ml_predictions && s.ml_predictions.logistic_regression) || 'N/A';

                return `
                    <tr class="${rowClass}" data-sentiment="${s.sentiment}">
                        <td style="font-weight:700; color:#4338ca;">${s.id || (idx + 1)}</td>
                        <td style="font-size:13px; color:#1e293b; line-height:1.5;">${s.text}</td>
                        <td><span class="badge ${badgeClass}">${s.sentiment}</span></td>
                        <td style="font-family:monospace; font-weight:700;">${comp}</td>
                        <td style="font-family:monospace;">${tbPol}</td>
                        <td><span style="font-size:11.5px; font-weight:600;">${ml}</span></td>
                    </tr>
                `;
            }).join('');
            filterSingleSentenceTable();
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
            if (pieChartInstance) pieChartInstance.destroy();

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
                    plugins: { legend: { display: false } },
                    cutout: '65%'
                }
            });
        }

        // ==========================================
        // BATCH & FILE UPLOAD HANDLERS
        // ==========================================
        async function uploadMultiFormatFile() {
            const fileInput = document.getElementById('multiFileInput');
            const errAlert = document.getElementById('batchErrorAlert');
            if (errAlert) errAlert.style.display = 'none';

            if (!fileInput.files.length) {
                if (errAlert) {
                    errAlert.style.display = 'block';
                    errAlert.innerText = 'Please select a supported file (.docx, .doc, .pdf, .csv, .xlsx, .xls, .txt) to upload.';
                } else {
                    alert('Please select a file to upload.');
                }
                return;
            }

            const file = fileInput.files[0];
            const mode = document.getElementById('docModeSelect').value || 'auto';
            const colName = document.getElementById('csvColumnInput').value.trim();
            const threshold = document.getElementById('thresholdSlider') ? parseFloat(document.getElementById('thresholdSlider').value) : 0.05;

            const formData = new FormData();
            formData.append('file', file);

            let url = `/api/upload-document?threshold=${threshold}&mode=${mode}`;
            if (colName) url += `&column_name=${encodeURIComponent(colName)}`;

            const spinner = document.getElementById('batchUploadSpinner');
            const btnText = document.getElementById('uploadBtnText');
            const btn = document.getElementById('uploadFileBtn');

            spinner.style.display = 'inline-block';
            btnText.innerText = 'Processing File...';
            btn.disabled = true;

            try {
                const res = await fetch(url, { method: 'POST', body: formData });
                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || 'File processing failed');
                }
                currentBatchData = data;
                renderBatchDashboard(data);
            } catch (err) {
                if (errAlert) {
                    errAlert.style.display = 'block';
                    errAlert.innerHTML = `<strong>Document Processing Error:</strong> ${err.message}`;
                } else {
                    alert('Error: ' + err.message);
                }
            } finally {
                spinner.style.display = 'none';
                btnText.innerText = '📁 Upload & Process File';
                btn.disabled = false;
            }
        }

        async function triggerPastedBatchAnalyze() {
            const raw = document.getElementById('batchInput').value.trim();
            const errAlert = document.getElementById('batchErrorAlert');
            if (errAlert) errAlert.style.display = 'none';

            if (!raw) {
                if (errAlert) {
                    errAlert.style.display = 'block';
                    errAlert.innerText = 'Please enter or paste at least one review in the text area.';
                } else {
                    alert('Please paste reviews to process.');
                }
                return;
            }

            const mode = document.getElementById('pastedModeSelect').value || 'auto';
            const threshold = document.getElementById('thresholdSlider') ? parseFloat(document.getElementById('thresholdSlider').value) : 0.05;

            const spinner = document.getElementById('batchPasteSpinner');
            const btnText = document.getElementById('processPastedBtnText');
            const btn = document.getElementById('processPastedBtn');

            spinner.style.display = 'inline-block';
            btnText.innerText = 'Processing Reviews...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/analyze-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: raw, threshold: threshold, mode: mode })
                });

                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Batch analysis failed');
                currentBatchData = data;
                renderBatchDashboard(data);
            } catch (err) {
                if (errAlert) {
                    errAlert.style.display = 'block';
                    errAlert.innerHTML = `<strong>Batch Processing Error:</strong> ${err.message}`;
                } else {
                    alert(err.message);
                }
            } finally {
                spinner.style.display = 'none';
                btnText.innerText = '⚡ Process All Reviews in Bulk';
                btn.disabled = false;
            }
        }

        function renderBatchDashboard(data) {
            document.getElementById('batchResultsContainer').style.display = 'block';

            // 1. Diagnostics card
            const diagCard = document.getElementById('batchDiagnosticsCard');
            if (data.diagnostics && diagCard) {
                diagCard.style.display = 'block';
                const d = data.diagnostics;
                document.getElementById('diagFilename').innerText = d.filename || 'Pasted text';
                const badge = document.getElementById('diagFormatBadge');
                if (badge) badge.innerText = (d.file_extension || 'BATCH').replace('.', '').toUpperCase();
                document.getElementById('diagSize').innerText = d.file_size_formatted || (d.file_size_bytes ? d.file_size_bytes + ' B' : 'N/A');
                document.getElementById('diagChars').innerText = d.character_count || 'N/A';
                document.getElementById('diagWords').innerText = d.word_count || 'N/A';
                document.getElementById('diagIdentified').innerText = `${d.reviews_identified || data.total_reviews} reviews`;
                document.getElementById('diagMode').innerText = d.processing_mode || 'auto';
                document.getElementById('diagPreview').innerText = d.preview || 'Processed reviews successfully.';

                const warnAlert = document.getElementById('diagWarningAlert');
                if (d.warnings && d.warnings.length) {
                    warnAlert.style.display = 'block';
                    warnAlert.innerHTML = `⚠️ <strong>Notice:</strong> ${d.warnings.join(' | ')}`;
                } else {
                    warnAlert.style.display = 'none';
                }
            } else if (diagCard) {
                diagCard.style.display = 'none';
            }

            // 1b. Document-wide sentence distribution banner
            const sentBanner = document.getElementById('batchSentenceSummaryBanner');
            if (sentBanner && data.sentence_distribution && data.sentence_distribution.total_sentences > 0) {
                const sd = data.sentence_distribution;
                sentBanner.style.display = 'block';
                document.getElementById('batchSentTotalCount').innerText = `${sd.total_sentences} sentences`;
                document.getElementById('batchSentPosCount').innerText = `${sd.positive_sentences || 0} (${sd.positive_percentage || 0}%)`;
                document.getElementById('batchSentNegCount').innerText = `${sd.negative_sentences || 0} (${sd.negative_percentage || 0}%)`;
                document.getElementById('batchSentNeuCount').innerText = `${sd.neutral_sentences || 0} (${sd.neutral_percentage || 0}%)`;
                const meanScore = Number(sd.mean_compound_score || 0);
                document.getElementById('batchSentMeanScore').innerText = `${meanScore >= 0 ? '+' : ''}${meanScore.toFixed(4)}`;
            } else if (sentBanner) {
                sentBanner.style.display = 'none';
            }

            // 2. Scorecard tiles with percentages
            const dist = data.sentiment_distribution || {};
            const posPct = dist.Positive_Percentage !== undefined ? dist.Positive_Percentage : 0;
            const negPct = dist.Negative_Percentage !== undefined ? dist.Negative_Percentage : 0;
            const neuPct = dist.Neutral_Percentage !== undefined ? dist.Neutral_Percentage : 0;

            document.getElementById('batchTotalCount').innerText = data.total_reviews || 0;
            document.getElementById('batchPosCount').innerText = `${dist.Positive || 0} (${posPct}%)`;
            document.getElementById('batchNegCount').innerText = `${dist.Negative || 0} (${negPct}%)`;
            document.getElementById('batchNeuCount').innerText = `${dist.Neutral || 0} (${neuPct}%)`;
            document.getElementById('batchFailedCount').innerText = (data.diagnostics && data.diagnostics.failed_records) ? data.diagnostics.failed_records : 0;

            // 3. Render Batch Doughnut & Bar Charts
            renderBatchCharts(dist.Positive || 0, dist.Negative || 0, dist.Neutral || 0);

            // 4. Aspect Breakdown
            const aspectContainer = document.getElementById('batchAspectTags');
            if (aspectContainer && data.aspect_distribution) {
                const aspectEntries = Object.entries(data.aspect_distribution);
                if (aspectEntries.length) {
                    aspectContainer.innerHTML = aspectEntries.map(([k, v]) => `
                        <div class="bubble-tag">
                            <strong>🏷️ ${k}:</strong> <span>${v} review${v > 1 ? 's' : ''}</span>
                        </div>
                    `).join('');
                } else {
                    aspectContainer.innerHTML = '<span style="font-size:12.5px; color:var(--text-muted);">No specific domain aspects identified.</span>';
                }
            }

            // 5. Populate Results Table
            populateBatchTable(data.detailed_results || []);
        }

        function renderBatchCharts(pos, neg, neu) {
            // Doughnut Chart
            const pieCtx = document.getElementById('batchPieChart').getContext('2d');
            if (batchPieChartInstance) batchPieChartInstance.destroy();
            batchPieChartInstance = new Chart(pieCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{
                        data: [pos || 0.001, neg || 0.001, neu || 0.001],
                        backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10.5 } } }
                    },
                    cutout: '60%'
                }
            });

            // Bar Chart
            const barCtx = document.getElementById('batchBarChart').getContext('2d');
            if (batchBarChartInstance) batchBarChartInstance.destroy();
            batchBarChartInstance = new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{
                        label: 'Review Count',
                        data: [pos, neg, neu],
                        backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { precision: 0 } }
                    }
                }
            });
        }

        function populateBatchTable(reviews) {
            const tbody = document.getElementById('batchTableBody');
            tbody.innerHTML = reviews.map((r, i) => {
                let badgeClass = 'badge-neu';
                if (r.sentiment === 'Positive') badgeClass = 'badge-pos';
                if (r.sentiment === 'Negative') badgeClass = 'badge-neg';

                const mlPred = (r.ml_model_predictions && r.ml_model_predictions.logistic_regression) ? r.ml_model_predictions.logistic_regression : 'N/A';
                const tbPol = (r.textblob_metrics && r.textblob_metrics.polarity !== undefined) ? r.textblob_metrics.polarity : 'N/A';
                const vaderComp = (r.overall_compound_score !== undefined && r.overall_compound_score !== null)
                    ? (r.overall_compound_score >= 0 ? '+' : '') + Number(r.overall_compound_score).toFixed(4)
                    : (r.vader_scores && r.vader_scores.compound_score !== undefined
                        ? (r.vader_scores.compound_score >= 0 ? '+' : '') + Number(r.vader_scores.compound_score).toFixed(4)
                        : 'N/A');
                const confHtml = (r.confidence_percentage !== null && r.confidence_percentage !== undefined)
                    ? `<strong>${r.confidence_percentage}%</strong>`
                    : `<span style="color:var(--text-muted);">N/A</span>`;
                const categories = (r.categories && r.categories.length) ? r.categories.join(', ') : 'General';
                const safeText = (r.original_review || '').replace(/"/g, '&quot;');
                const snippet = r.original_review.length > 75 ? (r.original_review.substring(0, 72) + '...') : r.original_review;

                return `
                    <tr data-sentiment="${r.sentiment}" data-text="${safeText.toLowerCase()}">
                        <td style="font-weight:700; color:#3730a3;">${r.source_id || (i + 1)}</td>
                        <td title="Click to view details" onclick="openReviewModal(${i})" style="cursor:pointer;">
                            <span style="color:#1e293b; text-decoration:underline dotted;">${snippet}</span>
                        </td>
                        <td><span class="badge ${badgeClass}">${r.sentiment}</span></td>
                        <td>${confHtml}</td>
                        <td>${vaderComp}</td>
                        <td>${tbPol}</td>
                        <td>${mlPred}</td>
                        <td><span style="font-size:11.5px; color:var(--text-muted);">${categories}</span></td>
                        <td><span style="font-weight:600; color:${r.priority === 'High' ? '#dc2626' : (r.priority === 'Medium' ? '#d97706' : '#059669')}">${r.priority}</span></td>
                        <td>
                            <button class="btn-secondary" style="padding:3px 8px; font-size:11px;" onclick="openReviewModal(${i})">View</button>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        function setSentimentFilter(sentiment) {
            activeSentimentFilter = sentiment;
            ['filterAll', 'filterPos', 'filterNeg', 'filterNeu'].forEach(id => {
                document.getElementById(id).classList.remove('active');
            });
            if (sentiment === 'All') document.getElementById('filterAll').classList.add('active');
            if (sentiment === 'Positive') document.getElementById('filterPos').classList.add('active');
            if (sentiment === 'Negative') document.getElementById('filterNeg').classList.add('active');
            if (sentiment === 'Neutral') document.getElementById('filterNeu').classList.add('active');
            filterBatchTable();
        }

        function filterBatchTable() {
            const query = (document.getElementById('batchSearchInput').value || '').toLowerCase().trim();
            const rows = document.querySelectorAll('#batchTableBody tr');
            rows.forEach(row => {
                const s = row.getAttribute('data-sentiment');
                const t = row.getAttribute('data-text');
                const matchesSentiment = (activeSentimentFilter === 'All' || s === activeSentimentFilter);
                const matchesQuery = (!query || t.includes(query));
                row.style.display = (matchesSentiment && matchesQuery) ? '' : 'none';
            });
        }

        // ==========================================
        // EXPORT CSV & PDF
        // ==========================================
        async function downloadBatchCSV() {
            if (!currentBatchData || !currentBatchData.detailed_results) {
                alert('No batch analysis data to export. Please process a document or reviews first.');
                return;
            }

            try {
                const res = await fetch('/api/export-batch-csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentBatchData)
                });
                if (!res.ok) throw new Error('Failed to generate CSV export');

                const blob = await res.blob();
                const now = new Date();
                const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 14);
                const filename = `sentiment_results_${ts}.csv`;

                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            } catch (err) {
                alert('Export Error: ' + err.message);
            }
        }

        async function downloadBatchPDF() {
            if (!currentBatchData || !currentBatchData.detailed_results) {
                alert('No batch analysis data to export. Please process a document or reviews first.');
                return;
            }

            try {
                const res = await fetch('/api/export-batch-pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentBatchData)
                });
                if (!res.ok) throw new Error('Failed to generate PDF report');

                const blob = await res.blob();
                const now = new Date();
                const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 14);
                const filename = `sentiment_results_${ts}.pdf`;

                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            } catch (err) {
                alert('PDF Export Error: ' + err.message);
            }
        }

        // ==========================================
        // METRICS & HISTORY
        // ==========================================
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

        async function loadHistory() {
            try {
                const res = await fetch('/api/history?limit=15');
                const data = await res.json();
                const tbody = document.getElementById('historyTableBody');

                if (!data.history || !data.history.length) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">No records found.</td></tr>';
                    return;
                }

                tbody.innerHTML = data.history.map(row => {
                    let badgeClass = 'badge-neu';
                    if (row.sentiment === 'Positive') badgeClass = 'badge-pos';
                    if (row.sentiment === 'Negative') badgeClass = 'badge-neg';

                    const snippet = row.review.length > 55 ? (row.review.substring(0, 52) + '...') : row.review;
                    return `
                        <tr>
                            <td>#${row.id}</td>
                            <td title="${row.review.replace(/"/g, '&quot;')}">${snippet}</td>
                            <td><span class="badge ${badgeClass}">${row.sentiment}</span></td>
                            <td>${row.confidence}%</td>
                            <td>${row.vader_compound}</td>
                            <td><span style="font-weight:600; color:${row.priority === 'High' ? '#dc2626' : (row.priority === 'Medium' ? '#d97706' : '#059669')}">${row.priority}</span></td>
                            <td style="font-size:11.5px; color:var(--text-muted);">${row.created_at || 'Just now'}</td>
                        </tr>
                    `;
                }).join('');

                renderTrendChart(data.history);
            } catch (err) {
                console.error(err);
            }
        }

        function renderTrendChart(history) {
            const ctx = document.getElementById('trendLineChart').getContext('2d');
            if (trendChartInstance) trendChartInstance.destroy();

            const rev = [...history].reverse();
            trendChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: rev.map((r, i) => `#${r.id}`),
                    datasets: [{
                        label: 'VADER Compound Score',
                        data: rev.map(r => r.vader_compound),
                        borderColor: '#4f46e5',
                        backgroundColor: 'rgba(79, 70, 229, 0.08)',
                        tension: 0.35,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { min: -1.0, max: 1.0 }
                    }
                }
            });
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
