import os
import sqlite3
import json
import csv
import io
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse, FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import joblib

from nlp_engine import complete_review_analysis, clean_and_tokenize
from ml_trainer import train_and_evaluate_models
from document_parser import validate_and_extract_document, split_into_reviews, DocumentParsingError
from pdf_exporter import generate_batch_pdf_report

logger = logging.getLogger("smart_review_analyzer.main")

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

@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(HTTPException)
async def fastapi_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_db_path() -> str:
    env_path = os.environ.get("SQLITE_DB_PATH")
    if env_path:
        return os.path.abspath(env_path)
    if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return "/tmp/reviews.db"
    return os.path.abspath(os.path.join(BASE_DIR, "reviews.db"))

DB_PATH = get_db_path()

def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    target_path = db_path or get_db_path()
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_sqlite_db(db_path: Optional[str] = None):
    target_path = db_path or get_db_path()
    try:
        dir_name = os.path.dirname(target_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with sqlite3.connect(target_path) as conn:
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vader_pos REAL DEFAULT 0.0,
                    vader_neg REAL DEFAULT 0.0,
                    vader_neu REAL DEFAULT 0.0,
                    source TEXT DEFAULT 'single'
                )
            """)
            conn.commit()

            cursor.execute("PRAGMA table_info(reviews_history)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            columns_to_add = [
                ("vader_pos", "REAL DEFAULT 0.0"),
                ("vader_neg", "REAL DEFAULT 0.0"),
                ("vader_neu", "REAL DEFAULT 0.0"),
                ("source", "TEXT DEFAULT 'single'")
            ]
            for col_name, col_def in columns_to_add:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE reviews_history ADD COLUMN {col_name} {col_def}")
            conn.commit()
    except Exception as e:
        logger.warning(f"SQLite Init Warning on '{target_path}': {e}")

init_sqlite_db()


def save_review_to_history(
    review_text: str,
    sentiment: str,
    confidence: float,
    categories: Any,
    priority: str,
    vader_compound: float,
    vader_scores: Optional[dict] = None,
    subjectivity: float = 0.0,
    source: str = "single",
    db_path: Optional[str] = None
) -> Optional[int]:
    """
    Persist an analyzed review into SQLite history with duplicate suppression.
    """
    clean_text = review_text.strip() if review_text else ""
    if not clean_text:
        return None

    target_path = db_path or get_db_path()
    try:
        with sqlite3.connect(target_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Prevent duplicate insertion if identical review was saved as immediate latest record
            cur.execute("""
                SELECT id, review, created_at FROM reviews_history
                WHERE review = ?
                ORDER BY id DESC LIMIT 1
            """, (clean_text,))
            last_entry = cur.fetchone()
            if last_entry:
                logger.info(f"Duplicate review detected; skipping re-insertion (matches record #{last_entry['id']})")
                return last_entry["id"]

            v_pos = vader_scores.get("pos", 0.0) if vader_scores else 0.0
            v_neg = vader_scores.get("neg", 0.0) if vader_scores else 0.0
            v_neu = vader_scores.get("neu", 0.0) if vader_scores else 0.0

            cat_str = ", ".join(categories) if isinstance(categories, list) else str(categories or "General")

            cur.execute("""
                INSERT INTO reviews_history (
                    review, sentiment, confidence, categories, priority,
                    vader_compound, subjectivity, vader_pos, vader_neg, vader_neu, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                clean_text,
                sentiment,
                float(confidence or 0.0),
                cat_str,
                priority or "Low",
                float(vader_compound or 0.0),
                float(subjectivity or 0.0),
                float(v_pos or 0.0),
                float(v_neg or 0.0),
                float(v_neu or 0.0),
                source
            ))
            conn.commit()
            return cur.lastrowid
    except Exception as e:
        logger.error(f"Error persisting review to history: {e}", exc_info=True)
        return None


def save_batch_reviews_to_history(
    results: List[dict],
    source: str = "batch_paste",
    db_path: Optional[str] = None
) -> int:
    """
    Persist multiple analyzed reviews into SQLite history in a single transaction.
    """
    if not results:
        return 0

    target_path = db_path or get_db_path()
    inserted_count = 0
    try:
        with sqlite3.connect(target_path) as conn:
            cur = conn.cursor()
            for r in results:
                raw_text = r.get("original_review", "").strip()
                if not raw_text:
                    continue
                v_scores = r.get("vader_scores", {})
                v_comp = r.get("overall_compound_score", v_scores.get("compound_score", 0.0))
                v_pos = v_scores.get("pos", 0.0)
                v_neg = v_scores.get("neg", 0.0)
                v_neu = v_scores.get("neu", 0.0)
                tb = r.get("textblob_metrics", {})
                subj = tb.get("subjectivity", 0.0)
                conf = r.get("confidence_percentage", 0.0)
                cats = ", ".join(r.get("categories", [])) if isinstance(r.get("categories"), list) else "General"
                prio = r.get("priority", "Low")
                sent = r.get("sentiment", "Neutral")

                cur.execute("""
                    INSERT INTO reviews_history (
                        review, sentiment, confidence, categories, priority,
                        vader_compound, subjectivity, vader_pos, vader_neg, vader_neu, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    raw_text, sent, float(conf), cats, prio,
                    float(v_comp), float(subj), float(v_pos), float(v_neg), float(v_neu), source
                ))
                inserted_count += 1
            conn.commit()
            logger.info(f"Persisted {inserted_count} batch reviews into SQLite history (source='{source}')")
    except Exception as e:
        logger.error(f"Error persisting batch reviews to history: {e}", exc_info=True)
    return inserted_count

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
    try:
        result = complete_review_analysis(text, threshold, vectorizer, lr_model, nb_model)
    except Exception as e:
        logger.error(f"Error analyzing review sentiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sentiment analysis engine error: {str(e)}")
    
    try:
        v_scores = result.get("vader_scores") or {}
        tb_metrics = result.get("textblob_metrics") or {}
        conf_val = result.get("confidence_percentage")
        if conf_val is None:
            conf_val = 0.0
        
        save_review_to_history(
            review_text=text,
            sentiment=result.get("sentiment", "Neutral"),
            confidence=float(conf_val),
            categories=result.get("categories", []),
            priority=result.get("priority", "Low"),
            vader_compound=float(result.get("overall_compound_score", v_scores.get("compound_score", 0.0)) or 0.0),
            vader_scores=v_scores,
            subjectivity=float(tb_metrics.get("subjectivity", 0.0) or 0.0),
            source="single"
        )
    except Exception as db_err:
        logger.warning(f"History persistence warning (analysis returned successfully): {db_err}")
        
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
                "extracted_text": text_content,
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

    # Persist analyzed batch reviews to SQLite history
    save_batch_reviews_to_history(results, source="batch_paste")

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
    logger.info(
        f"File upload received: filename='{file.filename}', mode='{mode}', "
        f"column_name='{column_name}', threshold={threshold}"
    )

    if hasattr(file, 'file') and hasattr(file.file, 'read'):
        if hasattr(file.file, 'seek'):
            try:
                file.file.seek(0)
            except Exception:
                pass
        contents = file.file.read()
    else:
        contents = file.read()
        
    try:
        extracted = validate_and_extract_document(file.filename, contents, column_name=column_name)
    except DocumentParsingError as e:
        logger.error(f"DocumentParsingError on '{file.filename}': {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error extracting document '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error processing document: {str(e)}")

    valid_mode = mode if mode in ["auto", "single", "multiple", "paragraph"] else "auto"
    review_items = split_into_reviews(extracted, mode=valid_mode, column_name=column_name)
    
    ext_diag = extracted.get("diagnostics", {})
    col_used = ext_diag.get("review_column_used", "N/A")
    raw_elem_cnt = ext_diag.get("raw_elements_count", 0)
    char_cnt = ext_diag.get("character_count", 0)
    avail_cols = ext_diag.get("available_columns", [])

    logger.info(
        f"Document Extraction: format='{ext_diag.get('file_extension')}', rows/elements={raw_elem_cnt}, "
        f"detected_columns={avail_cols}, selected_column='{col_used}', extracted_chars={char_cnt}"
    )
    logger.info(
        f"Review Segmentation (mode='{valid_mode}'): identified {len(review_items)} candidate review items"
    )

    if not review_items:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Review segmentation identified 0 readable reviews in '{file.filename}'. "
                f"Extracted {char_cnt} characters from column '{col_used}'. "
                f"Available columns: {avail_cols}. Please verify review text column."
            )
        )

    thresh = threshold if threshold is not None else 0.05
    results = []
    failed_records = 0
    last_failure_error = None

    for idx, item in enumerate(review_items, start=1):
        try:
            r_text = item["text"] if isinstance(item, dict) else str(item)
            source_id = item.get("id", f"#{idx}") if isinstance(item, dict) else f"#{idx}"
            
            if not r_text or not r_text.strip():
                failed_records += 1
                last_failure_error = f"Record #{idx} content was empty"
                continue
                
            r_analysis = complete_review_analysis(r_text, thresh, vectorizer, lr_model, nb_model)
            r_analysis["source_id"] = source_id
            results.append(r_analysis)
        except Exception as item_err:
            logger.warning(f"Item #{idx} sentiment analysis warning: {item_err}")
            last_failure_error = str(item_err)
            failed_records += 1

    total = len(results)
    logger.info(
        f"Sentiment Analysis Pipeline: passed={len(review_items)}, "
        f"successfully_analyzed={total}, failed={failed_records}"
    )

    if total == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Document text was extracted from '{file.filename}' ({len(review_items)} candidate reviews), "
                f"but sentiment analysis could not process any valid records. "
                f"Last error: {last_failure_error or 'Empty or unanalyzable content'}"
            )
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

    # Store analyzed document reviews into SQLite history
    save_batch_reviews_to_history(results, source="document_upload")

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
    target_path = get_db_path()
    try:
        with sqlite3.connect(target_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT id, review, sentiment, confidence, categories, priority,
                       vader_compound,
                       COALESCE(vader_pos, 0.0) as vader_pos,
                       COALESCE(vader_neg, 0.0) as vader_neg,
                       COALESCE(vader_neu, 0.0) as vader_neu,
                       subjectivity, created_at,
                       COALESCE(source, 'single') as source
                FROM reviews_history 
                ORDER BY id DESC
            """)
            rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Error reading history for CSV export from '{target_path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error reading history: {str(e)}")

    if not rows:
        raise HTTPException(status_code=404, detail="No historical records found to export.")

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow([
        "ID", "Review Text", "Final Sentiment", "Model Confidence (%)",
        "Aspect Categories", "Priority", "VADER Compound", "VADER Positive",
        "VADER Negative", "VADER Neutral", "TextBlob Subjectivity",
        "Timestamp", "Analysis Source"
    ])
    for r in rows:
        conf_str = f"{r['confidence']:.1f}" if isinstance(r['confidence'], (int, float)) else str(r['confidence'])
        comp_str = f"{r['vader_compound']:.4f}" if isinstance(r['vader_compound'], (int, float)) else str(r['vader_compound'])
        pos_str = f"{r['vader_pos']:.4f}" if isinstance(r['vader_pos'], (int, float)) else str(r['vader_pos'])
        neg_str = f"{r['vader_neg']:.4f}" if isinstance(r['vader_neg'], (int, float)) else str(r['vader_neg'])
        neu_str = f"{r['vader_neu']:.4f}" if isinstance(r['vader_neu'], (int, float)) else str(r['vader_neu'])
        subj_str = f"{r['subjectivity']:.4f}" if isinstance(r['subjectivity'], (int, float)) else str(r['subjectivity'])
        writer.writerow([
            r["id"],
            r["review"],
            r["sentiment"],
            conf_str,
            r["categories"],
            r["priority"],
            comp_str,
            pos_str,
            neg_str,
            neu_str,
            subj_str,
            r["created_at"],
            r["source"]
        ])

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"historical_sentiment_{timestamp_str}.csv"
    output.seek(0)
    return Response(
        content=output.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


@app.get("/api/metrics")
@app.get("/metrics")
def get_evaluation_metrics():
    return train_and_evaluate_models()


@app.get("/api/history")
@app.get("/history")
def get_analysis_history(limit: int = 15, offset: int = 0):
    """
    Retrieve historical sentiment analysis records ordered by newest first.
    Includes sentiment counts and summary statistics for charts.
    """
    target_path = get_db_path()
    try:
        with sqlite3.connect(target_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Total records count
            cur.execute("SELECT COUNT(*) FROM reviews_history")
            total_records = cur.fetchone()[0]

            # Fetch recent records ordered by newest first
            cur.execute("""
                SELECT id, review, sentiment, confidence, categories, priority,
                       vader_compound, subjectivity, created_at,
                       COALESCE(vader_pos, 0.0) as vader_pos,
                       COALESCE(vader_neg, 0.0) as vader_neg,
                       COALESCE(vader_neu, 0.0) as vader_neu,
                       COALESCE(source, 'single') as source
                FROM reviews_history 
                ORDER BY id DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            rows = [dict(row) for row in cur.fetchall()]

            # Sentiment distribution counts across all history
            cur.execute("""
                SELECT sentiment, COUNT(*) as cnt 
                FROM reviews_history 
                GROUP BY sentiment
            """)
            sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
            for row in cur.fetchall():
                sent = row[0]
                if sent in sentiment_counts:
                    sentiment_counts[sent] = row[1]

            # Average compound score
            cur.execute("SELECT AVG(vader_compound) FROM reviews_history")
            avg_comp_val = cur.fetchone()[0]
            avg_compound = round(avg_comp_val, 4) if avg_comp_val is not None else 0.0

            return {
                "status": "success",
                "total_count": total_records,
                "count": len(rows),
                "history": rows,
                "sentiment_summary": sentiment_counts,
                "average_compound": avg_compound
            }
    except Exception as e:
        logger.error(f"Error querying history from '{target_path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error querying history: {str(e)}")



@app.get("/assets/{filename}")
def serve_asset(filename: str):
    """
    Serve static assets (e.g. Sentiq logos, branding icons) securely from the assets directory.
    """
    assets_dir = os.path.abspath(os.path.join(BASE_DIR, "assets"))
    target_file = os.path.abspath(os.path.join(assets_dir, filename))
    if not target_file.startswith(assets_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    if os.path.isfile(target_file):
        media_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        return FileResponse(target_file, media_type=media_type)
    raise HTTPException(status_code=404, detail="Asset not found")


def get_sentiq_logo_uri() -> str:
    """
    Returns base64 data URI of Sentiq logo for instantaneous zero-latency rendering,
    with fallback to static asset URL.
    """
    b64_file = os.path.join(BASE_DIR, "assets", "sentiq-logo-jpg.base64.txt")
    if os.path.isfile(b64_file):
        try:
            with open(b64_file, "r", encoding="utf-8") as f:
                b64_data = f.read().strip()
                if b64_data:
                    return f"data:image/jpeg;base64,{b64_data}"
        except Exception as e:
            logger.warning(f"Error reading logo base64 file: {e}")
    return "/assets/sentiq-logo.png"

@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def get_interactive_dashboard():
    logo_src = get_sentiq_logo_uri()
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiq – Review Intelligence</title>
    <!-- Chart.js for Visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #142B52;            /* Deep Navy */
            --primary-hover: #0F172A;
            --primary-light: #EEF2F6;
            --secondary: #6D28D9;          /* Purple Accent */
            --secondary-hover: #5B21B6;
            --secondary-light: #F3E8FF;
            --secondary-subtle: #FAF5FF;
            --bg: #F8FAFC;                 /* Clean Slate 50 */
            --sidebar-bg: #FFFFFF;
            --card-bg: #FFFFFF;
            --card-border: #E2E8F0;         /* Slate 200 */
            --card-border-hover: #CBD5E1;
            --text-main: #0F172A;          /* Slate 900 */
            --text-body: #1E293B;          /* Slate 800 */
            --text-muted: #64748B;         /* Slate 500 */
            --text-subtle: #94A3B8;        /* Slate 400 */
            --pos: #059669;                /* Green 600 */
            --pos-bg: #ECFDF5;
            --pos-border: #A7F3D0;
            --pos-chart: #16a34a;          /* Green Chart Point */
            --neg: #DC2626;                /* Red 600 */
            --neg-bg: #FEF2F2;
            --neg-border: #FECACA;
            --neg-chart: #dc2626;          /* Red Chart Point */
            --neu: #D97706;                /* Amber 600 */
            --neu-bg: #FFFBEB;
            --neu-border: #FDE68A;
            --neu-chart: #d97706;          /* Amber Chart Point */
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --shadow-subtle: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
            --shadow-card: 0 2px 6px -1px rgba(0, 0, 0, 0.06), 0 1px 4px -2px rgba(0, 0, 0, 0.04);
            --shadow-float: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
            --sidebar-width: 256px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg);
            color: var(--text-body);
            line-height: 1.5;
            font-size: 14px;
            overflow-x: hidden;
        }

        /* ========================================== */
        /* SENTIQ STARTUP INTRO SCREEN               */
        /* ========================================== */
        body.sentiq-intro-active {
            overflow: hidden !important;
            height: 100vh !important;
        }

        .sentiq-intro-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: #FFFFFF;
            z-index: 9999999;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 1;
            animation: sentiqOverlayFadeOut 600ms cubic-bezier(0.4, 0, 0.2, 1) 1800ms forwards;
            pointer-events: all;
            user-select: none;
            -webkit-user-select: none;
        }

        .sentiq-intro-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 24px;
            max-width: 90vw;
        }

        .sentiq-logo-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 22px;
            opacity: 0;
            transform: scale(0.90);
            animation: sentiqLogoReveal 600ms cubic-bezier(0.16, 1, 0.3, 1) 300ms forwards;
        }

        .sentiq-intro-logo {
            width: 100%;
            max-width: 440px;
            height: auto;
            object-fit: contain;
            display: block;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }

        .sentiq-intro-tagline {
            color: #142B52;
            font-size: 1.15rem;
            font-weight: 500;
            letter-spacing: 0.5px;
            margin: 0;
            padding: 0;
            opacity: 0;
            transform: translateY(8px);
            animation: sentiqTaglineFadeIn 600ms cubic-bezier(0.16, 1, 0.3, 1) 900ms forwards;
        }

        @keyframes sentiqLogoReveal {
            0% { opacity: 0; transform: scale(0.90); }
            100% { opacity: 1; transform: scale(1.0); }
        }

        @keyframes sentiqTaglineFadeIn {
            0% { opacity: 0; transform: translateY(8px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        @keyframes sentiqOverlayFadeOut {
            0% { opacity: 1; visibility: visible; }
            99% { opacity: 0; visibility: visible; }
            100% { opacity: 0; visibility: hidden; pointer-events: none; }
        }

        @media (max-width: 768px) {
            .sentiq-intro-logo { max-width: 320px; }
            .sentiq-intro-tagline { font-size: 1rem; letter-spacing: 0.3px; }
        }

        @media (max-width: 480px) {
            .sentiq-intro-logo { max-width: 250px; }
            .sentiq-intro-tagline { font-size: 0.88rem; letter-spacing: 0.2px; }
        }

        @media (prefers-reduced-motion: reduce) {
            .sentiq-intro-overlay { animation-duration: 200ms !important; animation-delay: 400ms !important; }
            .sentiq-logo-wrapper { animation: none !important; opacity: 1 !important; transform: none !important; }
            .sentiq-intro-tagline { animation: none !important; opacity: 1 !important; transform: none !important; }
        }

        /* ========================================== */
        /* APPLICATION LAYOUT (SIDEBAR + MAIN)       */
        /* ========================================== */
        .app-shell {
            display: flex;
            min-height: 100vh;
            background-color: var(--bg);
        }

        /* FIXED VERTICAL SIDEBAR */
        .app-sidebar {
            width: var(--sidebar-width);
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--card-border);
            display: flex;
            flex-direction: column;
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 50;
            transition: transform 0.2s ease-in-out;
        }

        .sidebar-brand {
            padding: 24px 20px 20px 20px;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .sidebar-logo-img {
            height: 34px;
            width: auto;
            max-width: 140px;
            object-fit: contain;
        }

        .sidebar-brand-text h1 {
            font-size: 15px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.2px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .sidebar-brand-text p {
            font-size: 11px;
            font-weight: 500;
            color: var(--secondary);
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .sidebar-nav {
            padding: 18px 12px;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .sidebar-nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            border-radius: var(--radius-md);
            color: var(--text-muted);
            font-size: 13.5px;
            font-weight: 500;
            text-decoration: none;
            cursor: pointer;
            border: none;
            background: transparent;
            width: 100%;
            text-align: left;
            transition: all 0.15s ease;
        }

        .sidebar-nav-item:hover {
            color: var(--text-main);
            background-color: #F1F5F9;
        }

        .sidebar-nav-item.active {
            color: var(--secondary);
            background-color: var(--secondary-light);
            font-weight: 600;
        }

        .sidebar-nav-item svg {
            width: 18px;
            height: 18px;
            stroke-width: 2;
            flex-shrink: 0;
        }

        .sidebar-footer {
            padding: 16px;
            border-top: 1px solid var(--card-border);
            background-color: #FAFAFA;
        }

        .sidebar-footer-card {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .sidebar-status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11.5px;
            color: var(--text-muted);
        }

        .status-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            background-color: var(--pos);
            border-radius: 50%;
            margin-right: 5px;
        }

        /* MAIN CONTENT AREA */
        .app-main {
            flex: 1;
            margin-left: var(--sidebar-width);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            background-color: var(--bg);
        }

        .app-topbar {
            background-color: #FFFFFF;
            border-bottom: 1px solid var(--card-border);
            padding: 16px 36px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 40;
        }

        .topbar-title-wrap h2 {
            font-size: 18px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.3px;
        }

        .topbar-title-wrap p {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .topbar-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .topbar-menu-btn {
            display: none;
            background: transparent;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 6px 10px;
            cursor: pointer;
            color: var(--text-main);
        }

        .app-content {
            padding: 28px 36px 48px 36px;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
        }

        /* VIEWS TOGGLE */
        .app-view {
            display: none;
        }

        .app-view.active {
            display: block;
            animation: fadeIn 0.15s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* BUTTONS & UI PRIMITIVES */
        .btn-primary-purple {
            background-color: var(--secondary);
            color: #FFFFFF;
            border: none;
            border-radius: var(--radius-sm);
            padding: 11px 20px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: background-color 0.15s ease, transform 0.05s ease;
            text-decoration: none;
        }

        .btn-primary-purple:hover {
            background-color: var(--secondary-hover);
        }

        .btn-primary-purple:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background-color: #FFFFFF;
            color: var(--text-body);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-decoration: none;
            transition: all 0.15s ease;
        }

        .btn-secondary:hover {
            border-color: var(--card-border-hover);
            background-color: #F8FAFC;
        }

        .btn-ghost {
            background: transparent;
            color: var(--text-muted);
            border: none;
            font-size: 12.5px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: var(--radius-sm);
            transition: color 0.15s ease;
        }

        .btn-ghost:hover {
            color: var(--text-main);
            background-color: #F1F5F9;
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-lg);
            padding: 24px;
            box-shadow: var(--shadow-subtle);
            transition: border-color 0.15s ease;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
        }

        .card-header h3 {
            font-size: 15px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.2px;
        }

        /* SEGMENTED CONTROL */
        .segmented-control {
            display: inline-flex;
            background-color: #F1F5F9;
            padding: 4px;
            border-radius: var(--radius-md);
            margin-bottom: 22px;
            border: 1px solid var(--card-border);
        }

        .segmented-btn {
            background: transparent;
            border: none;
            padding: 7px 16px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .segmented-btn.active {
            background-color: #FFFFFF;
            color: var(--secondary);
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        /* ========================================== */
        /* VIEW 1: NEW REVIEW WORKSPACE              */
        /* ========================================== */
        .workspace-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 28px;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .workspace-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            margin-bottom: 18px;
        }

        .form-group label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 6px;
        }

        .review-textarea-wrapper {
            position: relative;
        }

        .review-textarea {
            width: 100%;
            height: 180px;
            padding: 14px;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            font-size: 14px;
            line-height: 1.6;
            color: var(--text-main);
            resize: vertical;
            outline: none;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
            background-color: #FFFFFF;
        }

        .review-textarea:focus {
            border-color: var(--secondary);
            box-shadow: 0 0 0 3px rgba(109, 40, 217, 0.1);
        }

        .textarea-meta-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            font-size: 12px;
            color: var(--text-muted);
        }

        .input-tools-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 18px;
            flex-wrap: wrap;
        }

        .btn-mic-clean {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-body);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s ease;
        }

        .btn-mic-clean:hover {
            border-color: var(--card-border-hover);
            background-color: #F8FAFC;
        }

        .btn-mic-clean.recording {
            background-color: #FEF2F2;
            border-color: #F87171;
            color: #DC2626;
            animation: pulseRecord 1.5s infinite;
        }

        @keyframes pulseRecord {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.3); }
            70% { box-shadow: 0 0 0 6px rgba(220, 38, 38, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }

        .sample-select {
            padding: 8px 12px;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            font-size: 13px;
            background-color: #FFFFFF;
            color: var(--text-body);
            outline: none;
            max-width: 320px;
            width: 100%;
        }

        .settings-panel {
            background-color: #F8FAFC;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            padding: 14px;
            margin-bottom: 20px;
        }

        .slider-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
        }

        .slider-row input[type="range"] {
            flex: 1;
            accent-color: var(--secondary);
            cursor: pointer;
        }

        .threshold-badge {
            font-size: 12px;
            font-weight: 600;
            color: var(--secondary);
            background-color: var(--secondary-light);
            padding: 2px 8px;
            border-radius: var(--radius-sm);
        }

        /* ANALYSIS RESULTS COLUMN */
        .results-empty-state {
            text-align: center;
            padding: 64px 24px;
            background-color: #FFFFFF;
            border: 1px dashed var(--card-border);
            border-radius: var(--radius-lg);
            color: var(--text-muted);
        }

        .results-empty-icon {
            width: 48px;
            height: 48px;
            margin: 0 auto 16px auto;
            color: var(--text-subtle);
        }

        .results-empty-state h4 {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 6px;
        }

        .results-container {
            display: none;
            flex-direction: column;
            gap: 20px;
        }

        /* OVERALL HERO BANNER */
        .sentiment-hero-banner {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-lg);
            padding: 20px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--shadow-subtle);
        }

        .sentiment-hero-left h4 {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .sentiment-badge-lg {
            font-size: 18px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            border-radius: var(--radius-md);
            margin-top: 6px;
        }

        .badge-positive {
            background-color: var(--pos-bg);
            color: #065F46;
            border: 1px solid var(--pos-border);
        }

        .badge-negative {
            background-color: var(--neg-bg);
            color: #991B1B;
            border: 1px solid var(--neg-border);
        }

        .badge-neutral {
            background-color: var(--neu-bg);
            color: #92400E;
            border: 1px solid var(--neu-border);
        }

        .sentiment-hero-compound {
            text-align: right;
        }

        .sentiment-hero-compound .val {
            font-size: 22px;
            font-weight: 700;
            color: var(--primary);
            font-family: monospace;
        }

        .sentiment-hero-compound .lbl {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        /* METRICS 5-TILE GRID */
        .metrics-five-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 12px;
        }

        .metric-mini-tile {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            padding: 12px;
            text-align: center;
        }

        .metric-mini-tile .tile-lbl {
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 500;
            margin-bottom: 4px;
        }

        .metric-mini-tile .tile-val {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-main);
            font-family: monospace;
        }

        /* SENTENCE LEVEL SECTION */
        .sentence-section {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-lg);
            padding: 20px;
        }

        .sentence-filter-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }

        .sentence-filter-pill {
            background: transparent;
            border: 1px solid var(--card-border);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .sentence-filter-pill.active {
            background-color: var(--secondary-light);
            border-color: var(--secondary);
            color: var(--secondary);
            font-weight: 600;
        }

        .sentence-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-height: 280px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .sentence-item {
            padding: 12px 14px;
            background-color: #F8FAFC;
            border: 1px solid var(--card-border);
            border-left-width: 4px;
            border-radius: var(--radius-sm);
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            font-size: 13px;
            line-height: 1.5;
        }

        .sentence-item.border-pos { border-left-color: var(--pos); }
        .sentence-item.border-neg { border-left-color: var(--neg); }
        .sentence-item.border-neu { border-left-color: var(--neu); }

        .sentence-badge-sm {
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 12px;
            white-space: nowrap;
        }

        /* MODEL PREDICTIONS COMPARISON */
        .models-compare-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .model-compare-tile {
            background-color: #F8FAFC;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 10px;
            text-align: center;
        }

        .model-compare-tile .model-name {
            font-size: 11.5px;
            font-weight: 600;
            color: var(--text-muted);
        }

        .model-compare-tile .model-pred {
            font-size: 13.5px;
            font-weight: 700;
            margin-top: 4px;
        }

        /* ASPECT TAGS */
        .aspects-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }

        .aspect-tag {
            font-size: 12px;
            padding: 4px 10px;
            background-color: #F1F5F9;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            color: var(--text-body);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .aspect-tag.active {
            background-color: var(--secondary-light);
            border-color: #DDD6FE;
            color: var(--secondary);
            font-weight: 600;
        }

        /* ========================================== */
        /* VIEW 2 & 4: DATA TABLES (MY REVIEWS / ALL) */
        /* ========================================== */
        .table-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }

        .table-search-input {
            padding: 8px 14px;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            font-size: 13px;
            width: 260px;
            background-color: #FFFFFF;
            outline: none;
        }

        .table-search-input:focus {
            border-color: var(--secondary);
        }

        .filter-group {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .select-filter {
            padding: 7px 10px;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            font-size: 12.5px;
            background-color: #FFFFFF;
            color: var(--text-body);
            outline: none;
        }

        .data-table-container {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-subtle);
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }

        .data-table th {
            background-color: #F8FAFC;
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 1px solid var(--card-border);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        .data-table td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--card-border);
            color: var(--text-body);
            vertical-align: middle;
        }

        .data-table tr:last-child td {
            border-bottom: none;
        }

        .data-table tr:hover td {
            background-color: #F8FAFC;
        }

        .table-pagination {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            border-top: 1px solid var(--card-border);
            font-size: 12.5px;
            color: var(--text-muted);
            background-color: #FFFFFF;
        }

        .btn-page {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            padding: 5px 10px;
            border-radius: var(--radius-sm);
            font-size: 12px;
            cursor: pointer;
            color: var(--text-body);
        }

        .btn-page:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        /* ========================================== */
        /* VIEW 3: DASHBOARD METRICS & CHARTS        */
        /* ========================================== */
        .dashboard-kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: var(--radius-lg);
            padding: 20px;
            box-shadow: var(--shadow-subtle);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 12.5px;
            font-weight: 500;
        }

        .kpi-body {
            margin-top: 12px;
            display: flex;
            align-items: baseline;
            justify-content: space-between;
        }

        .kpi-count {
            font-size: 28px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.5px;
        }

        .kpi-pct-badge {
            font-size: 12px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 12px;
        }

        .dashboard-charts-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        @media (max-width: 1024px) {
            .dashboard-charts-grid {
                grid-template-columns: 1fr;
            }
        }

        .chart-box {
            position: relative;
            height: 260px;
            width: 100%;
        }

        /* PRIORITY & ASPECT ANALYTICS BARS */
        .progress-bar-row {
            margin-bottom: 12px;
        }

        .progress-bar-info {
            display: flex;
            justify-content: space-between;
            font-size: 12.5px;
            margin-bottom: 4px;
        }

        .progress-track {
            height: 7px;
            background-color: #F1F5F9;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        /* MODAL STYLING */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(2px);
            z-index: 10000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .modal-card {
            background-color: #FFFFFF;
            border-radius: var(--radius-lg);
            width: 100%;
            max-width: 680px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: var(--shadow-float);
            padding: 24px;
        }

        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 14px;
            margin-bottom: 16px;
        }

        .modal-close-btn {
            background: transparent;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: var(--text-muted);
            line-height: 1;
        }

        /* TOAST CONTAINER */
        .toast-container {
            position: fixed;
            bottom: 24px;
            right: 24px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            z-index: 99999;
        }

        .toast-msg {
            background-color: var(--primary);
            color: #FFFFFF;
            padding: 10px 16px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            box-shadow: var(--shadow-card);
            animation: slideInRight 0.2s ease;
        }

        @keyframes slideInRight {
            from { transform: translateX(30px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* SPINNER */
        .btn-spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #FFFFFF;
            animation: spin 0.6s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* RESPONSIVE TOGGLES */
        @media (max-width: 900px) {
            .app-sidebar {
                transform: translateX(-100%);
            }
            .app-sidebar.open {
                transform: translateX(0);
                box-shadow: var(--shadow-float);
            }
            .app-main {
                margin-left: 0;
            }
            .topbar-menu-btn {
                display: block;
            }
        }
    </style>
</head>
<body>
    <!-- ========================================== -->
    <!-- SENTIQ STARTUP INTRO SCREEN               -->
    <!-- ========================================== -->
    <div id="sentiqIntroOverlay" class="sentiq-intro-overlay" aria-hidden="true">
        <div class="sentiq-intro-content">
            <div class="sentiq-logo-wrapper">
                <img src="__SENTIQ_LOGO_SRC__" alt="Sentiq – Review Intelligence" class="sentiq-intro-logo" id="sentiqIntroImg">
            </div>
            <p class="sentiq-intro-tagline">Every Review Has a Story. We Decode It.</p>
        </div>
    </div>

    <!-- MAIN SAAS SHELL -->
    <div class="app-shell">
        <!-- FIXED VERTICAL SIDEBAR -->
        <aside class="app-sidebar" id="appSidebar">
            <div class="sidebar-brand">
                <img src="__SENTIQ_LOGO_SRC__" alt="Sentiq Logo" class="sidebar-logo-img">
                <div class="sidebar-brand-text">
                    <h1>Sentiq</h1>
                    <p>Review Intelligence</p>
                </div>
            </div>

            <!-- PRIMARY 4 NAVIGATION ITEMS -->
            <nav class="sidebar-nav">
                <button class="sidebar-nav-item active" id="nav-new-review" onclick="navigateTo('new-review')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 5v14M5 12h14"/></svg>
                    <span>New Review</span>
                </button>

                <button class="sidebar-nav-item" id="nav-my-reviews" onclick="navigateTo('my-reviews')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    <span>My Reviews</span>
                </button>

                <button class="sidebar-nav-item" id="nav-dashboard" onclick="navigateTo('dashboard')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/></svg>
                    <span>Dashboard</span>
                </button>

                <button class="sidebar-nav-item" id="nav-reviews" onclick="navigateTo('reviews')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
                    <span>Reviews</span>
                </button>
            </nav>

            <div class="sidebar-footer">
                <div class="sidebar-footer-card">
                    <div class="sidebar-status-row">
                        <span><span class="status-dot"></span>Sentiq v3.0</span>
                        <span style="font-weight:600; color:var(--pos);">Active</span>
                    </div>
                    <button onclick="openDocModal()" class="btn-secondary" style="font-size:12px; padding:6px 10px; justify-content:center;">
                        📖 Technical Reference
                    </button>
                </div>
            </div>
        </aside>

        <!-- MAIN WORKSPACE -->
        <main class="app-main">
            <!-- TOPBAR -->
            <header class="app-topbar">
                <div style="display:flex; align-items:center; gap:12px;">
                    <button class="topbar-menu-btn" onclick="toggleSidebar()">☰</button>
                    <div class="topbar-title-wrap">
                        <h2 id="pageMainHeading">New Review</h2>
                        <p id="pageMainSubtitle">Enter customer feedback to understand its sentiment.</p>
                    </div>
                </div>

                <div class="topbar-actions">
                    <a href="/api/export-history-csv" class="btn-secondary" id="exportHistoryBtn" title="Download SQLite Historical Sentiment Analyses">
                        📥 Export History CSV
                    </a>
                </div>
            </header>

            <div class="app-content">
                <!-- ============================================================== -->
                <!-- 1. VIEW: NEW REVIEW (WORKSPACE)                               -->
                <!-- ============================================================== -->
                <div class="app-view active" id="view-new-review">
                    <!-- Segmented sub-tabs for Single Review vs Batch Analysis -->
                    <div class="segmented-control">
                        <button class="segmented-btn active" id="singleTabBtn" onclick="switchTab('single')">
                            Single Review
                        </button>
                        <button class="segmented-btn" id="batchTabBtn" onclick="switchTab('batch')">
                            Batch & Multi-Format File Upload
                        </button>
                    </div>

                    <!-- SUBMODE A: SINGLE REVIEW (TEXT-ONLY) -->
                    <div id="singleModeSection">
                        <div class="workspace-grid">
                            <!-- LEFT COLUMN: INPUT -->
                            <div class="card">
                                <div class="card-header">
                                    <h3>Customer Feedback Input</h3>
                                    <button class="btn-mic-clean" id="micBtn" onclick="toggleSpeechRecognition()" title="Voice input via microphone">
                                        <span>🎙️</span>
                                        <span id="micText">Speak Review</span>
                                    </button>
                                </div>

                                <div class="form-group">
                                    <label for="sampleDropdown">📌 Preset Test Sample (Optional):</label>
                                    <select class="sample-select" id="sampleDropdown" onchange="handleSampleSelect(this.value)">
                                        <option value="">-- Choose a pre-written test review --</option>
                                        <option value="samsung_a55">📱 Samsung Galaxy A55 (5-Sentence Long Paragraph)</option>
                                        <option value="tech_pos">🔋 Flagship Phone (Positive Battery & Display)</option>
                                        <option value="tech_neg">💻 Laptop Overheating (Negative Defect)</option>
                                        <option value="food_pos">🍕 Food Delivery (Fast & Fresh Positive)</option>
                                        <option value="ecom_neg">📦 Shipping Delay & Broken Item (Complaint)</option>
                                        <option value="spec_neu">📐 Product Dimension & Cable Specs (Neutral)</option>
                                        <option value="support_neg">📞 Customer Support (Negative Rude Service)</option>
                                        <option value="negation_pos">✨ Negation ("Not bad at all, actually quite good!")</option>
                                        <option value="negation_neg">❌ Negation ("The camera is not good.")</option>
                                    </select>
                                </div>

                                <div class="form-group">
                                    <label for="reviewInput">Review Text <span style="font-weight:400; color:var(--text-muted);">(Paste or type a product review here...)</span></label>
                                    <div class="review-textarea-wrapper">
                                        <textarea class="review-textarea" id="reviewInput" placeholder="Paste or type a product review here... Write or paste a customer review here to analyze sentiment, aspects, and sentences." oninput="updateCharCount()"></textarea>
                                        <div class="textarea-meta-bar">
                                            <span id="charCount">0 characters</span>
                                            <button type="button" class="btn-ghost" onclick="clearReviewInput()">Clear</button>
                                        </div>
                                    </div>
                                </div>

                                <div class="settings-panel">
                                    <div class="slider-row">
                                        <span style="font-weight:600; font-size:12.5px;">VADER Compound Threshold:</span>
                                        <input type="range" id="thresholdSlider" min="0.0" max="0.5" step="0.01" value="0.05" oninput="updateThreshold(this.value)">
                                        <span class="threshold-badge" id="thresholdVal">±0.05</span>
                                    </div>
                                    <div style="margin-top:10px; display:flex; align-items:center; gap:8px;">
                                        <input type="checkbox" id="stopwordsCheckbox" onchange="toggleStopwordsDisplay(this.checked)">
                                        <label for="stopwordsCheckbox" style="font-size:12px; color:var(--text-muted); cursor:pointer;">Highlight informative keywords</label>
                                    </div>
                                </div>

                                <div id="singleErrorAlert" style="display:none; padding:10px 14px; background-color:var(--neg-bg); border:1px solid var(--neg-border); color:var(--neg); border-radius:var(--radius-sm); font-size:13px; margin-bottom:14px;"></div>

                                <button id="analyzeBtn" class="btn-primary-purple" style="width:100%;" onclick="triggerAnalyze()">
                                    <span class="btn-spinner" id="analyzeSpinner" style="display:none;"></span>
                                    <span id="analyzeBtnText">⚡ Analyze Sentiment</span>
                                </button>
                            </div>

                            <!-- RIGHT COLUMN: ANALYSIS RESULTS -->
                            <div>
                                <!-- EMPTY PLACEHOLDER -->
                                <div class="results-empty-state" id="emptyPlaceholder">
                                    <svg class="results-empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                                    <h4>Your analysis will appear here.</h4>
                                    <p style="font-size:13px; max-width:340px; margin:0 auto;">Enter or speak customer feedback on the left and click Analyze to view overall sentiment, metrics, and sentence breakdown.</p>
                                </div>

                                <!-- RESULTS CONTAINER -->
                                <div class="results-container" id="singleResultsCard">
                                    <!-- OVERALL HERO BANNER -->
                                    <div class="sentiment-hero-banner">
                                        <div class="sentiment-hero-left">
                                            <h4>Predicted Sentiment</h4>
                                            <span class="sentiment-badge-lg badge-positive" id="singleSentimentBadge">Positive</span>
                                        </div>
                                        <div class="sentiment-hero-compound">
                                            <div class="val" id="singleCompoundScore">+0.0000</div>
                                            <div class="lbl">Compound Score</div>
                                        </div>
                                    </div>

                                    <!-- 5-TILE METRICS GRID -->
                                    <div class="metrics-five-grid">
                                        <div class="metric-mini-tile">
                                            <div class="tile-lbl">ML Confidence</div>
                                            <div class="tile-val" id="singleConfidence">--</div>
                                        </div>
                                        <div class="metric-mini-tile">
                                            <div class="tile-lbl">VADER Compound</div>
                                            <div class="tile-val" id="singleVaderCompound">--</div>
                                        </div>
                                        <div class="metric-mini-tile">
                                            <div class="tile-lbl">TB Polarity</div>
                                            <div class="tile-val" id="singlePolarity">--</div>
                                        </div>
                                        <div class="metric-mini-tile">
                                            <div class="tile-lbl">TB Subjectivity</div>
                                            <div class="tile-val" id="singleSubjectivity">--</div>
                                        </div>
                                        <div class="metric-mini-tile">
                                            <div class="tile-lbl">Urgency Priority</div>
                                            <div class="tile-val" id="singlePriorityBadge" style="font-size:13px;">Low</div>
                                        </div>
                                    </div>

                                    <!-- SENTENCE LEVEL ANALYSIS -->
                                    <div class="sentence-section">
                                        <div class="card-header" style="margin-bottom:10px;">
                                            <h3>Sentence-Level Breakdown</h3>
                                            <span id="sentenceCountBadge" style="font-size:12px; color:var(--text-muted);">0 sentences</span>
                                        </div>

                                        <div class="sentence-filter-bar">
                                            <button class="sentence-filter-pill active" onclick="setSentenceTableFilter('All', this)">All</button>
                                            <button class="sentence-filter-pill" onclick="setSentenceTableFilter('Positive', this)">Positive</button>
                                            <button class="sentence-filter-pill" onclick="setSentenceTableFilter('Negative', this)">Negative</button>
                                            <button class="sentence-filter-pill" onclick="setSentenceTableFilter('Neutral', this)">Neutral</button>
                                        </div>

                                        <div class="sentence-list" id="singleSentenceTableBody">
                                            <!-- Injected dynamically -->
                                        </div>
                                    </div>

                                    <!-- MODEL PREDICTIONS COMPARISON -->
                                    <div class="card" style="padding:18px;">
                                        <div class="card-header" style="margin-bottom:6px;">
                                            <h3>Model Predictions vs Consensus</h3>
                                            <span style="font-size:11.5px; color:var(--text-muted);">Multi-Model Agreement</span>
                                        </div>
                                        <div class="models-compare-grid">
                                            <div class="model-compare-tile">
                                                <div class="model-name">VADER Lexicon</div>
                                                <div class="model-pred" id="modelPredVader">--</div>
                                            </div>
                                            <div class="model-compare-tile">
                                                <div class="model-name">TextBlob Rule</div>
                                                <div class="model-pred" id="modelPredTextBlob">--</div>
                                            </div>
                                            <div class="model-compare-tile">
                                                <div class="model-name">Logistic Reg.</div>
                                                <div class="model-pred" id="modelPredLR">--</div>
                                            </div>
                                            <div class="model-compare-tile">
                                                <div class="model-name">Naive Bayes</div>
                                                <div class="model-pred" id="modelPredNB">--</div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- DETECTED ASPECTS -->
                                    <div class="card" style="padding:18px;">
                                        <h3 style="font-size:14px; margin-bottom:6px;">Aspect Extraction</h3>
                                        <div class="aspects-row" id="aspectBadgesWrap">
                                            <span class="aspect-tag" id="aspectQuality">📦 Product Quality</span>
                                            <span class="aspect-tag" id="aspectShipping">🚚 Delivery & Shipping</span>
                                            <span class="aspect-tag" id="aspectSupport">💬 Customer Support</span>
                                            <span class="aspect-tag" id="aspectPricing">💰 Pricing & Value</span>
                                        </div>
                                    </div>

                                    <!-- ACTION RECOMMENDATION & EXPLANATION -->
                                    <div class="card" id="singleExplanationCard" style="padding:18px; border-left:4px solid var(--secondary);">
                                        <h3 style="font-size:14px; margin-bottom:6px;">Action Recommendation</h3>
                                        <p id="singleRecommendationText" style="font-size:13px; color:var(--text-body); margin-bottom:8px;">--</p>
                                        <p id="singleExplanationText" style="font-size:12px; color:var(--text-muted); line-height:1.6;">--</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- SUBMODE B: BATCH & MULTI-FORMAT FILE UPLOAD -->
                    <div id="batchModeSection" style="display:none;">
                        <div class="card" style="margin-bottom:24px;">
                            <div class="card-header">
                                <h3>Batch Review Analysis & Multi-Format File Upload</h3>
                                <span style="font-size:12px; color:var(--text-muted);">Supported: .DOCX, .DOC, .PDF, .CSV, .XLSX, .XLS, .TXT</span>
                            </div>

                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
                                <!-- OPTION 1: FILE UPLOAD -->
                                <div style="background:#F8FAFC; border:1px dashed var(--card-border); border-radius:var(--radius-md); padding:20px; text-align:center;">
                                    <h4 style="font-size:14px; margin-bottom:8px;">Option 1: Upload a File</h4>
                                    <p style="font-size:12px; color:var(--text-muted); margin-bottom:14px;">Upload any customer review document or spreadsheet.</p>
                                    <input type="file" id="multiFileInput" accept=".docx,.doc,.pdf,.csv,.xlsx,.xls,.txt" style="display:none;" onchange="handleFileSelected(this.files)">
                                    <button class="btn-secondary" onclick="document.getElementById('multiFileInput').click()" style="margin-bottom:10px;">
                                        📁 Choose Document File
                                    </button>
                                    <div id="selectedFileName" style="font-size:12px; color:var(--text-muted); font-weight:500;">No file chosen</div>

                                    <div style="margin-top:14px; text-align:left;">
                                        <label for="docModeSelect" style="font-size:12px; font-weight:600; color:var(--text-main);">Parser Extraction Mode:</label>
                                        <select id="docModeSelect" class="sample-select" style="margin-top:4px;">
                                            <option value="auto">Auto-Detect Structure</option>
                                            <option value="csv_column">CSV Review Column</option>
                                            <option value="newline">Line by Line</option>
                                            <option value="paragraph">Paragraph Blocks</option>
                                        </select>
                                    </div>

                                    <button id="uploadFileBtn" class="btn-primary-purple" style="margin-top:14px; width:100%;" onclick="uploadMultiFormatFile()">
                                        <span class="btn-spinner" id="uploadFileSpinner" style="display:none;"></span>
                                        <span id="uploadFileBtnText">⚡ Upload & Analyze Document</span>
                                    </button>
                                </div>

                                <!-- OPTION 2: PASTE BULK -->
                                <div style="background:#F8FAFC; border:1px solid var(--card-border); border-radius:var(--radius-md); padding:20px;">
                                    <h4 style="font-size:14px; margin-bottom:8px;">Option 2: Paste Multiple Reviews</h4>
                                    <p style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">Paste customer reviews separated by line breaks.</p>
                                    <textarea id="batchPasteInput" style="width:100%; height:120px; padding:10px; border:1px solid var(--card-border); border-radius:var(--radius-sm); font-size:13px; resize:vertical;" placeholder="1. Great service and fast delivery!&#10;2. Product overheats and stopped working.&#10;3. Average quality for the price."></textarea>
                                    <button id="batchPasteBtn" class="btn-primary-purple" style="margin-top:10px; width:100%;" onclick="triggerPastedBatchAnalyze()">
                                        <span class="btn-spinner" id="batchPasteSpinner" style="display:none;"></span>
                                        <span id="batchPasteBtnText">⚡ Process All Reviews in Bulk</span>
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- BATCH RESULTS DISPLAY -->
                        <div id="batchResultsDashboard" style="display:none;">
                            <div class="dashboard-kpi-grid">
                                <div class="kpi-card">
                                    <div class="kpi-header"><span>Total Processed</span></div>
                                    <div class="kpi-body"><span class="kpi-count" id="batchTotalCount">0</span></div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-header"><span>Positive</span><span class="kpi-pct-badge badge-positive" id="batchPosPct">0%</span></div>
                                    <div class="kpi-body"><span class="kpi-count" style="color:var(--pos);" id="batchPosCount">0</span></div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-header"><span>Negative</span><span class="kpi-pct-badge badge-negative" id="batchNegPct">0%</span></div>
                                    <div class="kpi-body"><span class="kpi-count" style="color:var(--neg);" id="batchNegCount">0</span></div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-header"><span>Neutral</span><span class="kpi-pct-badge badge-neutral" id="batchNeuPct">0%</span></div>
                                    <div class="kpi-body"><span class="kpi-count" style="color:var(--neu);" id="batchNeuCount">0</span></div>
                                </div>
                            </div>

                            <div class="dashboard-charts-grid">
                                <div class="card">
                                    <h3 style="font-size:14px; margin-bottom:12px;">Sentiment Distribution</h3>
                                    <div class="chart-box"><canvas id="batchPieChart"></canvas></div>
                                </div>
                                <div class="card">
                                    <h3 style="font-size:14px; margin-bottom:12px;">Category & Model Breakdown</h3>
                                    <div class="chart-box"><canvas id="batchBarChart"></canvas></div>
                                </div>
                            </div>

                            <div class="card">
                                <div class="table-toolbar">
                                    <input type="text" class="table-search-input" id="batchSearchInput" placeholder="Search batch reviews..." oninput="filterBatchTable()">
                                    <div style="display:flex; gap:10px;">
                                        <button class="btn-secondary" id="batchDownloadCsvBtn" onclick="downloadBatchCSV()">Download CSV</button>
                                        <button class="btn-secondary" id="batchExportPdfBtn" onclick="downloadBatchPDF()">Export PDF Report</button>
                                    </div>
                                </div>
                                <div class="data-table-container">
                                    <table class="data-table">
                                        <thead>
                                            <tr>
                                                <th>ID</th>
                                                <th>Review Text</th>
                                                <th>Sentiment</th>
                                                <th>Compound</th>
                                                <th>Aspects</th>
                                                <th>Priority</th>
                                            </tr>
                                        </thead>
                                        <tbody id="batchTableBody"></tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ============================================================== -->
                <!-- 2. VIEW: MY REVIEWS (USER HISTORY)                            -->
                <!-- ============================================================== -->
                <div class="app-view" id="view-my-reviews">
                    <div class="card" style="margin-bottom:20px;">
                        <div class="table-toolbar">
                            <input type="text" class="table-search-input" id="myReviewsSearch" placeholder="Search my reviews..." oninput="filterMyReviews()">
                            <div class="filter-group">
                                <div class="sentence-filter-bar" style="margin:0;">
                                    <button class="sentence-filter-pill active" onclick="setMyReviewsFilter('All', this)">All</button>
                                    <button class="sentence-filter-pill" onclick="setMyReviewsFilter('Positive', this)">Positive</button>
                                    <button class="sentence-filter-pill" onclick="setMyReviewsFilter('Negative', this)">Negative</button>
                                    <button class="sentence-filter-pill" onclick="setMyReviewsFilter('Neutral', this)">Neutral</button>
                                </div>
                                <select class="select-filter" id="myReviewsSort" onchange="filterMyReviews()">
                                    <option value="newest">Newest First</option>
                                    <option value="oldest">Oldest First</option>
                                </select>
                                <button class="btn-secondary" onclick="loadHistory(true)">🔄 Refresh</button>
                            </div>
                        </div>

                        <div class="data-table-container">
                            <table class="data-table">
                                <thead>
                                    <tr>
                                        <th>ID</th>
                                        <th>Review Text</th>
                                        <th>Sentiment</th>
                                        <th>VADER Score</th>
                                        <th>Confidence</th>
                                        <th>Priority</th>
                                        <th>Date & Time</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody id="myReviewsTableBody">
                                    <tr><td colspan="8" style="text-align:center; padding:30px; color:var(--text-muted);">Loading reviews...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- ============================================================== -->
                <!-- 3. VIEW: DASHBOARD (OVERALL ANALYTICS)                        -->
                <!-- ============================================================== -->
                <div class="app-view" id="view-dashboard">
                    <div id="historicalTrendsCard">
                        <!-- TOP 4 KPI CARDS -->
                        <div class="dashboard-kpi-grid">
                            <div class="kpi-card">
                                <div class="kpi-header"><span>Total Analyzed</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>
                                <div class="kpi-body"><span class="kpi-count" id="kpiTotalReviews">0</span></div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-header"><span>Positive Reviews</span><span class="kpi-pct-badge badge-positive" id="kpiPosPct">0%</span></div>
                                <div class="kpi-body"><span class="kpi-count" style="color:var(--pos);" id="kpiPositiveCount">0</span></div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-header"><span>Negative Reviews</span><span class="kpi-pct-badge badge-negative" id="kpiNegPct">0%</span></div>
                                <div class="kpi-body"><span class="kpi-count" style="color:var(--neg);" id="kpiNegativeCount">0</span></div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-header"><span>Neutral Reviews</span><span class="kpi-pct-badge badge-neutral" id="kpiNeuPct">0%</span></div>
                                <div class="kpi-body"><span class="kpi-count" style="color:var(--neu);" id="kpiNeutralCount">0</span></div>
                            </div>
                        </div>

                        <!-- CHARTS SECTION -->
                        <div class="dashboard-charts-grid">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Sentiment Share</h3>
                                    <span style="font-size:11.5px; color:var(--text-muted);">Overall Distribution</span>
                                </div>
                                <div class="chart-box"><canvas id="dashboardDoughnutChart"></canvas></div>
                            </div>

                            <div class="card">
                                <div class="card-header">
                                    <h3>Historical Sentiment Trends</h3>
                                    <div style="display:flex; gap:8px;" id="historySummaryPills">
                                        <button class="btn-ghost" id="refreshHistoryBtn" onclick="loadHistory(true)">🔄</button>
                                    </div>
                                </div>
                                <div id="historyChartEmptyState" style="display:none; text-align:center; padding:40px; color:var(--text-muted);">
                                    No historical reviews yet. Submit a review above to see trends.
                                </div>
                                <div class="chart-box" id="historyChartWrapper">
                                    <canvas id="trendLineChart"></canvas>
                                </div>
                            </div>
                        </div>

                        <!-- ASPECT & PRIORITY ANALYTICS -->
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:24px;">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Feedback by Aspect</h3>
                                    <span style="font-size:11.5px; color:var(--text-muted);">Top Mentions</span>
                                </div>
                                <div id="aspectAnalyticsBars">
                                    <!-- Injected dynamically -->
                                </div>
                            </div>

                            <div class="card">
                                <div class="card-header">
                                    <h3>Priority & Urgency Overview</h3>
                                    <span style="font-size:11.5px; color:var(--text-muted);">Action Required</span>
                                </div>
                                <div id="priorityAnalyticsBars">
                                    <!-- Injected dynamically -->
                                </div>
                            </div>
                        </div>

                        <!-- RECENT REVIEWS -->
                        <div class="card">
                            <div class="card-header">
                                <h3>Recent Analyzed Customer Reviews</h3>
                                <button class="btn-secondary" onclick="navigateTo('reviews')">View All Reviews →</button>
                            </div>
                            <div class="data-table-container">
                                <table class="data-table">
                                    <thead>
                                        <tr>
                                            <th>ID</th>
                                            <th>Review Text</th>
                                            <th>Sentiment</th>
                                            <th>Priority</th>
                                            <th>Date</th>
                                            <th>Action</th>
                                        </tr>
                                    </thead>
                                    <tbody id="dashboardRecentTableBody"></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ============================================================== -->
                <!-- 4. VIEW: ALL REVIEWS (MANAGEMENT TABLE)                       -->
                <!-- ============================================================== -->
                <div class="app-view" id="view-reviews">
                    <div class="card">
                        <div class="table-toolbar">
                            <input type="text" class="table-search-input" id="allReviewsSearch" placeholder="Search across all customer reviews..." oninput="filterAllReviewsTable()">
                            
                            <div class="filter-group">
                                <select class="select-filter" id="filterSentiment" onchange="filterAllReviewsTable()">
                                    <option value="All">All Sentiments</option>
                                    <option value="Positive">Positive</option>
                                    <option value="Negative">Negative</option>
                                    <option value="Neutral">Neutral</option>
                                </select>

                                <select class="select-filter" id="filterAspect" onchange="filterAllReviewsTable()">
                                    <option value="All">All Aspects</option>
                                    <option value="Quality">Product Quality</option>
                                    <option value="Delivery">Delivery & Shipping</option>
                                    <option value="Support">Customer Support</option>
                                    <option value="Pricing">Pricing & Value</option>
                                </select>

                                <select class="select-filter" id="filterPriority" onchange="filterAllReviewsTable()">
                                    <option value="All">All Priorities</option>
                                    <option value="Urgent">Urgent</option>
                                    <option value="High">High</option>
                                    <option value="Medium">Medium</option>
                                    <option value="Low">Low</option>
                                </select>

                                <select class="select-filter" id="filterSort" onchange="filterAllReviewsTable()">
                                    <option value="newest">Newest First</option>
                                    <option value="oldest">Oldest First</option>
                                    <option value="compound_high">Highest Compound</option>
                                    <option value="compound_low">Lowest Compound</option>
                                </select>
                            </div>
                        </div>

                        <div class="data-table-container">
                            <table class="data-table">
                                <thead>
                                    <tr>
                                        <th>ID</th>
                                        <th>Review Preview</th>
                                        <th>Sentiment</th>
                                        <th>VADER Compound</th>
                                        <th>ML Confidence</th>
                                        <th>Aspects</th>
                                        <th>Priority</th>
                                        <th>Date</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody id="historyTableBody">
                                    <!-- Injected dynamically -->
                                </tbody>
                            </table>
                        </div>

                        <div class="table-pagination">
                            <span id="allReviewsPaginationInfo">Showing 0 reviews</span>
                            <div style="display:flex; gap:6px;">
                                <button class="btn-page" id="prevPageBtn" onclick="changeHistoryPage(-1)">Previous</button>
                                <button class="btn-page" id="nextPageBtn" onclick="changeHistoryPage(1)">Next</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- ========================================== -->
    <!-- REVIEW DETAILS MODAL                      -->
    <!-- ========================================== -->
    <div id="reviewModal" class="modal-overlay" onclick="handleModalBackdrop(event)">
        <div class="modal-card">
            <div class="modal-header">
                <div>
                    <h3 id="modalReviewTitle" style="font-size:16px; font-weight:700; color:var(--primary);">Review Details</h3>
                    <span id="modalReviewDate" style="font-size:12px; color:var(--text-muted);">--</span>
                </div>
                <button class="modal-close-btn" onclick="closeReviewModal()">✕</button>
            </div>

            <div style="margin-bottom:18px;">
                <h4 style="font-size:12px; color:var(--text-muted); text-transform:uppercase; margin-bottom:6px;">Full Customer Review</h4>
                <div id="modalReviewText" style="padding:14px; background:#F8FAFC; border:1px solid var(--card-border); border-radius:var(--radius-sm); font-size:13.5px; line-height:1.6;"></div>
            </div>

            <div class="metrics-five-grid" style="margin-bottom:18px;">
                <div class="metric-mini-tile"><div class="tile-lbl">Sentiment</div><div class="tile-val" id="modalSentiment">--</div></div>
                <div class="metric-mini-tile"><div class="tile-lbl">Compound</div><div class="tile-val" id="modalCompound">--</div></div>
                <div class="metric-mini-tile"><div class="tile-lbl">Confidence</div><div class="tile-val" id="modalConfidence">--</div></div>
                <div class="metric-mini-tile"><div class="tile-lbl">Subjectivity</div><div class="tile-val" id="modalSubjectivity">--</div></div>
                <div class="metric-mini-tile"><div class="tile-lbl">Priority</div><div class="tile-val" id="modalPriority">--</div></div>
            </div>

            <div style="margin-bottom:14px;">
                <h4 style="font-size:12px; color:var(--text-muted); text-transform:uppercase; margin-bottom:6px;">Aspects & Category</h4>
                <div id="modalAspects" class="aspects-row"></div>
            </div>

            <div style="text-align:right;">
                <button class="btn-secondary" onclick="closeReviewModal()">Close</button>
            </div>
        </div>
    </div>

    <!-- ========================================== -->
    <!-- TECHNICAL REFERENCE MODAL                 -->
    <!-- ========================================== -->
    <div id="docModal" class="modal-overlay" onclick="handleDocModalBackdrop(event)">
        <div class="modal-card" style="max-width:760px;">
            <div class="modal-header">
                <h3 style="font-size:16px; font-weight:700; color:var(--primary);">Sentiq Technical Reference</h3>
                <button class="modal-close-btn" onclick="closeDocModal()">✕</button>
            </div>
            <div style="font-size:13px; line-height:1.7; color:var(--text-body);">
                <p><strong>Architecture Overview:</strong> Sentiq integrates lexicon-based rule valuation (NLTK VADER) and probabilistic classification (TextBlob PatternAnalyzer) with supervised ML models (TF-IDF + Logistic Regression & Multinomial Naive Bayes).</p>
                <br/>
                <p><strong>Multi-Format Parsing:</strong> Robust document extraction engine supporting .DOCX, .DOC, .PDF (PyMuPDF / pdfplumber fallback), .CSV, .XLSX, .XLS, and .TXT.</p>
                <br/>
                <p><strong>Sentence-Level Analysis:</strong> Sentences are segmented with abbreviation and decimal-number preservation. Compound scores are calculated for each sentence and aggregated into mean valence.</p>
            </div>
            <div style="text-align:right; margin-top:20px;">
                <button class="btn-primary-purple" onclick="closeDocModal()">Done</button>
            </div>
        </div>
    </div>

    <!-- TOAST NOTIFICATION CONTAINER -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- ========================================== -->
    <!-- JAVASCRIPT APPLICATION LOGIC              -->
    <!-- ========================================== -->
    <script>
        // Sentiq Startup Controller
        (function initSentiqStartupIntro() {
            document.body.classList.add('sentiq-intro-active');
            const overlay = document.getElementById('sentiqIntroOverlay');
            if (!overlay) return;

            function dismissIntro() {
                if (overlay.dataset.dismissed === 'true') return;
                overlay.dataset.dismissed = 'true';
                overlay.style.opacity = '0';
                overlay.style.pointerEvents = 'none';
                setTimeout(() => {
                    overlay.style.display = 'none';
                    document.body.classList.remove('sentiq-intro-active');
                }, 300);
            }

            setTimeout(dismissIntro, 2400);
            overlay.addEventListener('click', dismissIntro);
        })();

        // Global State
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
        let currentSingleData = null;
        let activeSingleSentenceFilter = 'All';
        let currentBatchData = null;
        let currentHistoryData = [];
        let trendChartInstance = null;
        let dashboardDoughnutInstance = null;
        let batchPieInstance = null;
        let batchBarInstance = null;
        let isListening = false;
        let recognition = null;
        let historyPage = 1;
        const historyPageSize = 10;
        let activeMyReviewsFilter = 'All';

        // Navigation Controller
        const PAGE_TITLES = {
            'new-review': { title: 'New Review', subtitle: 'Enter customer feedback to understand its sentiment.' },
            'my-reviews': { title: 'My Reviews', subtitle: 'Your complete history of analyzed customer feedback.' },
            'dashboard': { title: 'Sentiment Dashboard', subtitle: 'Understand your customer feedback at a glance.' },
            'reviews': { title: 'All Reviews', subtitle: 'Search, filter, and inspect all customer feedback.' }
        };

        function navigateTo(viewId) {
            document.querySelectorAll('.sidebar-nav-item').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.app-view').forEach(v => v.classList.remove('active'));

            const navBtn = document.getElementById('nav-' + viewId);
            if (navBtn) navBtn.classList.add('active');

            const view = document.getElementById('view-' + viewId);
            if (view) view.classList.add('active');

            if (PAGE_TITLES[viewId]) {
                document.getElementById('pageMainHeading').innerText = PAGE_TITLES[viewId].title;
                document.getElementById('pageMainSubtitle').innerText = PAGE_TITLES[viewId].subtitle;
            }

            // Close mobile sidebar if open
            const sidebar = document.getElementById('appSidebar');
            if (sidebar) sidebar.classList.remove('open');

            if (viewId === 'my-reviews' || viewId === 'dashboard' || viewId === 'reviews') {
                loadHistory();
            }
        }

        function toggleSidebar() {
            const sidebar = document.getElementById('appSidebar');
            if (sidebar) sidebar.classList.toggle('open');
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

        function showToast(message) {
            const container = document.getElementById('toastContainer');
            if (!container) return;
            const toast = document.createElement('div');
            toast.className = 'toast-msg';
            toast.innerText = message;
            container.appendChild(toast);
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transition = 'opacity 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }

        // Single Review Form Handlers
        function updateCharCount() {
            const val = document.getElementById('reviewInput').value;
            document.getElementById('charCount').innerText = `${val.length} character${val.length === 1 ? '' : 's'}`;
        }

        function clearReviewInput() {
            document.getElementById('reviewInput').value = '';
            document.getElementById('sampleDropdown').value = '';
            updateCharCount();
        }

        function handleSampleSelect(val) {
            if (val && PRESETS[val]) {
                document.getElementById('reviewInput').value = PRESETS[val];
                updateCharCount();
            }
        }

        function updateThreshold(val) {
            document.getElementById('thresholdVal').innerText = `±${parseFloat(val).toFixed(2)}`;
        }

        function toggleStopwordsDisplay(checked) {
            // Visual toggle
        }

        // Voice Input (Web Speech API)
        function toggleSpeechRecognition() {
            const micBtn = document.getElementById('micBtn');
            const micText = document.getElementById('micText');

            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                alert('Web Speech API is not supported in this browser. Please use Google Chrome or Microsoft Edge.');
                return;
            }

            if (isListening) {
                stopListening();
                return;
            }

            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                isListening = true;
                micBtn.classList.add('recording');
                micText.innerText = 'Listening...';
                showToast('Microphone listening. Speak your customer review.');
            };

            recognition.onresult = function(event) {
                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript + ' ';
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }

                if (finalTranscript) {
                    const txtArea = document.getElementById('reviewInput');
                    txtArea.value = (txtArea.value ? txtArea.value.trim() + ' ' : '') + finalTranscript.trim();
                    updateCharCount();
                }
            };

            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                stopListening();
            };

            recognition.onend = function() {
                stopListening();
            };

            recognition.start();
        }

        function stopListening() {
            isListening = false;
            const micBtn = document.getElementById('micBtn');
            const micText = document.getElementById('micText');
            if (micBtn) micBtn.classList.remove('recording');
            if (micText) micText.innerText = 'Speak Review';
            if (recognition) {
                try { recognition.stop(); } catch(e) {}
            }
        }

        // Single Review Analysis
        async function triggerAnalyze() {
            const text = document.getElementById('reviewInput').value.trim();
            const threshold = parseFloat(document.getElementById('thresholdSlider').value);
            const errAlert = document.getElementById('singleErrorAlert');
            const spinner = document.getElementById('analyzeSpinner');
            const btnText = document.getElementById('analyzeBtnText');
            const btn = document.getElementById('analyzeBtn');

            if (errAlert) errAlert.style.display = 'none';

            if (!text) {
                if (errAlert) {
                    errAlert.style.display = 'block';
                    errAlert.innerText = 'Please write, paste or dictate a review text before analyzing.';
                } else {
                    alert('Please enter a review to analyze.');
                }
                return;
            }

            spinner.style.display = 'inline-block';
            btnText.innerText = 'Analyzing Sentiment...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ review: text, threshold: threshold })
                });

                const rawText = await res.text();
                let data = null;
                try {
                    data = JSON.parse(rawText);
                } catch (parseErr) {}

                if (!res.ok) {
                    const errMsg = (data && data.detail) ? data.detail : (rawText.trim() || `Analysis failed (${res.status})`);
                    throw new Error(errMsg);
                }

                if (!data) {
                    throw new Error('Analysis returned an empty or invalid response.');
                }

                currentSingleData = data;
                renderSingleResults(data);
                loadHistory();
                showToast('Review analyzed successfully!');
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
            const resCard = document.getElementById('singleResultsCard');
            resCard.style.display = 'flex';

            // Sentiment badge
            const badge = document.getElementById('singleSentimentBadge');
            badge.innerText = data.sentiment;
            badge.className = 'sentiment-badge-lg';
            if (data.sentiment === 'Positive') badge.classList.add('badge-positive');
            else if (data.sentiment === 'Negative') badge.classList.add('badge-negative');
            else badge.classList.add('badge-neutral');

            // Compound Score
            const vComp = (data.overall_compound_score !== undefined) ? data.overall_compound_score : (data.vader_scores?.compound_score || 0.0);
            const compEl = document.getElementById('singleCompoundScore');
            compEl.innerText = (vComp >= 0 ? '+' : '') + vComp.toFixed(4);

            // Metrics
            const confEl = document.getElementById('singleConfidence');
            confEl.innerText = (data.confidence_percentage !== null && data.confidence_percentage !== undefined) 
                ? `${parseFloat(data.confidence_percentage).toFixed(1)}%` 
                : 'N/A';

            document.getElementById('singleVaderCompound').innerText = (vComp >= 0 ? '+' : '') + vComp.toFixed(2);
            document.getElementById('singlePolarity').innerText = (data.textblob_metrics?.polarity || 0.0).toFixed(2);
            document.getElementById('singleSubjectivity').innerText = (data.textblob_metrics?.subjectivity || 0.0).toFixed(2);

            const prioBadge = document.getElementById('singlePriorityBadge');
            prioBadge.innerText = data.priority || 'Low';

            // Sentences
            const sa = data.sentence_analysis;
            currentSingleSentences = (sa && sa.sentences) ? sa.sentences : [];
            document.getElementById('sentenceCountBadge').innerText = `${currentSingleSentences.length} sentence${currentSingleSentences.length === 1 ? '' : 's'}`;
            renderSingleSentencesList(currentSingleSentences);

            // Model predictions
            const ml = data.ml_model_predictions || {};
            document.getElementById('modelPredVader').innerText = data.vader_scores?.sentiment || '--';
            document.getElementById('modelPredTextBlob').innerText = data.textblob_metrics?.polarity > 0.05 ? 'Positive' : (data.textblob_metrics?.polarity < -0.05 ? 'Negative' : 'Neutral');
            document.getElementById('modelPredLR').innerText = ml.logistic_regression?.prediction || 'N/A';
            document.getElementById('modelPredNB').innerText = ml.naive_bayes?.prediction || 'N/A';

            // Aspects
            const cats = data.categories || [];
            updateAspectBadge('aspectQuality', cats.some(c => c.toLowerCase().includes('quality')));
            updateAspectBadge('aspectShipping', cats.some(c => c.toLowerCase().includes('delivery') || c.toLowerCase().includes('shipping')));
            updateAspectBadge('aspectSupport', cats.some(c => c.toLowerCase().includes('support') || c.toLowerCase().includes('service')));
            updateAspectBadge('aspectPricing', cats.some(c => c.toLowerCase().includes('price') || c.toLowerCase().includes('value')));

            // Recommendation
            document.getElementById('singleRecommendationText').innerText = data.recommendation || 'No specific action required.';
            document.getElementById('singleExplanationText').innerText = data.explanation || '';
        }

        function updateAspectBadge(elementId, isActive) {
            const el = document.getElementById(elementId);
            if (!el) return;
            if (isActive) el.classList.add('active');
            else el.classList.remove('active');
        }

        function setSentenceTableFilter(filter, btn) {
            activeSingleSentenceFilter = filter;
            document.querySelectorAll('.sentence-filter-pill').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');

            if (filter === 'All') {
                renderSingleSentencesList(currentSingleSentences);
            } else {
                renderSingleSentencesList(currentSingleSentences.filter(s => s.sentiment === filter));
            }
        }

        function renderSingleSentencesList(sentences) {
            const list = document.getElementById('singleSentenceTableBody');
            if (!sentences.length) {
                list.innerHTML = '<div style="text-align:center; padding:18px; color:var(--text-muted); font-size:12.5px;">No sentences match this filter.</div>';
                return;
            }

            list.innerHTML = sentences.map(s => {
                let borderClass = 'border-neu';
                let badgeClass = 'badge-neutral';
                if (s.sentiment === 'Positive') { borderClass = 'border-pos'; badgeClass = 'badge-positive'; }
                else if (s.sentiment === 'Negative') { borderClass = 'border-neg'; badgeClass = 'badge-negative'; }

                const score = s.vader_scores ? (s.vader_scores.compound >= 0 ? '+' : '') + s.vader_scores.compound.toFixed(2) : '';

                return `
                    <div class="sentence-item ${borderClass}">
                        <div style="flex:1;">
                            <span style="font-size:11px; color:var(--text-muted); font-weight:600; margin-right:6px;">#${s.sentence_id}</span>
                            <span>${escapeHtml(s.sentence_text)}</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span class="sentence-badge-sm ${badgeClass}">${s.sentiment}</span>
                            <span style="font-family:monospace; font-size:11px; color:var(--text-muted);">${score}</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Batch Review Processing
        function handleFileSelected(files) {
            if (files && files.length > 0) {
                document.getElementById('selectedFileName').innerText = files[0].name;
            }
        }

        async function uploadMultiFormatFile() {
            const fileInput = document.getElementById('multiFileInput');
            const mode = document.getElementById('docModeSelect').value;
            const threshold = parseFloat(document.getElementById('thresholdSlider').value);
            const spinner = document.getElementById('uploadFileSpinner');
            const btnText = document.getElementById('uploadFileBtnText');

            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select a file to upload (.docx, .doc, .pdf, .csv, .xlsx, .xls, .txt).');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            const url = `/api/upload-document?threshold=${threshold}&mode=${mode}`;

            spinner.style.display = 'inline-block';
            btnText.innerText = 'Processing File...';

            try {
                const res = await fetch(url, { method: 'POST', body: formData });
                const rawText = await res.text();
                let data = null;
                try { data = JSON.parse(rawText); } catch(e) {}

                if (!res.ok) {
                    throw new Error((data && data.detail) || rawText.trim() || `File processing failed (${res.status})`);
                }

                currentBatchData = data;
                renderBatchDashboard(data);
                loadHistory();
                showToast(`Successfully analyzed ${data.total_processed} reviews from file.`);
            } catch (err) {
                alert('Document Processing Error: ' + err.message);
            } finally {
                spinner.style.display = 'none';
                btnText.innerText = '⚡ Upload & Analyze Document';
            }
        }

        async function triggerPastedBatchAnalyze() {
            const raw = document.getElementById('batchPasteInput').value.trim();
            const threshold = parseFloat(document.getElementById('thresholdSlider').value);
            const spinner = document.getElementById('batchPasteSpinner');
            const btnText = document.getElementById('batchPasteBtnText');

            if (!raw) {
                alert('Please paste reviews to process in bulk.');
                return;
            }

            spinner.style.display = 'inline-block';
            btnText.innerText = 'Processing Reviews...';

            try {
                const res = await fetch('/api/analyze-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: raw, threshold: threshold, mode: 'auto' })
                });

                const rawText = await res.text();
                let data = null;
                try { data = JSON.parse(rawText); } catch(e) {}

                if (!res.ok) {
                    throw new Error((data && data.detail) || rawText.trim() || `Batch analysis failed (${res.status})`);
                }

                currentBatchData = data;
                renderBatchDashboard(data);
                loadHistory();
                showToast(`Successfully analyzed ${data.total_processed} bulk reviews.`);
            } catch (err) {
                alert('Batch Processing Error: ' + err.message);
            } finally {
                spinner.style.display = 'none';
                btnText.innerText = '⚡ Process All Reviews in Bulk';
            }
        }

        function renderBatchDashboard(data) {
            document.getElementById('batchResultsDashboard').style.display = 'block';
            document.getElementById('batchTotalCount').innerText = data.total_processed || 0;

            const counts = data.sentiment_counts || {};
            const pos = counts.Positive || 0;
            const neg = counts.Negative || 0;
            const neu = counts.Neutral || 0;
            const total = data.total_processed || (pos + neg + neu) || 1;

            document.getElementById('batchPosCount').innerText = pos;
            document.getElementById('batchNegCount').innerText = neg;
            document.getElementById('batchNeuCount').innerText = neu;

            document.getElementById('batchPosPct').innerText = `${Math.round((pos / total) * 100)}%`;
            document.getElementById('batchNegPct').innerText = `${Math.round((neg / total) * 100)}%`;
            document.getElementById('batchNeuPct').innerText = `${Math.round((neu / total) * 100)}%`;

            renderBatchCharts(pos, neg, neu);
            populateBatchTable(data.detailed_results || []);
        }

        function renderBatchCharts(pos, neg, neu) {
            if (typeof Chart === 'undefined') return;

            // Batch Pie Chart
            const pieCtx = document.getElementById('batchPieChart');
            if (pieCtx) {
                if (batchPieInstance) batchPieInstance.destroy();
                batchPieInstance = new Chart(pieCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Positive', 'Negative', 'Neutral'],
                        datasets: [{
                            data: [pos, neg, neu],
                            backgroundColor: ['#10B981', '#EF4444', '#F59E0B'],
                            borderWidth: 0
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
                });
            }

            // Batch Bar Chart
            const barCtx = document.getElementById('batchBarChart');
            if (barCtx) {
                if (batchBarInstance) batchBarInstance.destroy();
                batchBarInstance = new Chart(barCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Positive', 'Negative', 'Neutral'],
                        datasets: [{
                            label: 'Reviews',
                            data: [pos, neg, neu],
                            backgroundColor: ['#10B981', '#EF4444', '#F59E0B'],
                            borderRadius: 4
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
                });
            }
        }

        function populateBatchTable(reviews) {
            const tbody = document.getElementById('batchTableBody');
            if (!tbody) return;
            if (!reviews.length) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:18px; color:var(--text-muted);">No reviews to display.</td></tr>';
                return;
            }

            tbody.innerHTML = reviews.map((r, idx) => {
                const s = r.sentiment || 'Neutral';
                let bClass = 'badge-neutral';
                if (s === 'Positive') bClass = 'badge-positive';
                else if (s === 'Negative') bClass = 'badge-negative';

                const cats = Array.isArray(r.categories) ? r.categories.join(', ') : (r.categories || 'General');
                const comp = (r.overall_compound_score !== undefined) ? r.overall_compound_score : (r.vader_scores?.compound_score || 0.0);

                return `
                    <tr>
                        <td style="font-family:monospace; font-weight:600;">#${r.review_id || (idx + 1)}</td>
                        <td style="max-width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(r.original_review)}</td>
                        <td><span class="sentence-badge-sm ${bClass}">${s}</span></td>
                        <td style="font-family:monospace;">${(comp >= 0 ? '+' : '') + comp.toFixed(2)}</td>
                        <td style="font-size:12px; color:var(--text-muted);">${escapeHtml(cats)}</td>
                        <td><span style="font-size:11px; font-weight:600; padding:2px 6px; border-radius:4px; background:#F1F5F9;">${r.priority || 'Low'}</span></td>
                    </tr>
                `;
            }).join('');
        }

        function filterBatchTable() {
            if (!currentBatchData || !currentBatchData.detailed_results) return;
            const q = document.getElementById('batchSearchInput').value.toLowerCase();
            const filtered = currentBatchData.detailed_results.filter(r => (r.original_review || '').toLowerCase().includes(q));
            populateBatchTable(filtered);
        }

        // Export Handlers
        async function downloadBatchCSV() {
            if (!currentBatchData) { alert('No batch data to export.'); return; }
            try {
                const res = await fetch('/api/export-batch-csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentBatchData)
                });
                if (!res.ok) throw new Error('Failed to generate CSV export');
                const blob = await res.blob();
                const a = document.createElement('a');
                a.href = window.URL.createObjectURL(blob);
                a.download = `sentiment_results_${new Date().toISOString().slice(0,10)}.csv`;
                a.click();
                showToast('CSV export downloaded.');
            } catch (e) { alert(e.message); }
        }

        async function downloadBatchPDF() {
            if (!currentBatchData) { alert('No batch data to export.'); return; }
            try {
                const res = await fetch('/api/export-batch-pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentBatchData)
                });
                if (!res.ok) throw new Error('Failed to generate PDF export');
                const blob = await res.blob();
                const a = document.createElement('a');
                a.href = window.URL.createObjectURL(blob);
                a.download = `sentiment_results_${new Date().toISOString().slice(0,10)}.pdf`;
                a.click();
                showToast('PDF report downloaded.');
            } catch (e) { alert(e.message); }
        }

        // History, Dashboard, & Reviews Management
        async function loadHistory(isManualRefresh = false) {
            try {
                const res = await fetch('/api/history?limit=100');
                const rawText = await res.text();
                let data = null;
                try { data = JSON.parse(rawText); } catch(e) {}

                if (!res.ok) throw new Error((data && data.detail) || rawText.trim() || `Server error (${res.status})`);

                const records = (data && data.history) || [];
                currentHistoryData = records;

                // Update Dashboard KPI cards
                const total = data.total_count || records.length;
                const pos = data.sentiment_summary?.Positive || 0;
                const neg = data.sentiment_summary?.Negative || 0;
                const neu = data.sentiment_summary?.Neutral || 0;

                document.getElementById('kpiTotalReviews').innerText = total;
                document.getElementById('kpiPositiveCount').innerText = pos;
                document.getElementById('kpiNegativeCount').innerText = neg;
                document.getElementById('kpiNeutralCount').innerText = neu;

                const denom = total > 0 ? total : 1;
                document.getElementById('kpiPosPct').innerText = `${Math.round((pos / denom) * 100)}%`;
                document.getElementById('kpiNegPct').innerText = `${Math.round((neg / denom) * 100)}%`;
                document.getElementById('kpiNeuPct').innerText = `${Math.round((neu / denom) * 100)}%`;

                // Update Charts & Visuals
                const chartWrapper = document.getElementById('historyChartWrapper');
                const chartEmpty = document.getElementById('historyChartEmptyState');
                const summaryPills = document.getElementById('historySummaryPills');

                if (!records.length) {
                    if (chartWrapper) chartWrapper.style.display = 'none';
                    if (chartEmpty) chartEmpty.style.display = 'block';
                    if (summaryPills) summaryPills.style.display = 'none';
                } else {
                    if (chartWrapper) chartWrapper.style.display = 'block';
                    if (chartEmpty) chartEmpty.style.display = 'none';
                    if (summaryPills) summaryPills.style.display = 'flex';
                    renderTrendChart(records);
                    renderDashboardDoughnut(pos, neg, neu);
                    renderAspectAnalytics(records);
                    renderPriorityAnalytics(records);
                    renderDashboardRecentTable(records.slice(0, 5));
                }

                // Populate My Reviews & All Reviews Tables
                filterMyReviews();
                filterAllReviewsTable();

                if (isManualRefresh) showToast('Data refreshed successfully.');
            } catch (err) {
                console.error('loadHistory error:', err);
            }
        }

        function renderTrendChart(history) {
            if (typeof Chart === 'undefined') return;
            const canvas = document.getElementById('trendLineChart');
            if (!canvas) return;

            if (trendChartInstance) {
                trendChartInstance.destroy();
            }

            const chron = [...history].reverse();
            const labels = chron.map((r, i) => `#${r.id}`);
            const dataPoints = chron.map(r => r.vader_compound);
            const pointColors = chron.map(r => {
                if (r.sentiment === 'Positive') return '#16a34a';
                if (r.sentiment === 'Negative') return '#dc2626';
                return '#d97706';
            });

            const ctx = canvas.getContext('2d');
            trendChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'VADER Compound',
                        data: dataPoints,
                        borderColor: '#6D28D9',
                        backgroundColor: 'rgba(109, 40, 217, 0.05)',
                        pointBackgroundColor: pointColors,
                        pointBorderColor: '#FFFFFF',
                        pointBorderWidth: 1.5,
                        pointRadius: 4,
                        borderWidth: 2,
                        tension: 0.25,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { min: -1.0, max: 1.0, ticks: { stepSize: 0.5 } },
                        x: { grid: { display: false } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        function renderDashboardDoughnut(pos, neg, neu) {
            if (typeof Chart === 'undefined') return;
            const canvas = document.getElementById('dashboardDoughnutChart');
            if (!canvas) return;

            if (dashboardDoughnutInstance) dashboardDoughnutInstance.destroy();

            dashboardDoughnutInstance = new Chart(canvas, {
                type: 'doughnut',
                data: {
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{
                        data: [pos, neg, neu],
                        backgroundColor: ['#10B981', '#EF4444', '#F59E0B'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }

        function renderAspectAnalytics(records) {
            const container = document.getElementById('aspectAnalyticsBars');
            if (!container) return;

            let q = 0, s = 0, c = 0, p = 0;
            records.forEach(r => {
                const cats = (r.categories || '').toLowerCase();
                if (cats.includes('quality') || cats.includes('product')) q++;
                if (cats.includes('shipping') || cats.includes('delivery')) s++;
                if (cats.includes('support') || cats.includes('service')) c++;
                if (cats.includes('price') || cats.includes('pricing') || cats.includes('value')) p++;
            });

            const max = Math.max(q, s, c, p, 1);
            container.innerHTML = `
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>📦 Product Quality</span><span>${q} mentions</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(q/max)*100}%; background-color:#6D28D9;"></div></div>
                </div>
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>🚚 Delivery & Shipping</span><span>${s} mentions</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(s/max)*100}%; background-color:#3B82F6;"></div></div>
                </div>
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>💬 Customer Support</span><span>${c} mentions</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(c/max)*100}%; background-color:#10B981;"></div></div>
                </div>
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>💰 Pricing & Value</span><span>${p} mentions</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(p/max)*100}%; background-color:#F59E0B;"></div></div>
                </div>
            `;
        }

        function renderPriorityAnalytics(records) {
            const container = document.getElementById('priorityAnalyticsBars');
            if (!container) return;

            let urgent = 0, high = 0, med = 0, low = 0;
            records.forEach(r => {
                const p = (r.priority || 'Low').toLowerCase();
                if (p.includes('urgent')) urgent++;
                else if (p.includes('high')) high++;
                else if (p.includes('medium')) med++;
                else low++;
            });

            const total = records.length || 1;
            container.innerHTML = `
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>🚨 Urgent Priority</span><span>${urgent} (${Math.round((urgent/total)*100)}%)</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(urgent/total)*100}%; background-color:#DC2626;"></div></div>
                </div>
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>⚠️ High Priority</span><span>${high} (${Math.round((high/total)*100)}%)</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(high/total)*100}%; background-color:#F97316;"></div></div>
                </div>
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>📌 Medium Priority</span><span>${med} (${Math.round((med/total)*100)}%)</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(med/total)*100}%; background-color:#F59E0B;"></div></div>
                </div>
                <div class="progress-bar-row">
                    <div class="progress-bar-info"><span>✅ Low Priority</span><span>${low} (${Math.round((low/total)*100)}%)</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width:${(low/total)*100}%; background-color:#10B981;"></div></div>
                </div>
            `;
        }

        function renderDashboardRecentTable(records) {
            const tbody = document.getElementById('dashboardRecentTableBody');
            if (!tbody) return;
            if (!records.length) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:18px; color:var(--text-muted);">No reviews yet.</td></tr>';
                return;
            }

            tbody.innerHTML = records.map(r => {
                const s = r.sentiment || 'Neutral';
                let bClass = 'badge-neutral';
                if (s === 'Positive') bClass = 'badge-positive';
                else if (s === 'Negative') bClass = 'badge-negative';

                return `
                    <tr>
                        <td style="font-family:monospace; font-weight:600;">#${r.id}</td>
                        <td style="max-width:280px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(r.review)}</td>
                        <td><span class="sentence-badge-sm ${bClass}">${s}</span></td>
                        <td><span style="font-size:11px; font-weight:600; padding:2px 6px; border-radius:4px; background:#F1F5F9;">${r.priority || 'Low'}</span></td>
                        <td style="font-size:12px; color:var(--text-muted);">${formatDate(r.created_at)}</td>
                        <td><button class="btn-ghost" onclick="openReviewModal(${r.id})">View</button></td>
                    </tr>
                `;
            }).join('');
        }

        // My Reviews Filtering
        function setMyReviewsFilter(sentiment, btn) {
            activeMyReviewsFilter = sentiment;
            document.querySelectorAll('#view-my-reviews .sentence-filter-pill').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            filterMyReviews();
        }

        function filterMyReviews() {
            const tbody = document.getElementById('myReviewsTableBody');
            if (!tbody) return;

            const q = (document.getElementById('myReviewsSearch')?.value || '').toLowerCase();
            const sort = document.getElementById('myReviewsSort')?.value || 'newest';

            let list = [...currentHistoryData];
            if (activeMyReviewsFilter !== 'All') {
                list = list.filter(r => r.sentiment === activeMyReviewsFilter);
            }
            if (q) {
                list = list.filter(r => (r.review || '').toLowerCase().includes(q));
            }
            if (sort === 'oldest') {
                list.sort((a,b) => a.id - b.id);
            } else {
                list.sort((a,b) => b.id - a.id);
            }

            if (!list.length) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:30px; color:var(--text-muted);">No reviews found.</td></tr>';
                return;
            }

            tbody.innerHTML = list.map(r => {
                const s = r.sentiment || 'Neutral';
                let bClass = 'badge-neutral';
                if (s === 'Positive') bClass = 'badge-positive';
                else if (s === 'Negative') bClass = 'badge-negative';

                const conf = r.confidence ? `${parseFloat(r.confidence).toFixed(1)}%` : 'N/A';
                const comp = r.vader_compound !== undefined ? (r.vader_compound >= 0 ? '+' : '') + parseFloat(r.vader_compound).toFixed(2) : '--';

                return `
                    <tr>
                        <td style="font-family:monospace; font-weight:600;">#${r.id}</td>
                        <td style="max-width:280px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(r.review)}</td>
                        <td><span class="sentence-badge-sm ${bClass}">${s}</span></td>
                        <td style="font-family:monospace;">${comp}</td>
                        <td style="font-family:monospace;">${conf}</td>
                        <td><span style="font-size:11px; font-weight:600; padding:2px 6px; border-radius:4px; background:#F1F5F9;">${r.priority || 'Low'}</span></td>
                        <td style="font-size:12px; color:var(--text-muted);">${formatDate(r.created_at)}</td>
                        <td><button class="btn-secondary" style="font-size:12px; padding:4px 8px;" onclick="openReviewModal(${r.id})">Details</button></td>
                    </tr>
                `;
            }).join('');
        }

        // All Reviews Management Table Filtering & Pagination
        function filterAllReviewsTable() {
            const tbody = document.getElementById('historyTableBody');
            if (!tbody) return;

            const q = (document.getElementById('allReviewsSearch')?.value || '').toLowerCase();
            const sent = document.getElementById('filterSentiment')?.value || 'All';
            const aspect = document.getElementById('filterAspect')?.value || 'All';
            const prio = document.getElementById('filterPriority')?.value || 'All';
            const sort = document.getElementById('filterSort')?.value || 'newest';

            let list = [...currentHistoryData];

            if (sent !== 'All') list = list.filter(r => r.sentiment === sent);
            if (prio !== 'All') list = list.filter(r => (r.priority || '').toLowerCase() === prio.toLowerCase());
            if (aspect !== 'All') list = list.filter(r => (r.categories || '').toLowerCase().includes(aspect.toLowerCase()));
            if (q) list = list.filter(r => (r.review || '').toLowerCase().includes(q));

            if (sort === 'oldest') list.sort((a,b) => a.id - b.id);
            else if (sort === 'compound_high') list.sort((a,b) => b.vader_compound - a.vader_compound);
            else if (sort === 'compound_low') list.sort((a,b) => a.vader_compound - b.vader_compound);
            else list.sort((a,b) => b.id - a.id);

            const totalFiltered = list.length;
            const totalPages = Math.ceil(totalFiltered / historyPageSize) || 1;
            if (historyPage > totalPages) historyPage = 1;

            const startIdx = (historyPage - 1) * historyPageSize;
            const pageSlice = list.slice(startIdx, startIdx + historyPageSize);

            document.getElementById('allReviewsPaginationInfo').innerText = `Showing ${pageSlice.length} of ${totalFiltered} review${totalFiltered === 1 ? '' : 's'}`;
            document.getElementById('prevPageBtn').disabled = historyPage <= 1;
            document.getElementById('nextPageBtn').disabled = historyPage >= totalPages;

            if (!pageSlice.length) {
                tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:32px; color:var(--text-muted); font-size:13px;">No records match your filters.</td></tr>';
                return;
            }

            tbody.innerHTML = pageSlice.map(r => {
                const s = r.sentiment || 'Neutral';
                let bClass = 'badge-neutral';
                if (s === 'Positive') bClass = 'badge-positive';
                else if (s === 'Negative') bClass = 'badge-negative';

                const conf = r.confidence ? `${parseFloat(r.confidence).toFixed(1)}%` : 'N/A';
                const comp = r.vader_compound !== undefined ? (r.vader_compound >= 0 ? '+' : '') + parseFloat(r.vader_compound).toFixed(4) : '--';

                return `
                    <tr>
                        <td style="font-family:monospace; font-weight:600;">#${r.id}</td>
                        <td style="max-width:260px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(r.review)}</td>
                        <td><span class="sentence-badge-sm ${bClass}">${s}</span></td>
                        <td style="font-family:monospace;">${comp}</td>
                        <td style="font-family:monospace;">${conf}</td>
                        <td style="font-size:12px; color:var(--text-muted); max-width:140px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(r.categories || 'General')}</td>
                        <td><span style="font-size:11px; font-weight:600; padding:2px 6px; border-radius:4px; background:#F1F5F9;">${r.priority || 'Low'}</span></td>
                        <td style="font-size:12px; color:var(--text-muted);">${formatDate(r.created_at)}</td>
                        <td><button class="btn-secondary" style="font-size:12px; padding:4px 8px;" onclick="openReviewModal(${r.id})">Details</button></td>
                    </tr>
                `;
            }).join('');
        }

        function changeHistoryPage(delta) {
            historyPage += delta;
            filterAllReviewsTable();
        }

        // Modal Handlers
        function openReviewModal(id) {
            const modal = document.getElementById('reviewModal');
            if (!modal) return;

            let rec = currentHistoryData.find(r => r.id === id);
            if (!rec && currentBatchData && currentBatchData.detailed_results) {
                rec = currentBatchData.detailed_results.find(r => (r.review_id === id || r.id === id));
            }

            if (!rec) return;

            document.getElementById('modalReviewTitle').innerText = `Review #${rec.id || rec.review_id || ''}`;
            document.getElementById('modalReviewDate').innerText = rec.created_at ? formatDate(rec.created_at) : '';
            document.getElementById('modalReviewText').innerText = rec.review || rec.original_review || '';

            const s = rec.sentiment || 'Neutral';
            document.getElementById('modalSentiment').innerText = s;
            document.getElementById('modalCompound').innerText = (rec.vader_compound !== undefined ? rec.vader_compound : (rec.overall_compound_score || 0.0)).toFixed(2);
            document.getElementById('modalConfidence').innerText = rec.confidence ? `${parseFloat(rec.confidence).toFixed(1)}%` : 'N/A';
            document.getElementById('modalSubjectivity').innerText = (rec.subjectivity || 0.0).toFixed(2);
            document.getElementById('modalPriority').innerText = rec.priority || 'Low';

            const cats = Array.isArray(rec.categories) ? rec.categories : (rec.categories || 'General').split(',');
            document.getElementById('modalAspects').innerHTML = cats.map(c => `<span class="aspect-tag active">${escapeHtml(c.trim())}</span>`).join('');

            modal.style.display = 'flex';
        }

        function closeReviewModal() {
            const modal = document.getElementById('reviewModal');
            if (modal) modal.style.display = 'none';
        }

        function handleModalBackdrop(e) {
            if (e.target.id === 'reviewModal') closeReviewModal();
        }

        function openDocModal() {
            document.getElementById('docModal').style.display = 'flex';
        }

        function closeDocModal() {
            document.getElementById('docModal').style.display = 'none';
        }

        function handleDocModalBackdrop(e) {
            if (e.target.id === 'docModal') closeDocModal();
        }

        function escapeHtml(str) {
            if (!str) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function formatDate(dt) {
            if (!dt) return '--';
            try {
                const d = new Date(dt);
                return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            } catch(e) { return dt; }
        }

        // Keyboard accessibility
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeReviewModal();
                closeDocModal();
            }
        });

        // Initialize Data on Load
        document.addEventListener('DOMContentLoaded', function() {
            loadHistory();
        });
        window.addEventListener('load', function() {
            loadHistory();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content.replace("__SENTIQ_LOGO_SRC__", logo_src))
