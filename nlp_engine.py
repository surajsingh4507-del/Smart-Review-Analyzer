import re
import string
import os
from typing import Optional, Dict, Any, List

# 1. Standalone VADER without external nltk downloads
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer()

# Domain sentiment augmentations for consumer and e-commerce reviews
# Ensures domain words that VADER lacks (such as 'late' as a complaint or 'overheats') are properly scored
DOMAIN_LEXICON_AUGMENTS = {
    "late": -1.5,
    "overheats": -2.0,
    "overheating": -2.0,
    "cracked": -2.0,
    "unresponsive": -2.0,
    "helpful": 2.0,
    "responsive": 1.8,
    "defective": -2.5
}
for word, score in DOMAIN_LEXICON_AUGMENTS.items():
    if word not in sia.lexicon:
        sia.lexicon[word] = score

# 2. English Negation Words - MUST NEVER be stripped during sentiment-bearing text processing
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nowhere", "nothing",
    "aren't", "can't", "cannot", "couldn't", "didn't", "doesn't", "don't",
    "hadn't", "hasn't", "haven't", "isn't", "mustn't", "shan't", "shouldn't",
    "wasn't", "weren't", "won't", "wouldn't", "hardly", "scarcely", "barely", "without"
}

# 3. Built-in English Stopwords (Offline, Zero download, Zero cold-start latency)
# Negations are isolated so they can be preserved when desired
ENGLISH_BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my",
    "myself", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
    "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}

# Combined set for general filtering while keeping track of negations
ENGLISH_STOPWORDS = ENGLISH_BASE_STOPWORDS.union(NEGATION_WORDS)


def morphological_lemmatize(word: str) -> str:
    """
    Robust rule-based morphological lemmatizer for key suffixes.
    Completely offline and fast for serverless lambda environments.
    """
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 4:
        return w[:-3]
    if w.endswith("ed") and len(w) > 3:
        return w[:-2]
    return w


def clean_and_tokenize(text: str, preserve_negations: bool = True) -> dict:
    """
    Cleans and tokenizes text. When preserve_negations is True (default),
    critical sentiment negation words like 'not', 'no', 'never' are retained
    in tokens_no_stops and cleaned_text so downstream sentiment is preserved.
    """
    if not text or not text.strip():
        return {
            "cleaned_text": "",
            "tokens": [],
            "lemmatized_tokens": [],
            "stopwords_found": [],
            "tokens_retained": [],
            "total_words": 0
        }
    
    lowered = text.lower().strip()
    clean_str = re.sub(r'[^a-zA-Z0-9\s]', ' ', lowered)
    clean_str = re.sub(r'\s+', ' ', clean_str).strip()
    raw_tokens = clean_str.split()
    
    stopwords_found = [t for t in raw_tokens if t in ENGLISH_STOPWORDS]
    
    # Filter stopwords, but keep negations if preserve_negations is enabled
    effective_stops = ENGLISH_BASE_STOPWORDS if preserve_negations else ENGLISH_STOPWORDS
    tokens_no_stops = [t for t in raw_tokens if t not in effective_stops and len(t) > 1]
    lemmatized = [morphological_lemmatize(t) for t in tokens_no_stops]
    
    return {
        "cleaned_text": " ".join(lemmatized),
        "tokens": raw_tokens,
        "stopwords_found": stopwords_found,
        "tokens_retained": tokens_no_stops,
        "lemmatized_tokens": lemmatized,
        "total_words": len(raw_tokens)
    }


def analyze_word_sentiments(text: str) -> list:
    """
    Tokenizes input text and scores individual words using VADER lexicon.
    """
    if not text:
        return []
    
    words = re.findall(r"[\w']+|[.,!?;]", text)
    word_details = []
    
    for w in words:
        w_lower = w.lower()
        if w_lower in sia.lexicon:
            score = sia.lexicon[w_lower]
            if score > 0.4:
                sentiment = "positive"
            elif score < -0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            score = 0.0
            sentiment = "neutral"
            
        word_details.append({
            "word": w,
            "sentiment": sentiment,
            "score": round(score, 2),
            "is_stopword": w_lower in ENGLISH_STOPWORDS and w_lower not in NEGATION_WORDS
        })
        
    return word_details


def analyze_vader_sentiment(text: str, threshold: float = 0.05) -> dict:
    """
    Analyzes sentiment using NLTK VADER polarity scores.
    Uses configurable threshold (default 0.05).
    """
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= threshold:
        sentiment = "Positive"
    elif compound <= -threshold:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return {
        "sentiment": sentiment,
        "threshold_used": threshold,
        "compound_score": round(compound, 4),
        "positive_score": round(scores['pos'], 4),
        "negative_score": round(scores['neg'], 4),
        "neutral_score": round(scores['neu'], 4),
        "compound": round(compound, 4),
        "pos": round(scores['pos'], 4),
        "neg": round(scores['neg'], 4),
        "neu": round(scores['neu'], 4)
    }


def analyze_textblob_metrics(text: str) -> dict:
    """
    Analyzes sentiment polarity and subjectivity using TextBlob.
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity = 0.0
        subjectivity = 0.0
    
    if polarity > 0.1:
        tb_sentiment = "Positive"
    elif polarity < -0.1:
        tb_sentiment = "Negative"
    else:
        tb_sentiment = "Neutral"
        
    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "tb_sentiment": tb_sentiment,
        "assessment": "Subjective (Opinion-based)" if subjectivity >= 0.5 else "Objective (Fact-based)"
    }


def predict_ml_sentiment(
    text: str,
    vectorizer=None,
    lr_model=None,
    nb_model=None
) -> dict:
    """
    Predicts sentiment using saved machine-learning models (Logistic Regression & Naive Bayes).
    Obtains genuine class probabilities via predict_proba() when supported.
    Gracefully handles missing or uninitialized models.
    """
    if vectorizer is None or lr_model is None or nb_model is None:
        return {
            "available": False,
            "logistic_regression": "N/A",
            "naive_bayes": "N/A",
            "probabilities": None,
            "confidence_percentage": None,
            "confidence_model": "N/A",
            "message": "ML models are not loaded"
        }
    
    try:
        vec = vectorizer.transform([text])
        lr_pred = str(lr_model.predict(vec)[0])
        nb_pred = str(nb_model.predict(vec)[0])
        
        # Genuine probabilities from calibrated model
        lr_prob = None
        lr_conf_pct = None
        if hasattr(lr_model, "predict_proba"):
            classes = list(lr_model.classes_)
            probs = lr_model.predict_proba(vec)[0]
            lr_prob = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
            if lr_pred in lr_prob:
                lr_conf_pct = round(lr_prob[lr_pred] * 100, 1)

        nb_prob = None
        if hasattr(nb_model, "predict_proba"):
            classes_nb = list(nb_model.classes_)
            probs_nb = nb_model.predict_proba(vec)[0]
            nb_prob = {classes_nb[i]: round(float(probs_nb[i]), 4) for i in range(len(classes_nb))}

        return {
            "available": True,
            "logistic_regression": lr_pred,
            "naive_bayes": nb_pred,
            "probabilities": lr_prob,
            "nb_probabilities": nb_prob,
            "confidence_percentage": lr_conf_pct,
            "confidence_model": "Logistic Regression (predict_proba)" if lr_conf_pct is not None else "N/A"
        }
    except Exception as e:
        return {
            "available": False,
            "logistic_regression": "Error",
            "naive_bayes": "Error",
            "probabilities": None,
            "confidence_percentage": None,
            "confidence_model": "N/A",
            "error": str(e)
        }


def resolve_final_sentiment(
    vader_res: dict,
    textblob_res: dict,
    ml_res: dict,
    threshold: float = 0.05
) -> dict:
    """
    Multi-engine consensus resolver that determines the final sentiment label.
    Resolves disagreements between VADER, TextBlob, and ML models transparently.
    
    Resolution Strategy:
    1. If VADER has decisive polarity (compound >= threshold or <= -threshold):
       - If ML agrees with VADER, consensus is high-confidence.
       - If ML disagrees: check TextBlob. If TextBlob agrees with ML (2 out of 3),
         the majority vote prevails. Otherwise VADER (tuned for consumer reviews) prevails.
    2. If VADER is Neutral:
       - If TextBlob and ML both agree on a non-neutral sentiment, that sentiment is adopted
         (handles phrases where VADER's general lexicon lacked domain terms).
       - If ML is confident and TextBlob leans in the same direction, ML sentiment is adopted.
       - Otherwise, final sentiment is Neutral.
    """
    vader_sent = vader_res["sentiment"]
    tb_sent = textblob_res["tb_sentiment"]
    tb_polarity = textblob_res["polarity"]
    
    ml_available = ml_res.get("available", False)
    lr_sent = ml_res.get("logistic_regression") if ml_available else None
    nb_sent = ml_res.get("naive_bayes") if ml_available else None
    
    disagreements = []
    if ml_available and lr_sent in ["Positive", "Negative", "Neutral"] and lr_sent != vader_sent:
        disagreements.append(f"VADER ({vader_sent}) vs Logistic Regression ({lr_sent})")
    if tb_sent != vader_sent:
        disagreements.append(f"VADER ({vader_sent}) vs TextBlob ({tb_sent})")
    if ml_available and lr_sent and tb_sent != lr_sent:
        disagreements.append(f"TextBlob ({tb_sent}) vs Logistic Regression ({lr_sent})")

    final_sentiment = vader_sent
    decision_reason = "VADER polarity alignment"

    if vader_sent in ["Positive", "Negative"]:
        if ml_available and lr_sent in ["Positive", "Negative"] and lr_sent != vader_sent:
            # VADER and ML disagree. Check TextBlob as tiebreaker.
            if tb_sent == lr_sent:
                final_sentiment = lr_sent
                decision_reason = f"Majority consensus (TextBlob & ML agreed on {lr_sent})"
            else:
                final_sentiment = vader_sent
                decision_reason = "VADER review lexicon prioritization"
        else:
            final_sentiment = vader_sent
            decision_reason = "VADER compound threshold exceeded"
    else:
        # VADER is Neutral: check if ML + TextBlob detect clear sentiment
        if ml_available and lr_sent in ["Positive", "Negative"]:
            if tb_sent == lr_sent:
                final_sentiment = lr_sent
                decision_reason = f"Consensus between ML and TextBlob ({lr_sent}) over Neutral VADER"
            elif (lr_sent == "Positive" and tb_polarity > 0.05) or (lr_sent == "Negative" and tb_polarity < -0.05):
                final_sentiment = lr_sent
                decision_reason = f"ML ({lr_sent}) supported by TextBlob polarity ({tb_polarity})"
            else:
                final_sentiment = "Neutral"
                decision_reason = "VADER Neutral confirmed by conflicting secondary signals"
        elif tb_sent in ["Positive", "Negative"] and abs(tb_polarity) >= 0.25:
            final_sentiment = tb_sent
            decision_reason = f"Strong TextBlob polarity ({tb_polarity}) over Neutral VADER"
        else:
            final_sentiment = "Neutral"
            decision_reason = "All engines indicate Neutral sentiment"

    return {
        "final_sentiment": final_sentiment,
        "decision_reason": decision_reason,
        "has_disagreement": len(disagreements) > 0,
        "disagreement_details": disagreements,
        "comparison": {
            "vader": vader_sent,
            "vader_compound": vader_res["compound_score"],
            "textblob": tb_sent,
            "textblob_polarity": tb_polarity,
            "logistic_regression": lr_sent or "N/A",
            "naive_bayes": nb_sent or "N/A",
            "has_disagreement": len(disagreements) > 0,
            "disagreement_details": disagreements
        }
    }


def extract_aspects_and_category(
    text: str,
    determined_sentiment: Optional[str] = None,
    compound_score: Optional[float] = None
) -> dict:
    """
    Extracts domain aspect categories, keywords, and computes priority recommendation.
    
    Priority Classification Rule (based strictly on sentiment severity and operational urgency):
      - High Priority: Severe negative sentiment (compound score <= -0.50 OR negative sentiment
        containing critical escalation keywords like 'broken', 'refund', 'worst', 'terrible',
        'urgent', 'cheat', 'scam', 'defective', 'damaged').
      - Medium Priority: Moderate negative sentiment (-0.50 < compound score < -0.05, or
        general negative customer review requiring remediation/QA review).
      - Low Priority: Neutral inquiries/specifications (-0.05 <= compound score <= 0.05) or
        positive customer feedback (compound score >= 0.05).
    """
    text_lower = text.lower()
    categories = []
    aspect_keywords_found = []
    
    delivery_kw = ["delivery", "shipping", "courier", "package", "arrived", "delay", "late"]
    quality_kw = ["quality", "material", "broken", "defective", "damaged", "size", "fit", "color", "feature", "battery", "screen", "display"]
    support_kw = ["support", "service", "staff", "executive", "call", "agent", "refund", "return", "policy", "help", "helpful", "care"]
    pricing_kw = ["price", "cost", "expensive", "cheap", "worth", "value", "money", "pricing"]
    
    for kw in delivery_kw:
        if kw in text_lower:
            aspect_keywords_found.append({"keyword": kw, "aspect": "Delivery"})
    if any(kw in text_lower for kw in delivery_kw):
        categories.append("Delivery & Shipping")
        
    for kw in quality_kw:
        if kw in text_lower:
            aspect_keywords_found.append({"keyword": kw, "aspect": "Quality"})
    if any(kw in text_lower for kw in quality_kw):
        categories.append("Product Quality")
        
    for kw in support_kw:
        if kw in text_lower:
            aspect_keywords_found.append({"keyword": kw, "aspect": "Customer Support"})
    if any(kw in text_lower for kw in support_kw):
        categories.append("Customer Service")
        
    for kw in pricing_kw:
        if kw in text_lower:
            aspect_keywords_found.append({"keyword": kw, "aspect": "Pricing"})
    if any(kw in text_lower for kw in pricing_kw):
        categories.append("Pricing & Value")
        
    if not categories:
        categories.append("General Feedback")
        
    # Use provided sentiment if available, otherwise analyze with VADER
    vader_res_aspect = analyze_vader_sentiment(text)
    sentiment = determined_sentiment or vader_res_aspect['sentiment']
    if compound_score is None:
        compound_score = vader_res_aspect['compound_score']

    severe_escalation_keywords = ['broken', 'refund', 'worst', 'terrible', 'urgent', 'cheat', 'scam', 'never', 'late', 'defective', 'damaged']
    # Priority based strictly on sentiment severity and operational impact
    if sentiment == 'Negative' and (compound_score <= -0.50 or any(w in text_lower for w in severe_escalation_keywords)):
        priority = 'High'
        recommendation = '🚨 High Severity: Initiate instant replacement or refund workflow; escalate ticket to Senior Technical Support within 2 hours.'
    elif sentiment == 'Negative' or compound_score < -0.05:
        priority = 'Medium'
        recommendation = '⚡ Attention Required: Flag review for quality assurance team and dispatch personalized customer apology email.'
    elif sentiment == 'Positive':
        priority = 'Low'
        recommendation = '⭐ Positive Feedback: Highlight this customer testimony on promotional banners and thank customer with loyalty rewards.'
    else:
        priority = 'Low'
        recommendation = 'ℹ️ Neutral Inquiry/Spec: Log in database for catalog reference; no immediate remediation needed.'
    return {
        "categories": categories,
        "aspect_keywords": aspect_keywords_found,
        "priority": priority,
        "recommendation": recommendation
    }


def explain_sentiment_result(
    review_text: str,
    sentiment: str,
    word_tokens: list,
    vader_scores: dict,
    textblob_metrics: dict,
    decision_reason: str = ""
) -> str:
    """
    Generates a clear, truthful explanation of the sentiment classification
    based on identified polarized words, negation patterns, and model consensus.
    """
    pos_words = [w["word"].lower() for w in word_tokens if w.get("sentiment") == "positive" and not w.get("is_stopword")]
    neg_words = [w["word"].lower() for w in word_tokens if w.get("sentiment") == "negative" and not w.get("is_stopword")]
    
    pos_words = list(dict.fromkeys(pos_words))
    neg_words = list(dict.fromkeys(neg_words))
    
    text_lower = review_text.lower()
    negation_patterns = ["not good", "not bad", "not worth", "never buy", "no issues", "no problem", "not working"]
    found_negations = [p for p in negation_patterns if p in text_lower]

    compound = vader_scores.get("compound_score", 0.0)
    polarity = textblob_metrics.get("polarity", 0.0)

    if sentiment == "Positive":
        reasons = []
        if pos_words:
            reasons.append(f"favorable terms ({', '.join(repr(w) for w in pos_words[:4])})")
        if any("not bad" in n or "no issue" in n or "no problem" in n for n in found_negations):
            reasons.append(f"mitigating negation ({', '.join(repr(n) for n in found_negations)})")
        if compound >= 0.05:
            reasons.append(f"VADER compound score +{compound:.2f}")
        reason_str = f"driven by {'; '.join(reasons)}" if reasons else "positive overall polarity"
        return f"Positive sentiment: {reason_str}."
    elif sentiment == "Negative":
        reasons = []
        if any(n in text_lower for n in ["not good", "not worth", "not working"]):
            reasons.append(f"critical negation ({', '.join(repr(n) for n in found_negations if n in ['not good', 'not worth', 'not working'])})")
        if neg_words:
            reasons.append(f"critical terms ({', '.join(repr(w) for w in neg_words[:4])})")
        if compound <= -0.05:
            reasons.append(f"negative compound score {compound:.2f}")
        elif polarity <= -0.05:
            reasons.append(f"TextBlob polarity {polarity:.2f}")
        reason_str = f"driven by {'; '.join(reasons)}" if reasons else "critical overall sentiment"
        return f"Negative sentiment: {reason_str}."
    else:
        if pos_words and neg_words:
            return f"Neutral sentiment: Mixed tone where positive ({', '.join(pos_words[:2])}) and critical ({', '.join(neg_words[:2])}) expressions counterbalance each other."
        return f"Neutral sentiment: Objective/factual statement with no strong polarized emotional terms (compound score {compound:.2f})."


def segment_sentences(text: str) -> List[str]:
    """
    Splits text into meaningful sentences preserving punctuation, abbreviations,
    and decimal numbers (e.g., '6.5-inch', '128 GB').
    Uses NLTK sent_tokenize with a deterministic regex fallback.
    """
    if not text or not str(text).strip():
        return []
    
    cleaned = str(text).strip()

    # Pre-protect decimals like 6.5 and abbreviations like e.g., i.e., Dr.
    protected = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL_DOT>\2', cleaned)
    for abbr in ['e.g.', 'i.e.', 'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'vs.', 'inc.', 'ltd.', 'co.', 'etc.', 'approx.']:
        pattern = re.compile(r'\b' + re.escape(abbr), re.IGNORECASE)
        protected = pattern.sub(lambda m: m.group(0).replace('.', '<ABBR_DOT>'), protected)
    
    # 1. Try NLTK sent_tokenize
    try:
        import nltk
        sentences = nltk.sent_tokenize(protected)
        if sentences:
            result = [s.replace('<DECIMAL_DOT>', '.').replace('<ABBR_DOT>', '.').strip() for s in sentences if s.strip()]
            if result:
                return result
    except Exception as e:
        logger.warning(f"NLTK sent_tokenize fallback: {e}")

    # 2. Robust Regex Fallback
    parts = re.split(r'(?<=[.!?])\s+', protected)
    results = [p.replace('<DECIMAL_DOT>', '.').replace('<ABBR_DOT>', '.').strip() for p in parts if p.strip()]
    return results if results else [cleaned]


def analyze_sentence(
    sentence: str,
    threshold: float = 0.05,
    vectorizer=None,
    lr_model=None,
    nb_model=None
) -> Optional[Dict[str, Any]]:
    """
    Analyzes an individual sentence:
    - VADER scores (pos, neg, neu, compound)
    - Threshold-based sentiment (Positive if compound > T, Negative if compound < -T, else Neutral)
    - TextBlob polarity & subjectivity
    - ML predictions (Logistic Regression & Naive Bayes if available)
    """
    s_clean = sentence.strip()
    if not s_clean:
        return None
        
    vader_res = analyze_vader_sentiment(s_clean, threshold=threshold)
    compound = vader_res["compound_score"]
    
    # User-configured threshold formula:
    # compound > T: Positive
    # compound < -T: Negative
    # Otherwise: Neutral
    if compound > threshold:
        sentiment = "Positive"
    elif compound < -threshold:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    tb_res = analyze_textblob_metrics(s_clean)
    ml_preds = predict_ml_sentiment(s_clean, vectorizer, lr_model, nb_model)

    return {
        "text": s_clean,
        "sentiment": sentiment,
        "vader_scores": {
            "pos": vader_res["positive_score"],
            "neg": vader_res["negative_score"],
            "neu": vader_res["neutral_score"],
            "compound": compound,
            "compound_score": compound,
            "positive_score": vader_res["positive_score"],
            "negative_score": vader_res["negative_score"],
            "neutral_score": vader_res["neutral_score"]
        },
        "vader_compound": compound,
        "textblob_polarity": tb_res["polarity"],
        "textblob_subjectivity": tb_res["subjectivity"],
        "ml_prediction": ml_preds.get("logistic_regression") if ml_preds.get("available") else None,
        "ml_predictions": ml_preds
    }


def analyze_paragraph_sentences(
    text: str,
    threshold: float = 0.05,
    vectorizer=None,
    lr_model=None,
    nb_model=None
) -> Dict[str, Any]:
    """
    Splits a paragraph or text into sentences, analyzes each sentence individually,
    calculates Positive, Negative, and Neutral sentence distribution and percentages,
    and calculates the overall compound score and final overall sentiment.
    """
    raw_sentences = segment_sentences(text)
    if not raw_sentences:
        return {
            "total_sentences": 0,
            "valid_sentences": 0,
            "failed_sentences": 0,
            "positive_sentences": 0,
            "negative_sentences": 0,
            "neutral_sentences": 0,
            "positive_percentage": 0.0,
            "negative_percentage": 0.0,
            "neutral_percentage": 0.0,
            "sentiment_distribution": {
                "Positive": 0,
                "Negative": 0,
                "Neutral": 0,
                "Positive_Percentage": 0.0,
                "Negative_Percentage": 0.0,
                "Neutral_Percentage": 0.0
            },
            "overall_compound_score": 0.0,
            "mean_compound_score": 0.0,
            "overall_sentiment": "Neutral",
            "threshold": threshold,
            "explanation": "No sentences could be identified in the text.",
            "sentences": []
        }

    analyzed_sentences = []
    failed_count = 0
    valid_compound_scores = []

    for idx, s in enumerate(raw_sentences, start=1):
        try:
            s_data = analyze_sentence(s, threshold, vectorizer, lr_model, nb_model)
            if s_data:
                s_data["id"] = idx
                s_data["sentence_id"] = f"S{idx}"
                analyzed_sentences.append(s_data)
                valid_compound_scores.append(s_data["vader_compound"])
            else:
                failed_count += 1
        except Exception as e:
            logger.warning(f"Failed to analyze sentence #{idx}: {e}")
            failed_count += 1

    total_valid = len(analyzed_sentences)

    pos_count = sum(1 for s in analyzed_sentences if s["sentiment"] == "Positive")
    neg_count = sum(1 for s in analyzed_sentences if s["sentiment"] == "Negative")
    neu_count = sum(1 for s in analyzed_sentences if s["sentiment"] == "Neutral")

    # Formula from prompt:
    # positive_percentage = positive_sentence_count / total_analyzed_sentences * 100
    # negative_percentage = negative_sentence_count / total_analyzed_sentences * 100
    # neutral_percentage = neutral_sentence_count / total_analyzed_sentences * 100
    pos_pct = round((pos_count / total_valid) * 100, 1) if total_valid > 0 else 0.0
    neg_pct = round((neg_count / total_valid) * 100, 1) if total_valid > 0 else 0.0
    neu_pct = round((neu_count / total_valid) * 100, 1) if total_valid > 0 else 0.0

    if valid_compound_scores:
        overall_compound = round(sum(valid_compound_scores) / len(valid_compound_scores), 4)
    else:
        overall_compound = 0.0

    # Determine final sentiment using selected threshold T:
    # Mean score > T: Overall Positive
    # Mean score < -T: Overall Negative
    # Otherwise: Overall Neutral
    if overall_compound > threshold:
        overall_sentiment = "Positive"
        calc_reason = f"Mean sentence compound score (+{overall_compound:.4f}) exceeds threshold (+{threshold:.2f})."
    elif overall_compound < -threshold:
        overall_sentiment = "Negative"
        calc_reason = f"Mean sentence compound score ({overall_compound:.4f}) is below negative threshold (-{threshold:.2f})."
    else:
        overall_sentiment = "Neutral"
        calc_reason = f"Mean sentence compound score ({overall_compound:+.4f}) falls within neutral boundary [-{threshold:.2f}, +{threshold:.2f}]."

    distribution = {
        "Positive": pos_count,
        "Negative": neg_count,
        "Neutral": neu_count,
        "Positive_Percentage": pos_pct,
        "Negative_Percentage": neg_pct,
        "Neutral_Percentage": neu_pct
    }

    return {
        "total_sentences": len(raw_sentences),
        "valid_sentences": total_valid,
        "failed_sentences": failed_count,
        "positive_sentences": pos_count,
        "negative_sentences": neg_count,
        "neutral_sentences": neu_count,
        "positive_percentage": pos_pct,
        "negative_percentage": neg_pct,
        "neutral_percentage": neu_pct,
        "sentiment_distribution": distribution,
        "overall_compound_score": overall_compound,
        "mean_compound_score": overall_compound,
        "overall_sentiment": overall_sentiment,
        "threshold": threshold,
        "explanation": f"Overall {overall_sentiment}: {calc_reason}",
        "sentences": analyzed_sentences
    }


def complete_review_analysis(
    review_text: str,
    threshold: float = 0.05,
    vectorizer=None,
    lr_model=None,
    nb_model=None
) -> dict:
    """
    Performs comprehensive review analysis combining sentence-level segmentation,
    VADER, TextBlob, ML models, aspect extraction, and overall aggregation.
    """
    sentence_analysis = analyze_paragraph_sentences(
        review_text, threshold, vectorizer, lr_model, nb_model
    )
    
    preprocessing = clean_and_tokenize(review_text)
    vader = analyze_vader_sentiment(review_text, threshold)
    textblob_metrics = analyze_textblob_metrics(review_text)
    ml_predictions = predict_ml_sentiment(review_text, vectorizer, lr_model, nb_model)
    
    consensus = resolve_final_sentiment(vader, textblob_metrics, ml_predictions, threshold)
    
    # If text contains multiple sentences, use sentence-aggregated overall sentiment
    if sentence_analysis["total_sentences"] > 1:
        final_sentiment = sentence_analysis["overall_sentiment"]
        overall_compound = sentence_analysis["overall_compound_score"]
        pos_c = sentence_analysis["sentiment_distribution"]["Positive"]
        neg_c = sentence_analysis["sentiment_distribution"]["Negative"]
        neu_c = sentence_analysis["sentiment_distribution"]["Neutral"]
        pos_p = sentence_analysis["sentiment_distribution"]["Positive_Percentage"]
        neg_p = sentence_analysis["sentiment_distribution"]["Negative_Percentage"]
        neu_p = sentence_analysis["sentiment_distribution"]["Neutral_Percentage"]
        explanation = (
            f"Overall {final_sentiment}: Mean VADER compound score ({overall_compound:+.4f}) "
            f"across {sentence_analysis['total_sentences']} sentences (threshold ±{threshold:.2f}). "
            f"Distribution: {pos_c} Positive ({pos_p}%), {neg_c} Negative ({neg_p}%), {neu_c} Neutral ({neu_p}%)."
        )
    else:
        final_sentiment = consensus["final_sentiment"]
        overall_compound = vader["compound_score"]
        word_tokens_temp = analyze_word_sentiments(review_text)
        explanation = explain_sentiment_result(
            review_text, final_sentiment, word_tokens_temp, vader, textblob_metrics, consensus["decision_reason"]
        )

    aspects = extract_aspects_and_category(review_text, final_sentiment, compound_score=overall_compound)
    word_tokens = analyze_word_sentiments(review_text)
    
    # Model Confidence Metric:
    # Must represent genuine predicted class probability from a calibrated model (Logistic Regression / Naive Bayes).
    # If genuine model probability is unavailable, set to None (rendered as 'N/A' in UI/exports).
    # VADER compound score is kept strictly separate in vader_scores and overall_compound_score.
    ml_confidence = None
    confidence_model = "N/A"
    if ml_predictions.get("available") and ml_predictions.get("confidence_percentage") is not None:
        ml_confidence = ml_predictions["confidence_percentage"]
        confidence_model = ml_predictions.get("confidence_model", "Logistic Regression (predict_proba)")

    return {
        "original_review": review_text,
        "preprocessing": preprocessing,
        "sentiment": final_sentiment,
        "final_sentiment": final_sentiment,
        "overall_compound_score": overall_compound,
        "confidence_percentage": ml_confidence,
        "confidence_model": confidence_model,
        "ml_confidence": ml_confidence,
        "explanation": explanation,
        "sentence_analysis": sentence_analysis,
        "vader_scores": vader,
        "textblob_metrics": textblob_metrics,
        "ml_model_predictions": ml_predictions,
        "model_comparison": consensus["comparison"],
        "decision_reason": consensus["decision_reason"],
        "categories": aspects["categories"],
        "aspect_keywords": aspects["aspect_keywords"],
        "priority": aspects["priority"],
        "recommendation": aspects["recommendation"],
        "word_tokens": word_tokens
    }
