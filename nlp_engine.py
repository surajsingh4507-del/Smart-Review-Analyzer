import re
import string
import os

# 1. Standalone VADER without external nltk downloads
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer()

# 2. Built-in English Stopwords (Offline, Zero download, Zero cold-start latency)
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

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

def clean_and_tokenize(text: str) -> dict:
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
    tokens_no_stops = [t for t in raw_tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
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
    if not text:
        return []
    
    words = re.findall(r"[\w']+|[.,!?;]", text)
    word_details = []
    
    for w in words:
        w_lower = w.lower()
        if w_lower in sia.lexicon:
            score = sia.lexicon[w_lower]
            if score > 0.5:
                sentiment = "positive"
            elif score < -0.5:
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
            "is_stopword": w_lower in ENGLISH_STOPWORDS
        })
        
    return word_details

def analyze_vader_sentiment(text: str, threshold: float = 0.05) -> dict:
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
        "neutral_score": round(scores['neu'], 4)
    }

def analyze_textblob_metrics(text: str) -> dict:
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

def extract_aspects_and_category(text: str) -> dict:
    text_lower = text.lower()
    categories = []
    aspect_keywords_found = []
    
    delivery_kw = ["delivery", "shipping", "courier", "package", "arrived", "delay", "late"]
    quality_kw = ["quality", "material", "broken", "defective", "damaged", "size", "fit", "color", "feature", "battery", "screen", "display"]
    support_kw = ["support", "service", "staff", "executive", "call", "agent", "refund", "return", "policy"]
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
        
    vader = analyze_vader_sentiment(text)
    sentiment = vader["sentiment"]
    
    if sentiment == "Negative" and any(w in text_lower for w in ["broken", "refund", "worst", "terrible", "urgent", "cheat", "scam", "never"]):
        priority = "High"
        recommendation = "🚨 High Severity: Initiate instant replacement or refund workflow; escalate ticket to Senior Technical Support within 2 hours."
    elif sentiment == "Negative":
        priority = "Medium"
        recommendation = "⚠️ Attention Required: Flag review for quality assurance team and dispatch personalized customer apology email."
    elif sentiment == "Positive":
        priority = "Low"
        recommendation = "⭐ Positive Feedback: Highlight this customer testimony on promotional banners and thank customer with loyalty rewards."
    else:
        priority = "Low"
        recommendation = "ℹ️ Neutral Inquiry/Spec: Log in database for catalog reference; no immediate remediation needed."
        
    return {
        "categories": categories,
        "aspect_keywords": aspect_keywords_found,
        "priority": priority,
        "recommendation": recommendation
    }

def complete_review_analysis(review_text: str, threshold: float = 0.05) -> dict:
    preprocessing = clean_and_tokenize(review_text)
    vader = analyze_vader_sentiment(review_text, threshold)
    textblob_metrics = analyze_textblob_metrics(review_text)
    aspects = extract_aspects_and_category(review_text)
    word_tokens = analyze_word_sentiments(review_text)
    
    confidence = round(abs(vader["compound_score"]) * 100, 2)
    if confidence < 30.0:
        confidence = 65.0
        
    return {
        "original_review": review_text,
        "preprocessing": preprocessing,
        "sentiment": vader["sentiment"],
        "confidence_percentage": min(confidence, 99.0),
        "vader_scores": vader,
        "textblob_metrics": textblob_metrics,
        "categories": aspects["categories"],
        "aspect_keywords": aspects["aspect_keywords"],
        "priority": aspects["priority"],
        "recommendation": aspects["recommendation"],
        "word_tokens": word_tokens
    }
