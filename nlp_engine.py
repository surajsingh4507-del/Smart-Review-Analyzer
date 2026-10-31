import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize lemmatizer and sentiment analyzer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def clean_and_tokenize(text: str) -> dict:
    """
    Step 1: NLP Preprocessing Pipeline
    - Lowercasing
    - Regex cleaning (removing punctuation, special characters, extra whitespaces)
    - Tokenization
    - Stop-words separation
    - Lemmatization (WordNet dictionary root form)
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
    
    # 1. Lowercasing
    lowered = text.lower().strip()
    
    # 2. Remove punctuation and special characters
    clean_str = re.sub(r'[^a-zA-Z0-9\s]', ' ', lowered)
    clean_str = re.sub(r'\s+', ' ', clean_str).strip()
    
    # 3. Simple Tokenization (split by whitespace)
    raw_tokens = clean_str.split()
    
    # 4. Stopwords identification and removal
    stopwords_found = [t for t in raw_tokens if t in stop_words]
    tokens_no_stops = [t for t in raw_tokens if t not in stop_words and len(t) > 1]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens_no_stops]
    
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
    Analyzes individual words or tokens in the text using VADER lexicon
    to enable color-coded highlighting in the UI:
    - Positive words (green)
    - Negative words (red)
    - Neutral words (regular)
    """
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
            "is_stopword": w_lower in stop_words
        })
        
    return word_details

def analyze_vader_sentiment(text: str, threshold: float = 0.05) -> dict:
    """
    Rule-based Sentiment Analysis using VADER with adjustable cutoff threshold.
    """
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    # Dynamic threshold classification
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
    """
    Polarity and Subjectivity metrics using TextBlob.
    - Polarity: [-1.0 to 1.0]
    - Subjectivity: [0.0 (Objective/Fact) to 1.0 (Subjective/Personal Opinion)]
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
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
    """
    Aspect-based category classifier, keyword frequency extractor, and business recommendation.
    """
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
        
    # Urgency Priority & Automated Business Action Recommendation
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
    """
    Master pipeline unifying Preprocessing, Lexicon Sentiment (VADER), 
    Subjectivity (TextBlob), Word-level Tagging, Aspects, and Action Recommendations.
    """
    preprocessing = clean_and_tokenize(review_text)
    vader = analyze_vader_sentiment(review_text, threshold)
    textblob_metrics = analyze_textblob_metrics(review_text)
    aspects = extract_aspects_and_category(review_text)
    word_tokens = analyze_word_sentiments(review_text)
    
    # Confidence Score based on absolute compound polarity
    confidence = round(abs(vader["compound_score"]) * 100, 2)
    if confidence < 30.0:
        confidence = 65.0  # baseline confidence for weak signals
        
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
