import pytest
from nlp_engine import clean_and_tokenize, analyze_vader_sentiment, analyze_textblob_metrics, complete_review_analysis

def test_preprocessing():
    sample = "The battery was NOT working properly and stopped suddenly!!!"
    result = clean_and_tokenize(sample)
    assert result["total_words"] > 0
    assert "cleaned_text" in result
    assert isinstance(result["lemmatized_tokens"], list)

def test_positive_sentiment():
    pos_sample = "This is the best phone I have ever used. Absolutely fantastic!"
    result = analyze_vader_sentiment(pos_sample)
    assert result["sentiment"] == "Positive"
    assert result["compound_score"] > 0

def test_negative_sentiment():
    neg_sample = "Terrible product, completely defective and broke within hours. Waste of money."
    result = analyze_vader_sentiment(neg_sample)
    assert result["sentiment"] == "Negative"
    assert result["compound_score"] < 0

def test_neutral_sentiment():
    neu_sample = "The device dimensions are 15cm by 10cm."
    result = analyze_vader_sentiment(neu_sample)
    assert result["sentiment"] == "Neutral"

def test_complete_pipeline():
    review = "Great sound quality and fast delivery, but battery overheats."
    analysis = complete_review_analysis(review)
    assert "original_review" in analysis
    assert "sentiment" in analysis
    assert "confidence_percentage" in analysis
    assert "categories" in analysis
    assert "priority" in analysis
