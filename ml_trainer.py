import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Comprehensive balanced benchmark dataset for College NLP Lab
DATASET = [
    # Positive
    ("The battery life is phenomenal and screen resolution is gorgeous.", "Positive"),
    ("Fast delivery and packaging was very secure. Completely satisfied!", "Positive"),
    ("Outstanding sound quality, love the noise cancellation feature.", "Positive"),
    ("Super easy setup, works seamlessly with my laptop and phone.", "Positive"),
    ("Great value for money, definitely worth buying for everyday use.", "Positive"),
    ("Exceptional customer service, resolved my concern immediately.", "Positive"),
    ("Highly recommend this to all students and professionals!", "Positive"),
    ("Extremely happy with this order, flawless build quality.", "Positive"),
    ("Smooth performance and top notch aesthetic design.", "Positive"),
    ("Brilliant device, exceeds all my expectations, five stars!", "Positive"),
    ("The camera captures stunning vibrant photos in low light.", "Positive"),
    ("Comfortable fit and premium soft leather material.", "Positive"),
    ("The keyboard tactile feedback is amazingly satisfying.", "Positive"),
    ("Very durable and feels solidly built in hands.", "Positive"),
    ("App interface is clean, intuitive and lightning responsive.", "Positive"),

    # Negative
    ("Worst purchase ever, it stopped working in just two days.", "Negative"),
    ("Customer support is completely unresponsive and extremely rude.", "Negative"),
    ("Terrible experience. The screen was cracked and broken inside.", "Negative"),
    ("Defective item received. Returning it immediately for full refund.", "Negative"),
    ("Low build quality, cheap plastic used, highly disappointed.", "Negative"),
    ("Do not buy this! It overheats within 15 minutes of usage.", "Negative"),
    ("Horrible delivery delay, took more than three weeks to arrive.", "Negative"),
    ("Waste of hard-earned money. Absolute scam and useless.", "Negative"),
    ("Audio is buzzing and mic is barely audible. Very poor quality.", "Negative"),
    ("Slow charging and battery drains within one single hour.", "Negative"),
    ("Constantly crashes and refuses to connect to bluetooth.", "Negative"),
    ("Missing accessories and misleading product description.", "Negative"),
    ("Poor customer care, refused to replace defective piece.", "Negative"),
    ("Cheap materials used, broke apart on the third day.", "Negative"),
    ("Extremely noisy and overheats dangerously during charging.", "Negative"),

    # Neutral
    ("Decent product for this price, nothing extraordinary.", "Neutral"),
    ("The package arrived on Tuesday afternoon as scheduled.", "Neutral"),
    ("The color is slightly darker than the photos shown online.", "Neutral"),
    ("It comes with a standard USB-C cable and user manual.", "Neutral"),
    ("The dimensions are 10cm by 15cm as stated in specifications.", "Neutral"),
    ("Average performance, meets basic daily functional expectations.", "Neutral"),
    ("Standard functionality, neither too good nor exceptionally bad.", "Neutral"),
    ("The product weight is approximately 250 grams.", "Neutral"),
    ("Shipping box was undamaged and sealed with tape.", "Neutral"),
    ("Device boots up and displays standard system menu options.", "Neutral"),
    ("Package contains one unit, battery, and warranty flyer.", "Neutral"),
    ("The item matches the generic product description.", "Neutral"),
    ("Delivery took 4 business days which is standard.", "Neutral"),
    ("Device operates on standard 110-240V AC power supply.", "Neutral"),
    ("Normal build quality, as expected for an entry level gadget.", "Neutral")
]

LABELS = ["Positive", "Negative", "Neutral"]

def train_and_evaluate_models():
    df = pd.DataFrame(DATASET, columns=["review", "sentiment"])
    
    X = df["review"]
    y = df["sentiment"]
    
    # 70% Train, 30% Test stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Feature extraction with sublinear TF scaling
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 1. Multinomial Naive Bayes
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    
    # 2. Logistic Regression
    lr_model = LogisticRegression(C=2.0, max_iter=300)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    
    nb_acc = round(float(accuracy_score(y_test, nb_preds)), 4)
    lr_acc = round(float(accuracy_score(y_test, lr_preds)), 4)
    
    # Confusion Matrix (Ordered: Positive, Negative, Neutral)
    lr_cm = confusion_matrix(y_test, lr_preds, labels=LABELS).tolist()
    nb_cm = confusion_matrix(y_test, nb_preds, labels=LABELS).tolist()
    
    lr_metrics = {
        "model_name": "Logistic Regression (TF-IDF)",
        "accuracy": lr_acc,
        "precision_macro": round(float(precision_score(y_test, lr_preds, average='macro', zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, lr_preds, average='macro', zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, lr_preds, average='macro', zero_division=0)), 4),
        "confusion_matrix": lr_cm,
        "labels": LABELS
    }
    
    nb_metrics = {
        "model_name": "Multinomial Naive Bayes (TF-IDF)",
        "accuracy": nb_acc,
        "precision_macro": round(float(precision_score(y_test, nb_preds, average='macro', zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, nb_preds, average='macro', zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, nb_preds, average='macro', zero_division=0)), 4),
        "confusion_matrix": nb_cm,
        "labels": LABELS
    }
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(lr_model, "models/sentiment_lr_model.pkl")
    joblib.dump(nb_model, "models/sentiment_nb_model.pkl")
    df.to_csv("models/dataset_benchmark.csv", index=False)
    
    return {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "naive_bayes": nb_metrics,
        "logistic_regression": lr_metrics,
        "labels": LABELS
    }

if __name__ == "__main__":
    res = train_and_evaluate_models()
    print("SUCCESSFULLY_TRAINED_WITH_CONFUSION_MATRIX")
    print(res)
