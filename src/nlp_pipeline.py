"""NLP preprocessing, sentiment analysis, TF-IDF, and occupation clustering."""

import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


STOP_WORDS = set(stopwords.words("english"))
VADER = SentimentIntensityAnalyzer()


def clean_text(text):
    """Remove HTML tags, lowercase, remove punctuation and stopwords."""
    if pd.isna(text) or text == "":
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))  # remove HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def get_vader_sentiment(text):
    """Return VADER compound sentiment score [-1, 1]."""
    if not text or text == "":
        return 0.0
    return VADER.polarity_scores(text)["compound"]


def get_textblob_sentiment(text):
    """Return TextBlob polarity score [-1, 1]."""
    if not text or text == "":
        return 0.0
    return TextBlob(text).sentiment.polarity


def add_sentiment_features(df_pd, text_col):
    """Add VADER and TextBlob sentiment columns for a text field."""
    cleaned = df_pd[text_col].apply(clean_text)
    df_pd[f"{text_col}_vader"] = cleaned.apply(get_vader_sentiment)
    df_pd[f"{text_col}_textblob"] = cleaned.apply(get_textblob_sentiment)
    return df_pd


def build_tfidf_features(texts, max_features=100):
    """
    Fit TF-IDF on a text series and return feature matrix + feature names.

    Args:
        texts: pandas Series of cleaned text
        max_features: number of top TF-IDF features

    Returns:
        (sparse_matrix, feature_names, fitted_vectorizer)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts.fillna(""))
    return tfidf_matrix, vectorizer.get_feature_names_out(), vectorizer


def cluster_occupations(emp_titles, n_clusters=20):
    """
    Reduce 512K+ unique emp_title values into occupation clusters.

    Uses TF-IDF on job titles + KMeans clustering.

    Returns:
        cluster labels (numpy array)
    """
    cleaned = emp_titles.fillna("unknown").apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(cleaned)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(tfidf_matrix)
    return labels


def create_missing_text_flag(df_pd, col):
    """Create binary flag indicating whether a text field is missing."""
    df_pd[f"{col}_missing"] = (df_pd[col].isna() | (df_pd[col] == "")).astype(int)
    return df_pd
