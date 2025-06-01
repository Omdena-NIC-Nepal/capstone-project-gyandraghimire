# nlp_utils.py 🔤 NLP Utilities for Climate News Analysis

import requests
import logging
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

# --- Setup ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
logging.basicConfig(level=logging.INFO, format="🔍 %(message)s")

# --- Load HuggingFace Transformers Pipelines ---
sentiment_en = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- Web Article Fetcher ---
def fetch_articles(urls):
    """
    Scrape article text from a list of URLs.
    Returns a list of raw article texts.
    """
    articles = []
    for url in urls:
        try:
            logging.info(f"📡 Fetching: {url}")
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            text = " ".join([p.get_text() for p in soup.find_all("p")])
            articles.append(text.strip())
        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch {url}: {e}")
            articles.append("")  # Keep index alignment
    return articles

# --- Sentiment Analysis ---
def analyze_sentiment(texts):
    """
    Run sentiment analysis on a list of texts.
    Truncates each text to 512 characters.
    """
    return [sentiment_en(t[:512])[0] if t.strip() else {} for t in texts]

# --- Named Entity Recognition ---
def extract_entities(texts):
    """
    Extract named entities from a list of texts.
    Truncates each text to 512 characters.
    """
    return [ner(t[:512]) if t.strip() else [] for t in texts]

# --- Summarization ---
def summarize_texts(texts):
    """
    Generate summaries for each input text.
    Truncates each text to 1024 characters.
    """
    return [
        summarizer(t[:1024])[0]["summary_text"] if t.strip() else "⚠️ No content to summarize."
        for t in texts
    ]

# --- Topic Modeling with LDA ---
def topic_modeling_sklearn(docs, num_topics=5, num_words=10):
    """
    Extract topics from text using sklearn's LDA.
    Returns a list of topic strings.
    """
    valid_docs = [doc for doc in docs if doc.strip()]
    if len(valid_docs) < 2:
        return ["⚠️ Not enough valid documents for topic modeling."]

    cleaned_docs = [
        " ".join([word for word in doc.lower().split() if word.isalpha() and word not in stop_words])
        for doc in valid_docs
    ]

    min_df = 1 if len(valid_docs) < 5 else 2

    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=min_df, stop_words="english")
        doc_term_matrix = vectorizer.fit_transform(cleaned_docs)
        words = vectorizer.get_feature_names_out()
    except Exception as e:
        return [f"⚠️ Vectorizer error: {e}"]

    try:
        n_topics = min(num_topics, doc_term_matrix.shape[0])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
    except Exception as e:
        return [f"⚠️ LDA model error: {e}"]

    topics = []
    for idx, topic_weights in enumerate(lda.components_):
        top_terms = [words[i] for i in topic_weights.argsort()[:-num_words - 1:-1]]
        topics.append(f"🔹 Topic {idx + 1}: " + ", ".join(top_terms))
    return topics

# --- Unified NLP Execution ---
def run_nlp_pipeline(urls=None, texts=None):
    """
    Unified NLP runner for URLs or preloaded texts.
    Returns dict of results: texts, sentiments, summaries, NER, and topics.
    """
    if urls:
        texts = fetch_articles(urls)
    elif not texts:
        raise ValueError("❗ You must provide either URLs or text content.")

    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        raise ValueError("⚠️ No valid textual content found.")

    logging.info("🚀 Running NLP pipeline...")

    return {
        "texts": texts,
        "sentiments": analyze_sentiment(valid_texts),
        "entities": extract_entities(valid_texts),
        "summaries": summarize_texts(valid_texts),
        "topics": topic_modeling_sklearn(valid_texts)
    }
