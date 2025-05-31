# nlp_utils.py 🔤 NLP Utilities for Climate News Analysis

import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are available
nltk.download('stopwords')

# --- Pipelines ---
sentiment_en = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- Article Scraper ---
def fetch_articles(urls):
    articles = []
    for url in urls:
        try:
            print(f"🔄 Fetching: {url}")
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.text for p in paragraphs])
            articles.append(text)
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")
    return articles

# --- Sentiment Analysis ---
def analyze_sentiment(texts):
    return [sentiment_en(t[:512])[0] for t in texts]

# --- Named Entity Recognition ---
def extract_entities(texts):
    return [ner(t[:512]) for t in texts]

# --- Summarization ---
def summarize_texts(texts):
    return [summarizer(t[:1024])[0]['summary_text'] for t in texts]

# --- Topic Modeling using LDA ---
def topic_modeling_sklearn(docs, num_topics=5, num_words=10):
    if len(docs) < 2:
        return ["⚠️ Not enough documents for topic modeling (need ≥2)."]

    stop_words = set(stopwords.words('english'))
    processed_docs = [
        ' '.join([w for w in doc.lower().split() if w.isalpha() and w not in stop_words])
        for doc in docs
    ]

    # Adjust min_df for small document sets
    min_df = 1 if len(docs) < 5 else 2

    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=min_df, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(processed_docs)
    except ValueError as e:
        return [f"⚠️ Topic model error: {str(e)}"]

    # Fit LDA model
    n_topics = min(num_topics, doc_term_matrix.shape[0])
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(f"Topic {idx + 1}: " + ", ".join(top_words))
    return topics

# --- Full NLP Pipeline ---
def run_nlp_pipeline(urls):
    if not urls:
        raise ValueError("⚠️ No URLs provided.")

    texts = fetch_articles(urls)
    if not any(texts):
        raise ValueError("⚠️ No valid article content fetched.")

    sentiments = analyze_sentiment(texts)
    entities = extract_entities(texts)
    summaries = summarize_texts(texts)
    topics = topic_modeling_sklearn(texts)

    return {
        'texts': texts,
        'sentiments': sentiments,
        'entities': entities,
        'summaries': summaries,
        'topics': topics
    }
