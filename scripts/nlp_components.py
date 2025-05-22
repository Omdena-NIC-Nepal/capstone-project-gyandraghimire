#!/usr/bin/env python
# coding: utf-8

# ## ✅ 7. Natural Language Processing Components

# In[ ]:


import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources
nltk.download('stopwords')

# 1. Fetch Articles
def fetch_articles(urls):
    articles = []
    for url in urls:
        try:
            print(f"🔄 Fetching: {url}")
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.text for p in paragraphs])
            articles.append(text)
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")
    return articles

# 2. Sentiment Analysis
sentiment_en = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(texts):
    return [sentiment_en(t[:512])[0] for t in texts]

# 3. Named Entity Recognition
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

def extract_entities(texts):
    return [ner(t[:512]) for t in texts]

# 4. Topic Modeling (No punkt dependency)
def topic_modeling_sklearn(docs, num_topics=5, num_words=10):
    stop_words = set(stopwords.words('english'))
    processed_docs = [
        ' '.join([w for w in doc.lower().split() if w.isalpha() and w not in stop_words])
        for doc in docs
    ]

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(processed_docs)

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(f"Topic {idx + 1}: " + ", ".join(top_words))
    return topics

# 5. Summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_texts(texts):
    return [summarizer(t[:1024])[0]['summary_text'] for t in texts]

# 6. NLP Pipeline Coordinator
def full_nlp_pipeline(urls):
    if not urls:
        raise ValueError("⚠️ No URLs provided. Please uncomment and include URLs in the list.")
    
    print("🔄 Fetching articles...")
    texts = fetch_articles(urls)
    
    print("🧠 Analyzing sentiment...")
    sentiments = analyze_sentiment(texts)
    
    print("🔍 Extracting named entities...")
    entities = extract_entities(texts)
    
    print("📝 Summarizing content...")
    summaries = summarize_texts(texts)
    
    print("📚 Running topic modeling...")
    topics = topic_modeling_sklearn(texts)

    return {
        'texts': texts,
        'sentiments': sentiments,
        'entities': entities,
        'summaries': summaries,
        'topics': topics
    }

# 7. Example usage
if __name__ == "__main__":
    urls = [
        "https://risingnepaldaily.com/news/59885",
        "https://risingnepaldaily.com/news/52538",
        "https://risingnepaldaily.com/news/51726",
        "https://risingnepaldaily.com/news/52739",
        "https://risingnepaldaily.com/news/55425",
        "https://risingnepaldaily.com/news/41304"
    ]

    results = full_nlp_pipeline(urls)

    print("\n✅ Summaries:")
    for i, summary in enumerate(results['summaries']):
        print(f"--- Article {i+1} ---\n{summary}\n")

    print("✅ Topics:")
    for topic in results['topics']:
        print(f"- {topic}")

    print("\n✅ Sentiment Snapshot:")
    for i, sent in enumerate(results['sentiments']):
        print(f"Article {i+1}: {sent['label']} (score={sent['score']:.2f})")

