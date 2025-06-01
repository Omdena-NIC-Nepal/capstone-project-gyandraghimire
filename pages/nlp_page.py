# pages/nlp_page.py

import streamlit as st
import pandas as pd
import docx
from nlp_utils import run_nlp_pipeline

def render():
    st.title("🗞️ NLP Dashboard: Climate News Analysis")
    st.markdown("""
        Analyze climate-related documents or news using **advanced NLP tools**:  
        - 💬 Sentiment Analysis  
        - 🧠 Topic Modeling  
        - 🔎 Named Entity Recognition  
        - 📝 Summarization  
    """)

    # --- Input Type Selection ---
    st.subheader("🔁 Select Input Method")
    input_mode = st.radio("Choose input format:", ["🌐 URLs", "🖋️ Raw Text", "📁 Upload File"])

    texts, urls = [], []

    # --- Handle Each Input Mode ---
    if input_mode == "🌐 URLs":
        st.info("Paste one article URL per line:")
        default_urls = "\n".join([
            "https://risingnepaldaily.com/news/59885",
            "https://risingnepaldaily.com/news/52538",
            "https://risingnepaldaily.com/news/51726",
            "https://risingnepaldaily.com/news/52739",
            "https://risingnepaldaily.com/news/55425",
            "https://risingnepaldaily.com/news/41304"
        ])
        url_input = st.text_area("🔗 Article URLs", default_urls, height=150)
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]

    elif input_mode == "🖋️ Raw Text":
        st.info("Paste or type any custom text you want to analyze:")
        raw_text = st.text_area("🖊️ Text Input", height=300)
        if raw_text.strip():
            texts = [raw_text.strip()]

    elif input_mode == "📁 Upload File":
        st.info("Upload a `.txt`, `.csv`, or `.docx` file for NLP analysis:")
        file = st.file_uploader("📂 Choose File", type=["txt", "csv", "docx"])
        if file:
            try:
                if file.name.endswith(".txt"):
                    texts = [file.read().decode("utf-8")]
                elif file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                    texts = df.iloc[:, 0].dropna().astype(str).tolist()
                elif file.name.endswith(".docx"):
                    doc = docx.Document(file)
                    texts = ['\n'.join(p.text for p in doc.paragraphs)]
                else:
                    st.error("Unsupported file type.")
                st.success(f"✅ Loaded content from `{file.name}`")
            except Exception as e:
                st.error(f"❌ Could not read file: {e}")

    # --- Run NLP Analysis ---
    if st.button("🔍 Run NLP Analysis"):
        if not (texts or urls):
            st.warning("⚠️ Please provide some input to analyze.")
            return

        try:
            with st.spinner("🔄 Running NLP pipeline..."):
                results = run_nlp_pipeline(urls=urls if urls else None, texts=texts if texts else None)
            st.success("✅ NLP Analysis Complete")
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            return

        # --- Summaries ---
        st.subheader("📝 Text Summaries")
        for i, summary in enumerate(results.get("summaries", [])):
            with st.expander(f"📄 Summary {i + 1}", expanded=False):
                st.markdown(summary)

        # --- Topics ---
        st.subheader("📚 Extracted Topics")
        for topic in results.get("topics", []):
            st.markdown(f"- {topic}")

        # --- Sentiment ---
        st.subheader("💬 Sentiment Analysis")
        for i, sentiment in enumerate(results.get("sentiments", [])):
            label = sentiment.get("label", "Unknown").upper()
            score = sentiment.get("score", 0.0)
            badge = "🟢 Positive" if label == "POSITIVE" else "🔴 Negative" if label == "NEGATIVE" else "⚪ Neutral"
            st.markdown(f"**Document {i + 1}:** {badge} (Confidence: {score:.2f})")

        # --- Entities ---
        st.subheader("🔎 Named Entity Recognition")
        for i, entities in enumerate(results.get("entities", [])):
            with st.expander(f"🧾 Entities in Document {i + 1}", expanded=False):
                if not entities:
                    st.info("No named entities detected.")
                else:
                    for ent in entities:
                        st.markdown(f"- `{ent['entity_group']}` → **{ent['word']}**")
