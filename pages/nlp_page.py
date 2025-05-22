# pages/nlp_page.py

import streamlit as st
from nlp_utils import run_nlp_pipeline

def render():
    st.title("🗞️ NLP Dashboard: Climate News Analysis")
    st.markdown(
        "Analyze climate-related news using **sentiment analysis**, "
        "**topic modeling**, **named entity recognition**, and **summarization**."
    )

    # --- Input URLs ---
    st.subheader("📥 Enter Article URLs")

    urls = [ 
        "https://risingnepaldaily.com/news/59885",
        "https://risingnepaldaily.com/news/52538",
        "https://risingnepaldaily.com/news/51726",
        "https://risingnepaldaily.com/news/52739",
        "https://risingnepaldaily.com/news/55425",
        "https://risingnepaldaily.com/news/41304"
    ]

    default_urls = "\n".join(urls)
    urls_input = st.text_area("Paste one URL per line:", default_urls, height=150)
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

    # --- Run Analysis ---
    if st.button("🔍 Analyze Articles"):
        if not urls:
            st.warning("⚠️ Please enter at least one valid URL.")
            return

        try:
            with st.spinner("Running NLP pipeline..."):
                results = run_nlp_pipeline(urls)
        except Exception as e:
            st.error(f"❌ Error during NLP processing: {e}")
            return

        st.success("✅ NLP Analysis Completed!")

        # --- Summarization ---
        st.subheader("📝 Article Summaries")
        for i, summary in enumerate(results["summaries"]):
            st.markdown(f"**Article {i+1}**")
            st.info(summary)

        # --- Topic Modeling ---
        st.subheader("📚 Extracted Topics")
        for topic in results["topics"]:
            st.markdown(f"- {topic}")

        # --- Sentiment Analysis ---
        st.subheader("💬 Sentiment Overview")
        for i, sent in enumerate(results["sentiments"]):
            label = sent["label"]
            score = sent["score"]
            st.markdown(f"**Article {i+1}:** `{label}` (confidence = {score:.2f})")

        # --- Named Entities ---
        st.subheader("🔎 Named Entity Recognition")
        for i, ents in enumerate(results["entities"]):
            st.markdown(f"**Article {i+1}:**")
            if not ents:
                st.write("No named entities detected.")
            else:
                for ent in ents:
                    st.markdown(f"- `{ent['entity_group']}` → **{ent['word']}**")
