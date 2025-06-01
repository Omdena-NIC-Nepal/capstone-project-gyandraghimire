# app.py 🌐 Main Streamlit App

import streamlit as st
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Nepal Climate Impact System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Page Styling ---
st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #555;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: 30px;
        }
        footer {visibility: hidden;}  /* Hide Streamlit branding */
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("📊 Climate Dashboard")
page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Overview",
        "📈 Data Exploration",
        "🧠 Model Training",
        "🔮 Predictions & Forecasts",
        "🗞️ NLP Dashboard"
    ]
)

# --- Overview Page Content ---
def render_overview():
    st.markdown('<div class="main-title">Climate Change Impact Assessment and Prediction System for Nepal</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore climate trends, model impacts, and forecast risks using real data and machine learning.</div>', unsafe_allow_html=True)

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/d/dd/Sagarmatha_Everest_Zone%2C_Nepal%2C_Himalayas.jpg",
        width=700,
        caption="📷 View of Khumbu Glacier and Himalayan peaks from Mount Pumori base. © Vyacheslav Argenberg / [vascoplanet.com](http://www.vascoplanet.com/)"
    )

    st.markdown("### 📌 System Features")
    st.markdown("""
- Visualize national and district-level climate data (1981–2019)  
- Monitor glacial retreat, monsoon variability, and dry spells  
- Train and evaluate machine learning models for:  
  - 🌡️ Heatwave prediction  
  - 🌾 Yield forecasting  
  - ❄️ Glacier loss analysis  
  - 🌧️ Drought risk classification  
- Forecast up to 2050 using real-world trends  
- Analyze climate-related news via NLP (sentiment, topics, summaries)  
    """)

    st.markdown('<div class="footer">Developed by Gyandra Ghimire | Capstone Project | Omdena-NIC Data Science Course | 📅 2025</div>', unsafe_allow_html=True)

# --- Page Routing with Error Handling ---
with st.spinner("🔄 Loading page..."):
    try:
        if page == "🏠 Overview":
            render_overview()
        elif page == "📈 Data Exploration":
            from pages import data_exploration
            data_exploration.render()
        elif page == "🧠 Model Training":
            from pages import model_training
            model_training.render()
        elif page == "🔮 Predictions & Forecasts":
            from pages import prediction_page
            prediction_page.render()
        elif page == "🗞️ NLP Dashboard":
            from pages import nlp_page
            nlp_page.render()
    except ModuleNotFoundError as mod_err:
        st.error(f"🚫 Page module not found: `{mod_err.name}`")
    except AttributeError as attr_err:
        st.error(f"⚠️ Page failed to render. Check `render()` method: {attr_err}")
    except Exception as e:
        st.exception(f"❌ Unexpected error occurred: {e}")
