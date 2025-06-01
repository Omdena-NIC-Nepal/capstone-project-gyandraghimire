# pages/data_exploration.py

import streamlit as st
import pandas as pd
from pathlib import Path
from visualizations import (
    plot_national_trends,
    plot_heatwave_trend,
    plot_glacier_area_loss,
    plot_temp_by_district,
    plot_temp_by_season
)

def render():
    st.title("📈 Climate Data Exploration")
    st.markdown(
        "Dive into Nepal's historical climate records from **1981 to 2019**, exploring national and local trends in temperature, precipitation, heatwaves, and glacier dynamics."
    )

    # --- Load Data ---
    @st.cache_data(show_spinner="📦 Loading datasets...")
    def load_data():
        try:
            climate = pd.read_csv(Path("data/processed/climate_yearly.csv"))
            glacier = pd.read_csv(Path("data/processed/glacier_features.csv"))
            daily = pd.read_csv(
                Path("data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"),
                compression="gzip",
                dayfirst=True
            )
            daily["DATE"] = pd.to_datetime(daily["DATE"], errors="coerce")
            return climate, glacier, daily
        except Exception as e:
            st.error(f"❌ Failed to load datasets: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    climate_df, glacier_df, climate_daily = load_data()

    # --- National Trends ---
    if not climate_df.empty:
        st.subheader("🌡️ National Climate Trends (1981–2019)")
        st.pyplot(plot_national_trends(climate_df))

        st.subheader("🔥 Heatwave Trends Across Nepal")
        heatwave_df = climate_df.groupby("YEAR", as_index=False)["heatwave_days"].sum()
        st.pyplot(plot_heatwave_trend(heatwave_df))
    else:
        st.warning("Climate summary data not available.")

    # --- District-Wise Temperature Trends ---
    if not climate_daily.empty:
        st.subheader("🏞️ District-Wise Temperature Trends")
        climate_daily['YEAR'] = climate_daily['DATE'].dt.year
        temp_by_district = (
            climate_daily.groupby(['YEAR', 'DISTRICT'])['T2M']
            .mean()
            .reset_index()
            .dropna()
        )
        st.pyplot(plot_temp_by_district(temp_by_district))

        # --- Seasonal Distribution ---
        st.subheader("🌀 Seasonal Temperature Patterns")

        def assign_season(month):
            if pd.isna(month):
                return None
            return (
                'Spring' if month in [3, 4, 5] else
                'Summer' if month in [6, 7, 8] else
                'Autumn' if month in [9, 10, 11] else
                'Winter' if month in [12, 1, 2] else
                'Unknown'
            )

        climate_daily['MONTH'] = climate_daily['DATE'].dt.month
        climate_daily['Season'] = climate_daily['MONTH'].apply(assign_season)
        season_df = climate_daily.dropna(subset=['T2M', 'Season'])
        st.pyplot(plot_temp_by_season(season_df))
    else:
        st.warning("Daily climate data not available.")

    # --- Glacier Retreat Overview ---
    if not glacier_df.empty:
        st.subheader("🧊 Glacier Area Loss by Sub-Basin (1980–2010)")
        st.pyplot(plot_glacier_area_loss(glacier_df))
    else:
        st.warning("Glacier dataset not available.")

    # --- Show Raw Data ---
    st.markdown("### 📋 Optional: View Sample Data Tables")
    with st.expander("🧾 Processed Climate Summary"):
        st.dataframe(climate_df.head())

    with st.expander("❄️ Glacier Retreat Summary"):
        st.dataframe(glacier_df.head())

    with st.expander("📅 Daily Climate Sample (T2M & Precipitation)"):
        st.dataframe(
            climate_daily[['DATE', 'DISTRICT', 'T2M', 'PRECTOT']]
            .dropna()
            .head()
        )
