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
    st.markdown("Explore historical climate patterns, regional temperature trends, extreme events, and glacier change metrics.")

    # --- Load Data ---
    @st.cache_data
    def load_data():
        climate = pd.read_csv(Path("data/processed/climate_yearly.csv"))
        glacier = pd.read_csv(Path("data/processed/glacier_features.csv"))
        daily = pd.read_csv(
            Path("data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"),
            compression="gzip",
            dayfirst=True
        )
        daily["DATE"] = pd.to_datetime(daily["DATE"], errors="coerce")
        return climate, glacier, daily

    climate_df, glacier_df, climate_daily = load_data()

    # --- National Trends ---
    st.subheader("🌡️ National Climate Trends (1981–2019)")
    st.pyplot(plot_national_trends(climate_df))

    # --- Heatwave Trends ---
    st.subheader("🔥 Heatwave Days per Year")
    heatwave_df = climate_df.groupby("YEAR", as_index=False)["heatwave_days"].sum()
    st.pyplot(plot_heatwave_trend(heatwave_df))

    # --- District-Level Temperature Trends ---
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
    st.subheader("🌀 Temperature by Season")

    def assign_season(month):
        if pd.isna(month):
            return None
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        elif month in [12, 1, 2]:
            return 'Winter'
        return 'Unknown'

    climate_daily['MONTH'] = climate_daily['DATE'].dt.month
    climate_daily['Season'] = climate_daily['MONTH'].apply(assign_season)
    season_df = climate_daily.dropna(subset=['T2M', 'Season'])
    st.pyplot(plot_temp_by_season(season_df))

    # --- Glacier Retreat Overview ---
    st.subheader("🧊 Glacier Area Loss by Sub-Basin (1980–2010)")
    st.pyplot(plot_glacier_area_loss(glacier_df))

    # --- Show Raw Data (Optional) ---
    with st.expander("📂 View Processed Climate Summary"):
        st.dataframe(climate_df.head())

    with st.expander("📂 View Glacier Retreat Summary"):
        st.dataframe(glacier_df.head())

    with st.expander("📂 View Raw Daily Climate Sample"):
        st.dataframe(climate_daily[['DATE', 'DISTRICT', 'T2M', 'PRECTOT']].dropna().head())
