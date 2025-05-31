# pages/prediction_page.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prediction import (
    get_climate_forecast,
    get_heatwave_forecast,
    get_drought_forecast,
    get_glacier_forecast
)

def render():
    st.title("🔮 Climate Forecasting Dashboard")
    st.markdown("Visualize and interact with forecasted climate indicators for Nepal through 2050.")

    # --- Forecast Category Selection ---
    option = st.selectbox(
        "Choose forecast type:",
        ["📈 Average Temperature", "🔥 Heatwave Days", "🌧️ Drought Risk", "🧊 Glacier Loss"]
    )

    # --- Forecast Viewer Logic ---
    if option == "📈 Average Temperature":
        df = get_climate_forecast()
        district = st.selectbox("Select District", sorted(df["DISTRICT"].unique()))
        df_d = df[df["DISTRICT"] == district]

        fig = px.line(
            df_d,
            x="YEAR", y="predicted_avg_temp",
            title=f"Forecasted Avg Temperature in {district} (2020–2050)",
            labels={"predicted_avg_temp": "Temperature (°C)", "YEAR": "Year"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_d)

    elif option == "🔥 Heatwave Days":
        df = get_heatwave_forecast()
        district = st.selectbox("Select District", sorted(df["DISTRICT"].unique()))
        df_d = df[df["DISTRICT"] == district]

        fig = px.line(
            df_d,
            x="YEAR", y="predicted_heatwave_days",
            title=f"Forecasted Heatwave Days in {district} (2020–2050)",
            labels={"predicted_heatwave_days": "Days >35°C", "YEAR": "Year"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_d)

    elif option == "🌧️ Drought Risk":
        df = get_drought_forecast()
        district = st.selectbox("Select District", sorted(df["DISTRICT"].unique()))
        df_d = df[df["DISTRICT"] == district]

        fig = px.line(
            df_d,
            x="YEAR", y="predicted_spi",
            title=f"SPI-based Drought Forecast in {district} (2020–2050)",
            labels={"predicted_spi": "SPI (z-score)", "YEAR": "Year"},
            color_discrete_sequence=["teal"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecasted Drought Categories")
        st.dataframe(df_d[["YEAR", "predicted_spi", "drought_risk"]])

    elif option == "🧊 Glacier Loss":
        df = get_glacier_forecast()
        subbasin = st.selectbox("Select Sub-Basin", sorted(df["sub-basin"].unique()))
        df_s = df[df["sub-basin"] == subbasin]

        fig = px.line(
            df_s,
            x="year", y="predicted_glacier_area",
            title=f"Forecasted Glacier Area in {subbasin} (2020–2050)",
            labels={"predicted_glacier_area": "Glacier Area (km²)", "year": "Year"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Ice Volume & Elevation Forecast")
        st.dataframe(df_s[["year", "predicted_ice_volume", "predicted_min_elev"]].round(2))
