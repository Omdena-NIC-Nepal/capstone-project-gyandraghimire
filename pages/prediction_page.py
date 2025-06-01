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
    st.markdown("Explore Nepal's projected climate indicators from **2020 to 2050**, including temperature, heatwaves, drought risk, and glacier loss.")

    # --- Forecast Category Selection ---
    option = st.selectbox(
        "🔍 Choose Forecast Type",
        ["📈 Average Temperature", "🔥 Heatwave Days", "🌧️ Drought Risk", "🧊 Glacier Loss"]
    )

    # --- Forecast Handlers ---
    try:
        if option == "📈 Average Temperature":
            df = get_climate_forecast()
            handle_district_forecast(df, "DISTRICT", "predicted_avg_temp", "Average Temperature (°C)")

        elif option == "🔥 Heatwave Days":
            df = get_heatwave_forecast()
            handle_district_forecast(df, "DISTRICT", "predicted_heatwave_days", "Heatwave Days (>35°C)")

        elif option == "🌧️ Drought Risk":
            df = get_drought_forecast()
            district = st.selectbox("📍 Select District", sorted(df["DISTRICT"].unique()))
            df_d = df[df["DISTRICT"] == district]

            fig = px.line(
                df_d, x="YEAR", y="predicted_spi",
                title=f"🌧️ SPI-based Drought Forecast for {district} (2020–2050)",
                labels={"predicted_spi": "SPI (z-score)", "YEAR": "Year"},
                color_discrete_sequence=["teal"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Drought Risk Classification")
            st.dataframe(df_d[["YEAR", "predicted_spi", "drought_risk"]])

        elif option == "🧊 Glacier Loss":
            df = get_glacier_forecast()
            subbasin_col = "sub-basin" if "sub-basin" in df.columns else df.columns[-1]
            subbasin = st.selectbox("🗻 Select Sub-Basin", sorted(df[subbasin_col].unique()))
            df_s = df[df[subbasin_col] == subbasin]

            fig = px.line(
                df_s, x="year", y="predicted_glacier_area",
                title=f"🧊 Forecasted Glacier Area in {subbasin} (2020–2050)",
                labels={"predicted_glacier_area": "Glacier Area (km²)", "year": "Year"}
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🧊 Ice Volume & Elevation Forecast")
            st.dataframe(df_s[["year", "predicted_ice_volume", "predicted_min_elev"]].round(2))

    except Exception as e:
        st.error(f"❌ Failed to load forecast: {e}")

# --- Helper Function for District Forecasts ---
def handle_district_forecast(df, location_col, value_col, value_label):
    if location_col not in df.columns:
        st.warning(f"⚠️ Missing expected column: `{location_col}`")
        return

    location = st.selectbox(f"📍 Select {location_col.title()}", sorted(df[location_col].unique()))
    df_filtered = df[df[location_col] == location]

    fig = px.line(
        df_filtered, x="YEAR", y=value_col,
        title=f"📈 Forecasted {value_label} in {location} (2020–2050)",
        labels={value_col: value_label, "YEAR": "Year"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"#### Forecasted Values for {location}")
    st.dataframe(df_filtered)
