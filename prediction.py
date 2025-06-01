# prediction.py 🔮 Forecasting and Inference Utilities

import joblib
import pandas as pd
from pathlib import Path
import streamlit as st

# --- Load Saved Model ---
@st.cache_resource(show_spinner="📦 Loading model...")
def load_model(model_name: str, folder: str = "data/processed"):
    """
    Load a trained machine learning model from disk.
    
    Args:
        model_name (str): Filename of the model (e.g. 'model.joblib').
        folder (str): Directory containing the model.

    Returns:
        model: Loaded model object.
    """
    path = Path(folder) / model_name
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"❌ Model not found at: {path}")
        raise
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        raise

# --- Single Prediction ---
def predict_single(model, input_dict: dict):
    """
    Predict a single data instance.

    Args:
        model: Trained model.
        input_dict (dict): Input features as a dictionary.

    Returns:
        Prediction value.
    """
    try:
        df = pd.DataFrame([input_dict])
        return model.predict(df)[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

# --- Batch Prediction ---
def predict_batch(model, df: pd.DataFrame):
    """
    Predict a batch of data rows.

    Args:
        model: Trained model.
        df (pd.DataFrame): DataFrame of features.

    Returns:
        np.ndarray: Array of predictions.
    """
    try:
        return model.predict(df)
    except Exception as e:
        raise RuntimeError(f"Batch prediction failed: {e}")

# --- Load Forecast CSV ---
@st.cache_data(show_spinner="📊 Loading forecast data...")
def load_forecast_csv(name: str, folder: str = "data/processed") -> pd.DataFrame:
    """
    Load a precomputed forecast CSV file.

    Args:
        name (str): Filename of the CSV.
        folder (str): Directory path.

    Returns:
        pd.DataFrame
    """
    path = Path(folder) / name
    if not path.exists():
        st.warning(f"⚠️ Forecast file not found: {path}")
        raise FileNotFoundError(path)
    return pd.read_csv(path)

# --- Specific Forecast Loaders ---
def get_climate_forecast():
    return load_forecast_csv("climate_forecast_2020_2050.csv")

def get_heatwave_forecast():
    return load_forecast_csv("heatwave_days_forecast_2020_2050.csv")

def get_drought_forecast():
    return load_forecast_csv("drought_forecast_spi_2020_2050.csv")

def get_glacier_forecast():
    return load_forecast_csv("glacier_forecast_2020_2050.csv")
