# data_utils.py ⚙️ Data Loading & Preprocessing Utilities

import pandas as pd
import geopandas as gpd
from pathlib import Path
import streamlit as st

# --- Base Data Paths ---
RAW_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = RAW_DATA_DIR / "processed"
SHAPEFILE_DIR = RAW_DATA_DIR / "local_unit_shapefiles"

# --- Cached CSV Loader ---
@st.cache_data(show_spinner="📥 Loading CSV...")
def load_csv(filename, folder=RAW_DATA_DIR, **kwargs):
    """
    Load a CSV file with caching.
    Args:
        filename (str): Name of the file.
        folder (Path): Directory where file is stored.
        kwargs: Optional arguments for pandas.read_csv.

    Returns:
        pd.DataFrame
    """
    path = folder / filename
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except FileNotFoundError:
        st.warning(f"❌ File not found: `{path}`")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Error loading `{path.name}`: {e}")
        return pd.DataFrame()

# --- Cached Shapefile Loader ---
@st.cache_data(show_spinner="📍 Loading shapefile...")
def load_shapefile(filename="local_unit.shp"):
    path = SHAPEFILE_DIR / filename
    try:
        gdf = gpd.read_file(path)
        return gdf
    except Exception as e:
        st.warning(f"❌ Error loading shapefile: {e}")
        return gpd.GeoDataFrame()

# --- Data Loaders ---
def load_daily_climate():
    """Load compressed daily climate data (1981–2019)."""
    return load_csv(
        "climate_data_nepal_district_wise_daily_1981_2019.csv.gz",
        compression="gzip",
        parse_dates=["DATE"],
        dayfirst=True
    )

def load_climate_summary():
    """Load processed yearly climate summary."""
    return load_csv("climate_yearly.csv", folder=PROCESSED_DATA_DIR)

def load_glacier_features():
    """Load glacier features dataset."""
    return load_csv("glacier_features.csv", folder=PROCESSED_DATA_DIR)

def load_land_use_long():
    """Load land use statistics in long format."""
    return load_csv("land_use_long.csv", folder=PROCESSED_DATA_DIR)

def load_yield_features():
    """Load cereal yield features."""
    return load_csv("yield_features.csv", folder=PROCESSED_DATA_DIR)

def load_district_centroids():
    """Load geographic centroids for districts."""
    return load_csv("district_centroids.csv", folder=PROCESSED_DATA_DIR)
