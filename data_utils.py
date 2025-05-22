# data_utils.py ⚙️ Data Loading & Preprocessing Utilities

import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Base Data Paths ---
RAW_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = RAW_DATA_DIR / "processed"
SHAPEFILE_DIR = RAW_DATA_DIR / "local_unit_shapefiles"

# --- Safe CSV Loader ---
def load_csv(filename, folder=RAW_DATA_DIR, **kwargs):
    path = folder / filename
    try:
        df = pd.read_csv(path, **kwargs)
        print(f"✅ Loaded: {path.name} — shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error loading {path.name}: {e}")
        return pd.DataFrame()

# --- Safe Shapefile Loader ---
def load_shapefile(filename="local_unit.shp"):
    path = SHAPEFILE_DIR / filename
    try:
        gdf = gpd.read_file(path)
        print(f"✅ Loaded shapefile: {path.name} — records: {len(gdf)}")
        return gdf
    except Exception as e:
        print(f"❌ Error loading shapefile: {e}")
        return gpd.GeoDataFrame()

# --- Load Daily Climate Data (Compressed .gz) ---
def load_daily_climate():
    return load_csv(
        "climate_data_nepal_district_wise_daily_1981_2019.csv.gz",
        compression="gzip",
        parse_dates=["DATE"],
        dayfirst=True
    )

# --- Load Processed Climate Summary ---
def load_climate_summary():
    return load_csv("climate_yearly.csv", folder=PROCESSED_DATA_DIR)

# --- Load Glacier Metrics ---
def load_glacier_features():
    return load_csv("glacier_features.csv", folder=PROCESSED_DATA_DIR)

# --- Load Land Use Long Format ---
def load_land_use_long():
    return load_csv("land_use_long.csv", folder=PROCESSED_DATA_DIR)

# --- Load Cereal Yield Long Format ---
def load_yield_features():
    return load_csv("yield_features.csv", folder=PROCESSED_DATA_DIR)

# --- Load District Centroids ---
def load_district_centroids():
    return load_csv("district_centroids.csv", folder=PROCESSED_DATA_DIR)
