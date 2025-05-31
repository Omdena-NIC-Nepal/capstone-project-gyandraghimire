# visualizations.py 📊 Custom plotting functions

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd

sns.set(style="whitegrid")

# --- Temperature & Precipitation Trend Plot ---
def plot_national_trends(df):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    sns.lineplot(data=df, x="YEAR", y="avg_temp", ax=ax1, color="tab:red", label="Avg Temp (°C)", marker='o')
    ax1.set_ylabel("Avg Temperature (°C)", color="tab:red")
    ax1.tick_params(axis='y', labelcolor="tab:red")

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="YEAR", y="annual_precip", ax=ax2, color="tab:blue", label="Annual Precip (mm)", marker='o')
    ax2.set_ylabel("Annual Precipitation (mm)", color="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    ax1.set_xlabel("Year")
    ax1.set_title("National Climate Trends (1981–2019)")
    fig.tight_layout()
    return fig

# --- Annual Heatwave Days ---
def plot_heatwave_trend(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df, x="YEAR", y="heatwave_days", color="salmon", ax=ax)
    ax.set_title("Annual Heatwave Days (T2M_MAX > 35°C)")
    ax.set_ylabel("Number of Days")
    ax.set_xlabel("Year")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig

# --- Glacier Area Loss by Sub-Basin ---
def plot_glacier_area_loss(df):
    df = df.sort_values("area_loss_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="area_loss_pct",
        y="sub-basin",
        hue="retreat_severity",
        dodge=False,
        ax=ax
    )
    ax.set_title("Glacier Area Loss by Sub-Basin (1980–2010)")
    ax.set_xlabel("Area Loss (%)")
    ax.set_ylabel("Sub-Basin")
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    return fig

# --- Choropleth Map with Geopandas ---
def plot_choropleth(gdf, column, title, cmap="YlOrRd"):
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(
        column=column,
        cmap=cmap,
        legend=True,
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    fig.tight_layout()
    return fig

# --- District-Wise Temperature Lineplot ---
def plot_temp_by_district(temp_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=temp_df, x="YEAR", y="T2M", hue="DISTRICT", legend=False, ax=ax, linewidth=1)
    ax.set_title("District-Wise Annual Avg Temperature (1981–2019)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Year")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig

# --- Violin Plot of Temperature by Season ---
def plot_temp_by_season(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x='Season', y='T2M', palette='Set2', ax=ax)
    ax.set_title("Temperature Distribution by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig
