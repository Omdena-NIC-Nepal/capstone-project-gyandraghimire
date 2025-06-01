#!/usr/bin/env python
# coding: utf-8

# ## ✅ 4. Exploratory Data Analysis (EDA)

# ### ✅ Temperature & Precipitation Trends

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load climate data
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
climate_df = pd.read_csv(climate_path)

# Step 2: Parse DATE and extract YEAR
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['YEAR'] = climate_df['DATE'].dt.year

# Step 3: Compute national yearly averages
temp_precip = (
    climate_df.groupby('YEAR')[['T2M', 'PRECTOT']]
    .mean()
    .reset_index()
)

# Step 4: Plotting setup
sns.set(style='whitegrid')
fig, ax1 = plt.subplots(figsize=(12, 5))

# Plot average temperature on primary y-axis
sns.lineplot(
    data=temp_precip, x='YEAR', y='T2M',
    label='Avg Temperature (°C)', marker='o', ax=ax1, color='tab:red'
)
ax1.set_ylabel('Avg Temperature (°C)', fontsize=12, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Plot average precipitation on secondary y-axis
ax2 = ax1.twinx()
sns.lineplot(
    data=temp_precip, x='YEAR', y='PRECTOT',
    label='Avg Precipitation (mm)', marker='o', ax=ax2, color='tab:blue'
)
ax2.set_ylabel('Avg Precipitation (mm)', fontsize=12, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Title and axis formatting
ax1.set_xlabel('Year', fontsize=12)
ax1.set_title('National Climate Trends in Nepal (1981–2019)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.5)

# Merge legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, title='Climate Variable', loc='upper left')

# Finalize layout
plt.tight_layout()
plt.show()


# ### ✅ Temperature Trends Across Regions and Elevations

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the climate dataset
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
climate_df = pd.read_csv(climate_path)

# Step 2: Ensure DATE column is datetime and extract YEAR
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['YEAR'] = climate_df['DATE'].dt.year

# Step 3: Compute average annual temperature by district
temp_trend = (
    climate_df.groupby(['YEAR', 'DISTRICT'])['T2M']
    .mean()
    .reset_index()
)

# Step 4: Sort for consistent line plotting
temp_trend.sort_values(by=['DISTRICT', 'YEAR'], inplace=True)

# Step 5: Plot temperature trends
plt.figure(figsize=(14, 7))
sns.lineplot(
    data=temp_trend,
    x='YEAR',
    y='T2M',
    hue='DISTRICT',
    legend=False,  # Avoid cluttering legend if there are too many districts
    palette='tab20',
    linewidth=1
)

plt.title('Average Annual Temperature Trend by District (1981–2019)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### ✅ Precipitation Patterns Over Time

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the climate dataset
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
climate_df = pd.read_csv(climate_path)

# Step 2: Ensure 'DATE' is in datetime format
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')

# Step 3: Extract year
climate_df['YEAR'] = climate_df['DATE'].dt.year

# Step 4: Group by year and compute total precipitation
precip_trend = (
    climate_df.groupby('YEAR', as_index=False)['PRECTOT']
    .sum(min_count=1)  # Ensures NaNs don't default to zero
)

# Step 5: Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=precip_trend, x='YEAR', y='PRECTOT', marker='o', color='steelblue')

plt.title('Total Annual Precipitation in Nepal (1981–2019)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Precipitation (mm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### ✅ Extreme Weather Event Frequency

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the climate dataset
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
climate_df = pd.read_csv(climate_path)

# Step 2: Ensure 'DATE' is datetime
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')

# Step 3: Extract year
climate_df['YEAR'] = climate_df['DATE'].dt.year

# Step 4: Define extreme event thresholds
heatwave_days = (
    climate_df[climate_df['T2M_MAX'] > 35]
    .groupby('YEAR')
    .size()
    .rename('Heatwave Days')
)

heavy_rain_days = (
    climate_df[climate_df['PRECTOT'] > 50]
    .groupby('YEAR')
    .size()
    .rename('Heavy Rain Days')
)

# Step 5: Merge into single DataFrame with all years represented
all_years = pd.Series(range(climate_df['YEAR'].min(), climate_df['YEAR'].max() + 1), name='YEAR')
extreme_df = pd.DataFrame(all_years)
extreme_df = extreme_df.merge(heatwave_days, left_on='YEAR', right_index=True, how='left')
extreme_df = extreme_df.merge(heavy_rain_days, left_on='YEAR', right_index=True, how='left')
extreme_df.fillna(0, inplace=True)
extreme_df[['Heatwave Days', 'Heavy Rain Days']] = extreme_df[['Heatwave Days', 'Heavy Rain Days']].astype(int)

# Step 6: Plot the trends
extreme_df.set_index('YEAR').plot(
    kind='line',
    marker='o',
    figsize=(12, 6),
    title='Annual Frequency of Extreme Weather Events in Nepal (1981–2019)'
)
plt.xlabel('Year')
plt.ylabel('Number of Days')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ### ✅ Correlation Between Climate Variables

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the climate dataset
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
climate_df = pd.read_csv(climate_path)

# Step 2: Select numeric climate variables
variables = ['PRECTOT', 'T2M', 'RH2M', 'QV2M', 'WS10M']
climate_subset = climate_df[variables].dropna()  # Remove rows with any missing values

# Step 3: Compute correlation matrix
climate_corr = climate_subset.corr()

# Step 4: Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    climate_corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 10}
)

plt.title('Correlation Matrix of Climate Variables', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ### ✅ Violin Plot: Temperature by Season

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load climate data
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
climate_df = pd.read_csv(climate_path)

# Step 2: Parse date and extract month
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['MONTH'] = climate_df['DATE'].dt.month

# Step 3: Assign season if not already present
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

climate_df['Season'] = climate_df['MONTH'].apply(assign_season)

# Step 4: Filter clean rows for plotting
climate_df = climate_df.dropna(subset=['Season', 'T2M'])

# Step 5: Create violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=climate_df,
    x='Season',
    y='T2M',
    hue='Season',           # Color by season
    palette='Set2',
    dodge=False,            # Overlay not split
    legend=False            # Hue = x, no extra legend needed
)

plt.title('Distribution of Mean Temperature by Season', fontsize=14)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# ### ✅ Glacier Area, Volume, Elevation Over Time

# #### 1. Average Glacier Area Over Time

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load glacier dataset
glacier_path = "../data/glaciers_change_in_basins_subbasins_1980_1990_2000_2010.csv"
glacier_df = pd.read_csv(glacier_path)

# Step 2: Clean and normalize column names if needed
glacier_df.columns = (
    glacier_df.columns
    .str.strip()
    .str.lower()
    .str.replace('~', '', regex=False)
    .str.replace(' ', '_')
    .str.replace(r'\(km2\)', '', regex=True)
    .str.replace(r'[()]', '', regex=True)
)

# Step 3: Compute average glacier area for each year
avg_glacier_area = glacier_df[
    ['glacier_area_in_1980', 'glacier_area_1990', 'glacier_area_2000', 'glacier_area_2010']
].mean()

# Step 4: Rename index for clarity in plot
avg_glacier_area.index = ['1980', '1990', '2000', '2010']

# Step 5: Plot
plt.figure(figsize=(8, 5))
avg_glacier_area.plot(marker='o', linestyle='-', color='teal')

plt.title('Average Glacier Area Over Time (Nepal)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Glacier Area (km²)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# #### 2. Average Glacier Ice Volume Over Time

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load glacier dataset
glacier_path = "../data/glaciers_change_in_basins_subbasins_1980_1990_2000_2010.csv"
glacier_df = pd.read_csv(glacier_path)

# Step 2: Clean and normalize column names
glacier_df.columns = (
    glacier_df.columns
    .str.strip()
    .str.lower()
    .str.replace('~', '', regex=False)
    .str.replace(' ', '_')
    .str.replace(r'\(km3\)', '', regex=True)
    .str.replace(r'[()]', '', regex=True)
)

# Step 3: Compute average glacier ice volume
avg_glacier_volume = glacier_df[
    ['estimated_ice_reserved_1980', 'estimated_ice_reserved_1990',
     'estimated_ice_reserved2000', 'estimated_ice_reserved2010']
].mean()

# Step 4: Set year labels as index
avg_glacier_volume.index = ['1980', '1990', '2000', '2010']

# Step 5: Plot
plt.figure(figsize=(8, 5))
avg_glacier_volume.plot(marker='o', linestyle='-', color='teal')

plt.title('Average Glacier Ice Volume Over Time (Nepal)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Ice Volume (km³)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# #### 3. Average Minimum Glacier Elevation Over Time

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load glacier dataset
glacier_path = "../data/glaciers_change_in_basins_subbasins_1980_1990_2000_2010.csv"
glacier_df = pd.read_csv(glacier_path)

# Step 2: Clean and normalize column names
glacier_df.columns = (
    glacier_df.columns
    .str.strip()
    .str.lower()
    .str.replace('~', '', regex=False)
    .str.replace(' ', '_')
    .str.replace(r'\(masl\)', '', regex=True)
    .str.replace(r'[()]', '', regex=True)
)

# Step 3: Compute average minimum elevation per year
avg_min_elev = glacier_df[
    ['minimum_elevation_in_1980', 'minimum_elevation_in1990',
     'minimum_elevation_in2000', 'minimum_elevation_in2010']
].mean()

# Step 4: Set proper year labels
avg_min_elev.index = ['1980', '1990', '2000', '2010']

# Step 5: Plot
plt.figure(figsize=(8, 5))
avg_min_elev.plot(marker='o', linestyle='-', color='darkred')

plt.title('Average Minimum Glacier Elevation Over Time (Nepal)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Minimum Elevation (m a.s.l.)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ### ✅ Geospatial Map for Average Temperature by District

# In[ ]:


import pandas as pd
import geopandas as gpd
from fuzzywuzzy import process
import matplotlib.pyplot as plt

# Step 1: Load data
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
shapefile_path = "../data/local_unit_shapefiles/local_unit.shp"

climate_df = pd.read_csv(climate_path)
gdf = gpd.read_file(shapefile_path)

# Step 2: Standardize district names to uppercase
climate_df['DISTRICT'] = climate_df['DISTRICT'].str.strip().str.upper()
gdf['DISTRICT'] = gdf['DISTRICT'].str.strip().str.upper()

# Step 3: Compute average temperature per district
avg_temp = climate_df.groupby('DISTRICT', as_index=False)['T2M'].mean()

# Step 4: Fuzzy match district names
climate_districts = avg_temp['DISTRICT'].unique()

gdf['DISTRICT_CLEAN'] = gdf['DISTRICT'].apply(
    lambda x: process.extractOne(x, climate_districts)[0] if pd.notnull(x) else None
)

# Step 5: Dissolve geometries by matched names
gdf_dissolved = gdf.dissolve(by='DISTRICT_CLEAN', as_index=False)

# Step 6: Filter only matched districts
gdf_matched = gdf_dissolved[gdf_dissolved['DISTRICT_CLEAN'].isin(avg_temp['DISTRICT'])]

# Step 7: Merge temperature data
gdf_final = gdf_matched.merge(avg_temp, left_on='DISTRICT_CLEAN', right_on='DISTRICT', how='left')

# Step 8: Clean up invalid geometries
gdf_final = gdf_final.dropna(subset=['T2M'])
gdf_final = gdf_final[gdf_final.geometry.notnull() & gdf_final.is_valid & ~gdf_final.is_empty]

# Step 9: Plot choropleth
fig, ax = plt.subplots(figsize=(10, 8))
gdf_final.plot(
    column='T2M',
    cmap='OrRd',
    legend=True,
    ax=ax,
    edgecolor='black',
    linewidth=0.5
)

ax.set_title('Average Temperature by District (1981–2019)', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# ### Diagnostics for Spatial Join and Fuzzy Matching Logic

# In[ ]:


# Step 1: Count of matched districts with valid geometry
matched_district_count = gdf_final['DISTRICT_CLEAN'].nunique()
print(f"✅ Matched districts with geometry: {matched_district_count}")

# Step 2: Total unique districts in the climate dataset
total_climate_districts = avg_temp['DISTRICT'].nunique()
print(f"📊 Total unique districts in climate data: {total_climate_districts}")

# Step 3: Report unmatched district count
unmatched = total_climate_districts - matched_district_count
print(f"❗ Unmatched districts (likely due to name mismatch or missing geometry): {unmatched}")

# Step 4: List unmatched districts
matched_set = set(gdf_final['DISTRICT_CLEAN'].dropna())
all_set = set(avg_temp['DISTRICT'].dropna())
unmatched_districts = sorted(all_set - matched_set)

# Output
if unmatched_districts:
    print("🔍 Unmatched Districts:")
    for dist in unmatched_districts:
        print(f"  - {dist}")
else:
    print("✅ All districts matched successfully.")


# ### ✅ Choropleth map for average maximum temperature (T2M_MAX)

# In[ ]:


import pandas as pd
import geopandas as gpd
from fuzzywuzzy import process
import matplotlib.pyplot as plt

# === Step 1: Load Data ===
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
shapefile_path = "../data/local_unit_shapefiles/local_unit.shp"

climate_df = pd.read_csv(climate_path)
gdf = gpd.read_file(shapefile_path)

# === Step 2: Standardize District Names ===
climate_df['DISTRICT'] = climate_df['DISTRICT'].str.strip().str.upper()
gdf['DISTRICT'] = gdf['DISTRICT'].str.strip().str.upper()

# === Step 3: Compute District-wise Climate Averages ===
avg_temp = (
    climate_df.groupby('DISTRICT', as_index=False)[['T2M', 'T2M_MAX']]
    .mean()
)

# === Step 4: Fuzzy Match Shapefile Districts to Climate Districts ===
climate_districts = avg_temp['DISTRICT'].unique()
gdf['DISTRICT_CLEAN'] = gdf['DISTRICT'].apply(
    lambda x: process.extractOne(x, climate_districts)[0] if pd.notnull(x) else None
)

# === Step 5: Dissolve Local Units into Districts ===
gdf_dissolved = gdf.dissolve(by='DISTRICT_CLEAN', as_index=False)

# === Step 6: Filter to Matched Districts ===
gdf_matched = gdf_dissolved[gdf_dissolved['DISTRICT_CLEAN'].isin(avg_temp['DISTRICT'])]

# === Step 7: Merge with Climate Averages ===
gdf_final = gdf_matched.merge(
    avg_temp, left_on='DISTRICT_CLEAN', right_on='DISTRICT', how='left'
)

# === Step 8: Clean Geometries ===
gdf_final = gdf_final.dropna(subset=['T2M', 'T2M_MAX'])
gdf_final = gdf_final[gdf_final.geometry.notnull() & gdf_final.is_valid & ~gdf_final.is_empty]

# === Step 9: Choropleth Plot of T2M_MAX ===
fig, ax = plt.subplots(figsize=(10, 8))
gdf_final.plot(
    column='T2M_MAX',
    cmap='YlOrRd',
    legend=True,
    ax=ax,
    edgecolor='black',
    linewidth=0.5
)

ax.set_title('Average Maximum Temperature by District (1981–2019)', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()

# === Step 10: Matching Diagnostics ===
matched_district_count = gdf_final['DISTRICT_CLEAN'].nunique()
total_climate_districts = avg_temp['DISTRICT'].nunique()
unmatched = total_climate_districts - matched_district_count

print(f"\n✅ Matched districts: {matched_district_count}")
print(f"📊 Total districts in climate data: {total_climate_districts}")
print(f"❗ Unmatched districts: {unmatched}")

unmatched_districts = sorted(set(avg_temp['DISTRICT']) - set(gdf_final['DISTRICT_CLEAN']))
if unmatched_districts:
    print("🔍 Unmatched Districts:")
    for dist in unmatched_districts:
        print(f"  - {dist}")
else:
    print("✅ All districts matched successfully.")


# ### ✅ Geospatial Map for Average Precipitation by District

# In[ ]:


import pandas as pd
import geopandas as gpd
from fuzzywuzzy import process
import matplotlib.pyplot as plt

# === Step 1: Load Data ===
climate_path = "../data/climate_data_nepal_district_wise_daily_1981_2019.csv.gz"
shapefile_path = "../data/local_unit_shapefiles/local_unit.shp"

climate_df = pd.read_csv(climate_path)
gdf = gpd.read_file(shapefile_path)

# === Step 2: Standardize District Names ===
climate_df['DISTRICT'] = climate_df['DISTRICT'].str.strip().str.upper()
gdf['DISTRICT'] = gdf['DISTRICT'].str.strip().str.upper()

# === Step 3: Compute Average Annual Precipitation per District ===
avg_precip = climate_df.groupby('DISTRICT', as_index=False)['PRECTOT'].mean()

# === Step 4: Fuzzy Match Climate Districts to Shapefile ===
climate_districts = avg_precip['DISTRICT'].unique()
gdf['DISTRICT_CLEAN'] = gdf['DISTRICT'].apply(
    lambda x: process.extractOne(x, climate_districts)[0] if pd.notnull(x) else None
)

# === Step 5: Dissolve by Clean District Names ===
gdf_dissolved = gdf.dissolve(by='DISTRICT_CLEAN', as_index=False)

# === Step 6: Filter to Climate-Matched Districts ===
gdf_matched = gdf_dissolved[gdf_dissolved['DISTRICT_CLEAN'].isin(avg_precip['DISTRICT'])]

# === Step 7: Merge Average Precipitation Data ===
gdf_precip = gdf_matched.merge(avg_precip, left_on='DISTRICT_CLEAN', right_on='DISTRICT', how='left')

# === Step 8: Drop Invalid or Missing Geometries ===
gdf_precip = gdf_precip.dropna(subset=['PRECTOT'])
gdf_precip = gdf_precip[
    gdf_precip.geometry.notnull() & gdf_precip.is_valid & ~gdf_precip.is_empty
]

# === Step 9: Plot Choropleth ===
fig, ax = plt.subplots(figsize=(10, 8))
gdf_precip.plot(
    column='PRECTOT',
    cmap='Blues',
    legend=True,
    ax=ax,
    edgecolor='black',
    linewidth=0.5
)

ax.set_title('Average Annual Precipitation by District (1981–2019)', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# ### ✅ Interactive Choropleth Map for Average Temperature

# In[ ]:


import folium
from folium import Choropleth
import geopandas as gpd

# Step 1: Reproject to WGS84 (EPSG:4326) for web mapping
gdf_final = gdf_final.to_crs(epsg=4326)

# Step 2: Prepare columns for mapping
gdf_final['DISTRICT_CLEAN'] = gdf_final['DISTRICT_CLEAN'].astype(str)
gdf_final['T2M'] = gdf_final['T2M'].round(2)  # Round for display

# Step 3: Initialize Folium map centered on Nepal
m = folium.Map(location=[28.3, 84.0], zoom_start=7, tiles='CartoDB positron')

# Step 4: Add choropleth layer
Choropleth(
    geo_data=gdf_final,
    data=gdf_final,
    columns=['DISTRICT_CLEAN', 'T2M'],
    key_on='feature.properties.DISTRICT_CLEAN',
    fill_color='OrRd',
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name='Average Temperature (°C)',
    highlight=True,
    nan_fill_color='gray'
).add_to(m)

# Step 5: Add tooltips to each district
folium.GeoJson(
    gdf_final,
    name="District Labels",
    tooltip=folium.GeoJsonTooltip(
        fields=["DISTRICT_CLEAN", "T2M"],
        aliases=["District", "Temperature (°C)"],
        localize=True,
        sticky=True,
        labels=True,
        style="""
            background-color: white;
            color: #333;
            font-family: Arial;
            font-size: 12px;
            padding: 5px;
        """
    )
).add_to(m)

# Step 6: Add layer control
folium.LayerControl().add_to(m)

# Step 7: Save interactive map to file
m.save("average_temperature_map.html")

# Step 8: Display map in Jupyter notebook (if supported)
m


# ### ✅ Interactive Choropleth Map for Average Precipitation

# In[ ]:


import folium
from folium import Choropleth
import geopandas as gpd

# Step 1: Reproject GeoDataFrame to WGS84 (EPSG:4326) for web mapping
gdf_precip = gdf_precip.to_crs(epsg=4326)

# Step 2: Ensure clean column data for JSON export
gdf_precip['DISTRICT_CLEAN'] = gdf_precip['DISTRICT_CLEAN'].astype(str)
gdf_precip['PRECTOT'] = gdf_precip['PRECTOT'].round(2)

# Step 3: Initialize Folium map centered on Nepal
m = folium.Map(location=[28.3, 84.0], zoom_start=7, tiles='CartoDB positron')

# Step 4: Add choropleth layer
Choropleth(
    geo_data=gdf_precip,
    data=gdf_precip,
    columns=['DISTRICT_CLEAN', 'PRECTOT'],
    key_on='feature.properties.DISTRICT_CLEAN',
    fill_color='YlGnBu',
    fill_opacity=0.7,
    line_opacity=0.4,
    legend_name='Average Precipitation (mm)',
    highlight=True,
    nan_fill_color='gray'
).add_to(m)

# Step 5: Add GeoJSON with tooltips
folium.GeoJson(
    gdf_precip,
    name="District Info",
    tooltip=folium.GeoJsonTooltip(
        fields=["DISTRICT_CLEAN", "PRECTOT"],
        aliases=["District", "Precipitation (mm)"],
        localize=True,
        labels=True,
        sticky=True,
        style="""
            background-color: white;
            color: #333;
            font-family: Arial;
            font-size: 12px;
            padding: 5px;
        """
    )
).add_to(m)

# Step 6: Add layer control
folium.LayerControl().add_to(m)

# Step 7: Save map to file
m.save('avg_precipitation_map.html')

# Step 8: Display in Jupyter (if supported)
m


# ### ✅ 7. Plotly Line Chart for Temperature Trends by District

# In[ ]:


import plotly.express as px

# Step 1: Normalize column names (in case of prior inconsistencies)
temp_trend.columns = temp_trend.columns.str.strip().str.upper()

# Step 2: Rename columns for clarity in plot
temp_trend = temp_trend.rename(columns={
    'T2M': 'Temperature',
    'YEAR': 'Year',
    'DISTRICT': 'District'
})

# Step 3: Generate interactive line chart
fig = px.line(
    temp_trend,
    x='Year',
    y='Temperature',
    color='District',
    title='Average Temperature Trend by District (1981–2019)',
    labels={
        'Temperature': 'Temperature (°C)',
        'Year': 'Year',
        'District': 'District'
    }
)

# Step 4: Update layout for styling
fig.update_layout(
    template='plotly_white',
    legend_title_text='District',
    margin=dict(l=40, r=40, t=60, b=40),
    height=600
)

# Step 5: Display figure
fig.show()


# ### ✅ Statistical Tests

# #### ✅ Statistical Test for Temperature Trend

# In[ ]:


import pandas as pd
from scipy.stats import ttest_ind

# Step 1: Ensure DATE column is datetime and extract YEAR
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['YEAR'] = climate_df['DATE'].dt.year

# Step 2: Drop rows with missing temperature or year
climate_df_clean = climate_df.dropna(subset=['T2M', 'YEAR'])

# Step 3: Split into groups: before and after 2000
before_2000 = climate_df_clean[climate_df_clean['YEAR'] < 2000]['T2M']
after_2000 = climate_df_clean[climate_df_clean['YEAR'] >= 2000]['T2M']

# Step 4: Perform Welch's t-test (unequal variances)
t_stat, p_value = ttest_ind(before_2000, after_2000, equal_var=False)

# Step 5: Report results
print("🧪 Welch's T-Test: Mean Temperature Before vs After 2000")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.5f}")

if p_value < 0.05:
    print("✅ Statistically significant difference in mean temperature.")
else:
    print("❌ No statistically significant difference in mean temperature.")


# #### ✅ Statistical Test for Precipitation Trend

# In[ ]:


import pandas as pd
from scipy.stats import ttest_ind

# Step 1: Ensure DATE is datetime and extract YEAR
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['YEAR'] = climate_df['DATE'].dt.year

# Step 2: Clean data — drop rows with missing precipitation or year
climate_df_clean = climate_df.dropna(subset=['PRECTOT', 'YEAR'])

# Step 3: Split into groups: before and after 2000
precip_before_2000 = climate_df_clean[climate_df_clean['YEAR'] < 2000]['PRECTOT']
precip_after_2000 = climate_df_clean[climate_df_clean['YEAR'] >= 2000]['PRECTOT']

# Step 4: Perform Welch’s t-test (unequal variances)
t_stat, p_value = ttest_ind(precip_before_2000, precip_after_2000, equal_var=False)

# Step 5: Output results
print("\n🧪 Welch's T-Test: Mean Precipitation Before vs After 2000")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.5f}")

if p_value < 0.05:
    print("✅ Statistically significant difference in mean precipitation.")
else:
    print("❌ No statistically significant difference in mean precipitation.")


# #### ✅ Correlation Tests (Temperature vs Precipitation) 

# In[ ]:


from scipy.stats import pearsonr, spearmanr

# --- 1. Define variable pairs to test ---
var_pairs = [
    ('T2M', 'PRECTOT'),       # Temperature vs Precipitation
    ('T2M', 'RH2M'),          # Temperature vs Relative Humidity
    ('T2M', 'QV2M'),          # Temperature vs Specific Humidity
    ('T2M', 'WS10M'),         # Temperature vs Wind Speed
    ('T2M_MAX', 'PRECTOT'),   # Max Temperature vs Precipitation
    ('T2M_RANGE', 'PRECTOT')  # Temp Range vs Precipitation
]

# --- 2. Run correlation tests ---
print("\n🔗 Correlation Analysis:")
header = f"{'Variable Pair':<30} {'Pearson r':>10} {'p':>10} {'Status':>14} | {'Spearman ρ':>10} {'p':>10} {'Status':>14}"
print(header)
print("-" * len(header))

for x, y in var_pairs:
    # Drop NaNs and align both series
    pair_df = climate_df[[x, y]].dropna()

    # Skip if not enough data
    if pair_df.shape[0] < 3:
        print(f"{x} vs {y:<20} {'-':>10} {'-':>10} {'Insufficient':>14} | {'-':>10} {'-':>10} {'data':>14}")
        continue

    try:
        # Compute correlations
        r, p_r = pearsonr(pair_df[x], pair_df[y])
        rho, p_rho = spearmanr(pair_df[x], pair_df[y])

        # Interpret significance
        pearson_status = "✅ Correlated" if p_r < 0.05 else "❌ Not"
        spearman_status = "✅ Correlated" if p_rho < 0.05 else "❌ Not"

        print(f"{x} vs {y:<20} {r:10.2f} {p_r:10.4f} {pearson_status:>14} | {rho:10.2f} {p_rho:10.4f} {spearman_status:>14}")

    except Exception as e:
        print(f"{x} vs {y:<20} {'-':>10} {'-':>10} {'Error':>14} | {'-':>10} {'-':>10} {'Error':>14}")


# #### ✅ ANOVA (Temperature difference between Seasons) 

# In[ ]:


import pandas as pd
from scipy.stats import f_oneway

# Ensure DATE is datetime
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')

# Extract MONTH if missing
if 'MONTH' not in climate_df.columns:
    climate_df['MONTH'] = climate_df['DATE'].dt.month

# Define seasons
def get_season(month):
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

# Add or replace Season column
climate_df['Season'] = climate_df['MONTH'].apply(get_season)

# Drop rows with missing values in T2M or Season
climate_df_clean = climate_df.dropna(subset=['T2M', 'Season'])

# Group by season
season_groups = [
    group['T2M'].values
    for _, group in climate_df_clean.groupby('Season')
    if group['T2M'].notnull().sum() > 1
]

# Check if enough groups are available
if len(season_groups) < 2:
    print("⚠️ Not enough valid seasonal groups for ANOVA.")
else:
    f_stat, p_val = f_oneway(*season_groups)
    result = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"

    print("🧪 One-Way ANOVA: Temperature Differences Across Seasons")
    print(f"F-statistic: {f_stat:.2f}")
    print(f"p-value    : {p_val:.4f}")
    print(f"Result     : {result} difference in mean temperature across seasons.")


# #### ✅ Mann-Kendall Trend Test (Temperature, Precipitation) 

# In[ ]:


import pymannkendall as mk
import pandas as pd

# Step 1: Ensure DATE is datetime and YEAR is extracted
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['YEAR'] = climate_df['DATE'].dt.year

# --- Annual Mean Temperature ---
annual_temp = climate_df.groupby('YEAR')['T2M'].mean().dropna().sort_index()

if len(annual_temp) >= 10:
    temp_result = mk.original_test(annual_temp)
    print("📈 Mann-Kendall Test – Annual Mean Temperature")
    print(f"Trend     : {temp_result.trend}")
    print(f"p-value   : {temp_result.p:.5f}")
    print(f"Slope     : {temp_result.slope:.5f}")
    print(f"Decision  : {'✅ Significant trend detected' if temp_result.h else '❌ No significant trend'}")
else:
    print("⚠️ Not enough data points for temperature trend analysis.")

# --- Annual Mean Precipitation ---
annual_precip = climate_df.groupby('YEAR')['PRECTOT'].mean().dropna().sort_index()

if len(annual_precip) >= 10:
    precip_result = mk.original_test(annual_precip)
    print("\n🌧️ Mann-Kendall Test – Annual Mean Precipitation")
    print(f"Trend     : {precip_result.trend}")
    print(f"p-value   : {precip_result.p:.5f}")
    print(f"Slope     : {precip_result.slope:.5f}")
    print(f"Decision  : {'✅ Significant trend detected' if precip_result.h else '❌ No significant trend'}")
else:
    print("\n⚠️ Not enough data points for precipitation trend analysis.")


# ### ✅ Mann-Kendall Test for Annual Mean T2M_MAX

# In[ ]:


import pymannkendall as mk

# Step 1: Ensure DATE and YEAR
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['YEAR'] = climate_df['DATE'].dt.year

# Step 2: Group by YEAR and compute mean T2M_MAX
annual_max_temp = climate_df.groupby('YEAR')['T2M_MAX'].mean().dropna().sort_index()

# Step 3: Run Mann-Kendall test if enough data
if len(annual_max_temp) >= 10:
    t2mmax_result = mk.original_test(annual_max_temp)
    print("📈 Mann-Kendall Test – Annual Mean Maximum Temperature (T2M_MAX)")
    print(f"Trend     : {t2mmax_result.trend}")
    print(f"p-value   : {t2mmax_result.p:.5f}")
    print(f"Slope     : {t2mmax_result.slope:.5f}")
    print(f"Decision  : {'✅ Significant trend detected' if t2mmax_result.h else '❌ No significant trend'}")
else:
    print("⚠️ Not enough data points for T2M_MAX trend analysis.")


# ### ✅ Mann-Kendall Test for T2M_MIN

# In[ ]:


import pymannkendall as mk
import matplotlib.pyplot as plt

# Step 1: Compute annual mean minimum temperature
annual_min_temp = climate_df.groupby('YEAR')['T2M_MIN'].mean().dropna().sort_index()

# Step 2: Run Mann-Kendall test
if len(annual_min_temp) >= 10:
    t2mmin_result = mk.original_test(annual_min_temp)
    print("🌙 Mann-Kendall Test – Annual Mean Minimum Temperature (T2M_MIN)")
    print(f"Trend     : {t2mmin_result.trend}")
    print(f"p-value   : {t2mmin_result.p:.5f}")
    print(f"Slope     : {t2mmin_result.slope:.5f}")
    print(f"Decision  : {'✅ Significant trend detected' if t2mmin_result.h else '❌ No significant trend'}")
else:
    print("⚠️ Not enough data points for T2M_MIN trend analysis.")


# ### ✅ Trend in Daily Temperature Range (T2M_RANGE)

# In[ ]:


annual_range = climate_df.groupby('YEAR')['T2M_RANGE'].mean().dropna().sort_index()
range_result = mk.original_test(annual_range)

print("\n🌗 Mann-Kendall Test – Annual Temperature Range (T2M_MAX - T2M_MIN)")
print(f"Trend     : {range_result.trend}")
print(f"p-value   : {range_result.p:.5f}")
print(f"Slope     : {range_result.slope:.5f}")
print(f"Decision  : {'✅ Significant trend detected' if range_result.h else '❌ No significant trend'}")


# ### ✅ Trend in Daily Temperature Range (T2M_RANGE) by District

# In[ ]:


import pandas as pd
import pymannkendall as mk
import geopandas as gpd
import matplotlib.pyplot as plt

# === Step 1: Ensure DATE and YEAR are available ===
climate_df['DATE'] = pd.to_datetime(climate_df['DATE'], errors='coerce')
climate_df['YEAR'] = climate_df['DATE'].dt.year

# === Step 2: Compute annual average T2M_RANGE per district ===
annual_range_by_dist = (
    climate_df
    .dropna(subset=['YEAR', 'DISTRICT', 'T2M_RANGE'])
    .groupby(['DISTRICT', 'YEAR'])['T2M_RANGE']
    .mean()
    .reset_index()
)

# === Step 3: Run Mann-Kendall trend test per district ===
district_trends = []

for district, group in annual_range_by_dist.groupby('DISTRICT'):
    series = group.sort_values('YEAR')['T2M_RANGE']
    if len(series) >= 10:
        result = mk.original_test(series)
        district_trends.append({
            'DISTRICT': district,
            'trend': result.trend,
            'p_value': result.p,
            'slope': result.slope,
            'significant': int(result.h)
        })

# === Step 4: Create trend DataFrame ===
trend_df = pd.DataFrame(district_trends)
trend_df['DISTRICT'] = trend_df['DISTRICT'].str.strip().str.upper()

# === Step 5: Load and prep GeoDataFrame if not preloaded ===
try:
    gdf_districts
except NameError:
    shapefile_path = "../data/local_unit_shapefiles/local_unit.shp"
    gdf = gpd.read_file(shapefile_path)
    gdf['DISTRICT'] = gdf['DISTRICT'].str.strip().str.upper()
    gdf_districts = gdf.dissolve(by='DISTRICT', as_index=False)

# === Step 6: Merge trend results into GeoDataFrame ===
gdf_range_trend = gdf_districts.merge(trend_df, on='DISTRICT', how='left')

# === Step 7: Plot slope of T2M_RANGE trend per district ===
fig, ax = plt.subplots(figsize=(10, 8))
gdf_range_trend.plot(
    column='slope',
    cmap='coolwarm_r',
    legend=True,
    ax=ax,
    edgecolor='black',
    linewidth=0.5,
    missing_kwds={'color': 'lightgrey', 'label': 'No data'}
)

ax.set_title("Trend in Daily Temperature Range (T2M_RANGE) by District", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# #### ✅ Chi-Square Test (Extreme Event and Season) 

# In[ ]:


import pandas as pd
from scipy.stats import chi2_contingency

# Step 1: Drop rows with missing values and create a clean copy
climate_df_clean = climate_df.dropna(subset=['T2M_MAX', 'Season', 'YEAR']).copy()

# Step 2: Create binary column: Extreme Heat (T2M_MAX > 35°C)
climate_df_clean['ExtremeHeat'] = (climate_df_clean['T2M_MAX'] > 35).astype(int)

# --- Chi-square Test: Season vs Extreme Heat ---
season_ct = pd.crosstab(climate_df_clean['Season'], climate_df_clean['ExtremeHeat'])
chi2_season, p_season, dof_season, expected_season = chi2_contingency(season_ct)
season_status = "✅ Significant" if p_season < 0.05 else "❌ Not Significant"

print("📊 Chi-square Test – Season vs Extreme Heat")
print(f"χ² = {chi2_season:.2f}, df = {dof_season}, p = {p_season:.4f} → {season_status}")

# --- Chi-square Test: Year vs Extreme Heat ---
year_ct = pd.crosstab(climate_df_clean['YEAR'], climate_df_clean['ExtremeHeat'])
chi2_year, p_year, dof_year, expected_year = chi2_contingency(year_ct)
year_status = "✅ Significant" if p_year < 0.05 else "❌ Not Significant"

print("\n📊 Chi-square Test – Year vs Extreme Heat")
print(f"χ² = {chi2_year:.2f}, df = {dof_year}, p = {p_year:.4f} → {year_status}")


# #### ✅ OLS Regression (Predict precipitation from temperature/humidity/etc.)

# In[ ]:


import statsmodels.api as sm
import pandas as pd

# Step 1: Drop rows with missing values in predictors or response
regression_df = climate_df[['T2M', 'RH2M', 'PRECTOT']].dropna().copy()

# Step 2: Define predictors (X) and response variable (y)
X = regression_df[['T2M', 'RH2M']]   # Independent variables
y = regression_df['PRECTOT']         # Dependent variable

# Step 3: Add constant term to predictors (intercept)
X = sm.add_constant(X)

# Step 4: Fit the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Step 5: Print regression summary
print(model.summary())


# In[ ]:


import pandas as pd
import statsmodels.api as sm

# Step 1: Clean and define your features
extreme_df = climate_df[['PRECTOT', 'T2M', 'RH2M']].dropna().copy()

# Step 2: Create binary target for heavy rainfall (1 = extreme, 0 = not)
extreme_df['HeavyRain'] = (extreme_df['PRECTOT'] > 50).astype(int)

# Step 3: Define predictors and response
X = extreme_df[['T2M', 'RH2M']]      # Features
y = extreme_df['HeavyRain']          # Binary target

# Step 4: Add intercept
X = sm.add_constant(X)

# Step 5: Fit logistic regression model
logit_model = sm.Logit(y, X).fit()

# Step 6: Output model summary
print(logit_model.summary())

