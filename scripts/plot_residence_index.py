# Plot BBAL residence index heatmap on reference map

import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from riskscape.config import PROJECT_ROOT, paths
from riskscape.grid import load_grid

from riskscape.config import cfg


# --- Config ------------------------------------------------------------------

species_name = "BBAL"
years = (2022, 2023)

bbox = cfg["region"]["bbox"]

xmin = bbox["xmin"]
ymin = bbox["ymin"]
xmax = bbox["xmax"]
ymax = bbox["ymax"]

bbox_poly = box(xmin, ymin, xmax, ymax)
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")


# --- Load reference layers ----------------------------------------------------

land = gpd.read_file(PROJECT_ROOT / cfg["references"]["land"])
coast = gpd.read_file(PROJECT_ROOT / cfg["references"]["coastline"])


# --- Load species presence ----------------------------------------------------

frames = []

for year in years:
    path = (
        paths["data"]
        / "features"
        / "species_presence"
        / f"year={year}"
        / "part.parquet"
    )

    if not path.exists():
        continue

    df_year = pd.read_parquet(path)
    df_year = df_year[df_year["species"] == species_name].copy()
    df_year["year"] = year

    frames.append(df_year)

if not frames:
    raise RuntimeError(f"No records found for species: {species_name}")

df = pd.concat(frames, ignore_index=True)

df["residence_index"] = df["presence_count"] / df["individual_count"]


# --- Aggregate to H3 ----------------------------------------------------------

residence = (
    df.groupby("h3", as_index=False)
    .agg(
        residence_index=("residence_index", "mean"),
        presence_count=("presence_count", "sum"),
        individual_count=("individual_count", "max"),
        days=("date", "nunique"),
    )
)

residence["ri_log"] = np.log1p(residence["residence_index"])


# --- Load grid and join -------------------------------------------------------

grid = load_grid(uint64=True)

plot_gdf = grid.merge(residence, on="h3", how="left")


# --- Plot --------------------------------------------------------------------

ax = land.plot(color="grey", edgecolor="none", figsize=(10, 10))

ax.set_facecolor("#e5fbfa")

coast.plot(ax=ax, color="darkgrey", linewidth=0.5)

grid.plot(ax=ax, edgecolor="darkgrey", facecolor="none", linewidth=0.2)

plot_gdf.dropna(subset=["ri_log"]).plot(
    ax=ax,
    column="ri_log",
    legend=True,
    edgecolor="none",
    linewidth=0,
)

bbox_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1)

mid_lat = (ymin + ymax) / 2
scale = math.cos(math.radians(mid_lat))
margin = 0.5

ax.set_xlim(xmin - margin, xmax + margin)
ax.set_ylim(ymin - margin * scale, ymax + margin * scale)

ax.set_title(f"{species_name} Residence Index Heatmap")

plt.show()