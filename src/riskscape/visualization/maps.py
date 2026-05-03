"""Map model prediction outputs."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

from riskscape.config import cfg, paths
from riskscape.grid import load_grid


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def prediction_path(year: int) -> Path:
    """Return prediction partition path."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / f"year={year}"
        / "part.parquet"
    )


def figure_root() -> Path:
    """Return figure output directory."""
    path = paths["data"] / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_references() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load land and coastline references."""
    land = gpd.read_file(PROJECT_ROOT / cfg["references"]["land"])
    coast = gpd.read_file(PROJECT_ROOT / cfg["references"]["coastline"])
    return land, coast


def bbox_gdf() -> gpd.GeoDataFrame:
    """Return region bounding box as GeoDataFrame."""
    bbox = cfg["region"]["bbox"]

    geom = box(
        bbox["xmin"],
        bbox["ymin"],
        bbox["xmax"],
        bbox["ymax"],
    )

    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


def set_region_extent(ax) -> None:
    """Set plot extent from configured region."""
    bbox = cfg["region"]["bbox"]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    mid_lat = (ymin + ymax) / 2
    scale = math.cos(math.radians(mid_lat))
    margin = 0.5

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin * scale, ymax + margin * scale)


def load_predictions(year: int) -> pd.DataFrame:
    """Load prediction output for one year."""
    path = prediction_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    return pd.read_parquet(path)


def summarize_h3(
    df: pd.DataFrame,
    value_col: str,
    species: str | None = None,
    month: int | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Summarize prediction values by H3 cell."""
    out = df.copy()

    if species is not None:
        out = out[out["species"] == species]

    if month is not None:
        out = out[out["date"].dt.month == month]

    out = out[out[value_col] > 0]

    if out.empty:
        raise ValueError("No nonzero prediction rows found")

    grouped = (
        out.groupby("h3", as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: f"{value_col}_{agg}"})
    )

    return grouped


def add_risk_class(
    gdf: gpd.GeoDataFrame,
    value_col: str,
) -> gpd.GeoDataFrame:
    """Add percentile risk class for nonzero values."""
    out = gdf.copy()
    values = out[value_col].dropna()

    if values.empty:
        raise ValueError(f"No values found for {value_col}")

    q90 = values.quantile(0.90)
    q95 = values.quantile(0.95)
    q99 = values.quantile(0.99)

    out["risk_class"] = "none"
    out.loc[out[value_col] > 0, "risk_class"] = "low"
    out.loc[out[value_col] >= q90, "risk_class"] = "high"
    out.loc[out[value_col] >= q95, "risk_class"] = "very_high"
    out.loc[out[value_col] >= q99, "risk_class"] = "extreme"

    return out


def plot_h3_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    out_file: Path,
    cmap: str = "inferno",
) -> Path:
    """Plot H3 map with land and coastline."""
    land, coast = load_references()
    bbox = bbox_gdf()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#e5fbfa")

    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        legend=True,
        edgecolor="none",
        linewidth=0,
        missing_kwds={
            "color": "white",
            "alpha": 0.0,
        },
    )

    coast.plot(ax=ax, color="darkgrey", linewidth=0.5)
    land.plot(ax=ax, color="grey", edgecolor="none")
    bbox.boundary.plot(ax=ax, edgecolor="red", linewidth=1)

    set_region_extent(ax)

    ax.set_title(title)
    ax.set_axis_off()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def plot_prediction_map(
    year: int,
    value_col: str,
    species: str | None = None,
    month: int | None = None,
    agg: str = "mean",
) -> Path:
    """Plot summarized prediction map."""
    df = load_predictions(year)

    summary = summarize_h3(
        df=df,
        value_col=value_col,
        species=species,
        month=month,
        agg=agg,
    )

    grid = load_grid(uint64=True)
    value_name = f"{value_col}_{agg}"

    gdf = grid.merge(summary, on="h3", how="left")

    species_label = species if species is not None else "all_species"
    month_label = f"month_{month:02d}" if month is not None else "all_months"

    title = (
        f"{value_col} ({agg}) - {species_label} - "
        f"{year} - {month_label}"
    )

    out_file = (
        figure_root()
        / f"{value_col}_{agg}_{species_label}_{year}_{month_label}.png"
    )

    return plot_h3_map(
        gdf=gdf,
        value_col=value_name,
        title=title,
        out_file=out_file,
    )