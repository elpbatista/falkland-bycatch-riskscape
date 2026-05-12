"""Plot weekly latent-risk operator climatology maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    MAP_CRS,
    MapBounds,
    OCEAN_COLOR,
    draw_bathymetry_base_layer,
    draw_reference_layers,
    load_reference_layers,
)
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    SPECIES_USE_LOG_COLOR_MAX,
)


MODEL_NAME = "hybrid_presence_gate_extra_trees_kmeans_k15_blockcv_bayesian_gmm_k30"
PRODUCT_NAME = "joint"
START_YEAR = 2014
END_YEAR = 2023
SPECIES = ("BBAL", "SAFS")
REPRESENTATIVE_WEEKS = (3, 24, 36, 50)
CMAP = "YlOrRd"
RISK_ALPHA = 0.90
OUTPUT_ROOT = paths["plots"] / "predictions" / "weekly_operator"


def climatology_path(
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
) -> Path:
    """Return weekly climatology parquet path."""
    return (
        paths["data"]
        / "modeling"
        / "weekly_operator"
        / model_name
        / product_name
        / f"latent_risk_iso_week_climatology_{start_year}-{end_year}.parquet"
    )


def load_weekly_climatology(path: Path) -> pd.DataFrame:
    """Load weekly latent-risk climatology."""
    if not path.exists():
        raise FileNotFoundError(f"Weekly climatology not found: {path}")

    out = pd.read_parquet(path)
    out["h3"] = out["h3"].astype("uint64")
    return out


def risk_norm() -> colors.LogNorm:
    """Return fixed latent-risk color scale."""
    baseline = float(np.log1p(MINIMUM_EFFORT_UNIT))
    return colors.LogNorm(
        vmin=baseline,
        vmax=float(SPECIES_USE_LOG_COLOR_MAX + baseline),
    )


def panel_title(species: str, week: int) -> str:
    """Return panel title."""
    return f"{species} — ISO week {week:02d}"


def plot_week_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    climatology: pd.DataFrame,
    species: str,
    week: int,
    norm: colors.LogNorm,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one weekly climatology panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)
    draw_bathymetry_base_layer(ax, legend=False, draw_grid=False)

    mask = (
        cast(pd.Series, climatology["species"]).eq(species)
        & cast(pd.Series, climatology["iso_week"]).eq(week)
    )
    values = climatology.loc[
        mask,
        ["h3", "display_latent_risk_log_pred_mean"],
    ]
    plot_gdf = grid.merge(values, on="h3", how="left")
    plot_gdf = plot_gdf.dropna(subset=["display_latent_risk_log_pred_mean"])

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            column="display_latent_risk_log_pred_mean",
            cmap=CMAP,
            norm=norm,
            legend=False,
            alpha=RISK_ALPHA,
            edgecolor="none",
            linewidth=0,
        )

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[bounds.geometry()],
        crs=grid.crs or MAP_CRS,
    )
    draw_reference_layers(ax, bbox_gdf, land, coast)
    ax.set_title(panel_title(species, week), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def output_path(
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
) -> Path:
    """Return output figure path."""
    return (
        OUTPUT_ROOT
        / (
            f"{model_name}_{product_name}_latent_risk_iso_week_"
            f"climatology_{start_year}-{end_year}_small_multiples.png"
        )
    )


def plot_small_multiples(
    climatology: pd.DataFrame,
    species_values: list[str],
    weeks: list[int],
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
) -> Path:
    """Plot species by representative-week climatology small multiples."""
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    norm = risk_norm()

    fig, axes = plt.subplots(
        nrows=len(species_values),
        ncols=len(weeks),
        figsize=(4.0 * len(weeks), 5.1 * len(species_values)),
        constrained_layout=False,
    )
    axes_array = np.atleast_2d(axes)

    for row, species in enumerate(species_values):
        for col, week in enumerate(weeks):
            plot_week_panel(
                ax=cast(Axes, axes_array[row, col]),
                grid=grid,
                climatology=climatology,
                species=species,
                week=week,
                norm=norm,
                bounds=bounds,
                land=land,
                coast=coast,
            )

    fig.suptitle(
        f"Weekly Latent-Risk Climatology — {start_year}-{end_year}",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.91,
        top=0.93,
        bottom=0.035,
        wspace=0.09,
        hspace=0.16,
    )

    cax = fig.add_axes((0.935, 0.19, 0.018, 0.62))
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP)),
        cax=cax,
    )
    cbar.set_label("Latent Risk", fontsize=10)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_ticks([])
    cbar.ax.tick_params(which="both", length=0, labelleft=False, labelright=False)

    out_file = output_path(
        model_name=model_name,
        product_name=product_name,
        start_year=start_year,
        end_year=end_year,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot weekly latent-risk climatology small multiples.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--product-name", default=PRODUCT_NAME)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--species", nargs="+", default=list(SPECIES))
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        default=list(REPRESENTATIVE_WEEKS),
    )
    return parser.parse_args()


def main() -> int:
    """Run weekly climatology plotting."""
    args = parse_args()
    path = climatology_path(
        model_name=args.model_name,
        product_name=args.product_name,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    climatology = load_weekly_climatology(path)
    out_file = plot_small_multiples(
        climatology=climatology,
        species_values=list(args.species),
        weeks=list(args.weeks),
        model_name=args.model_name,
        product_name=args.product_name,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"Saved: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
