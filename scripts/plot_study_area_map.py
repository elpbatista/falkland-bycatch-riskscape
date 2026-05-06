"""Plot study area with fisheries and conservation-zone reference layers."""

from __future__ import annotations

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from riskscape.config import PROJECT_ROOT, cfg, paths
from riskscape.visualization.base_map import (
    draw_bathymetry_base_layer,
    draw_map_context,
    format_map_axes,
    load_reference_layers,
    setup_map,
)


def load_study_area_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load fisheries grid and FICZ/FOCZ limits."""
    fisheries = gpd.read_file(PROJECT_ROOT / cfg["references"]["fisheries"])
    limits = gpd.read_file(PROJECT_ROOT / cfg["references"]["limits"])

    return fisheries.to_crs("EPSG:4326"), limits.to_crs("EPSG:4326")


def plot_study_area_map() -> None:
    """Plot the study area reference layers."""
    title = "Falkland Islands Study Area"

    land, coast = load_reference_layers()
    fisheries, limits = load_study_area_layers()
    fig, ax, bbox_gdf = setup_map()

    draw_bathymetry_base_layer(
        ax,
        legend=True,
        draw_grid=True,
    )

    fisheries.plot(
        ax=ax,
        edgecolor="#f5a623",
        color="none",
        linewidth=0.45,
        alpha=0.75,
    )
    limits.plot(
        ax=ax,
        edgecolor="#f5a623",
        color="none",
        linewidth=1.1,
        alpha=0.95,
    )

    draw_map_context(ax, bbox_gdf, land, coast)
    format_map_axes(ax, title)

    out_file = paths["plots"] / "study_area_reference_layers.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_file}")


def main() -> int:
    """Run study area map plot."""
    plot_study_area_map()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
