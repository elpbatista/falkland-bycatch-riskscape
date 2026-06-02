"""Plot static H3 feature maps."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from dataclasses import dataclass
from typing import cast

import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.maps import MapStyle, plot_h3_map


INPUT_FILE = paths["data"] / "features" / "static" / "static.parquet"
OUTPUT_ROOT = paths["plots"] / "static_features"


@dataclass(frozen=True)
class StaticFeatureSpec:
    """Display and scaling settings for one static feature."""

    column: str
    display_column: str
    label: str
    colorbar: str
    cmap: str
    quantile: float = 0.99
    invert_colorbar: bool = False
    color_min: float | None = None
    color_scale: str = "linear"


FEATURE_SPECS = [
    StaticFeatureSpec(
        column="depth_m",
        display_column="depth_m",
        label="Bathymetric Depth",
        colorbar="Bathymetric depth (m)",
        cmap="Blues",
        quantile=1.0,
        invert_colorbar=True,
        color_min=0.0,
        color_scale="sqrt",
    ),
    StaticFeatureSpec(
        column="slope",
        display_column="slope",
        label="Bathymetric Slope",
        colorbar="Bathymetric slope (m/m)",
        cmap="cividis",
    ),
    StaticFeatureSpec(
        column="dist_coast_m",
        display_column="dist_coast_km",
        label="Distance to Coast",
        colorbar="Distance to coast (km)",
        cmap="viridis",
    ),
]


def selected_specs(feature_names: str | None) -> list[StaticFeatureSpec]:
    """Return feature specs after applying an optional CLI filter."""
    if feature_names is None:
        return FEATURE_SPECS

    requested = {
        name.strip()
        for name in feature_names.split(",")
        if name.strip()
    }
    specs = [
        spec
        for spec in FEATURE_SPECS
        if spec.column in requested or spec.display_column in requested
    ]
    known = {spec.column for spec in FEATURE_SPECS} | {
        spec.display_column for spec in FEATURE_SPECS
    }
    missing = requested - known

    if missing:
        raise ValueError("Unknown static features requested: " + ", ".join(sorted(missing)))

    return specs


def load_static_features(specs: list[StaticFeatureSpec]) -> pd.DataFrame:
    """Load selected static H3 feature columns."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Static feature file not found: {INPUT_FILE}")

    columns = ["h3", *sorted({spec.column for spec in specs})]
    out = pd.read_parquet(INPUT_FILE, columns=columns)
    out["h3"] = out["h3"].astype("uint64")

    if "dist_coast_m" in out.columns:
        out["dist_coast_km"] = out["dist_coast_m"] / 1000.0
    if "depth_m" in out.columns:
        out["depth_m"] = out["depth_m"].clip(lower=0.0)

    return out


def feature_limits(values: pd.Series, spec: StaticFeatureSpec) -> tuple[float, float]:
    """Return color limits for a static feature map."""
    clean = values.dropna()
    if clean.empty:
        raise ValueError(f"No values found for {spec.display_column}")

    color_min = (
        spec.color_min
        if spec.color_min is not None
        else float(clean.quantile(1.0 - spec.quantile))
    )
    color_max = float(clean.quantile(spec.quantile))

    if color_max <= color_min:
        color_min = float(clean.min())
        color_max = float(clean.max())
    if color_max <= color_min:
        color_max = color_min + 1.0

    return color_min, color_max


def endpoint_label(value: float) -> str:
    """Return compact endpoint label text for a colorbar."""
    if abs(value) < 1e-9:
        return "0"

    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot_static_feature_maps(
    df: pd.DataFrame,
    specs: list[StaticFeatureSpec],
) -> list[Path]:
    """Plot one H3 map per selected static feature."""
    grid = load_grid(uint64=True)
    gdf = grid.merge(df, on="h3", how="left")
    out_files: list[Path] = []

    for spec in specs:
        values = cast(pd.Series, gdf[spec.display_column])
        color_min, color_max = feature_limits(values, spec)
        out_file = OUTPUT_ROOT / f"{spec.display_column}.png"
        bottom_label = endpoint_label(color_max if spec.invert_colorbar else color_min)
        top_label = endpoint_label(color_min if spec.invert_colorbar else color_max)
        out_files.append(
            plot_h3_map(
                gdf=gdf,
                value_col=spec.display_column,
                title=spec.label,
                out_file=out_file,
                style=MapStyle(
                    legend_mode="continuous",
                    cmap=spec.cmap,
                    color_scale=spec.color_scale,
                    colorbar_title=spec.colorbar,
                    color_min=color_min,
                    color_max=color_max,
                    color_quantile=None,
                    colorbar_bottom_label=bottom_label,
                    colorbar_top_label=top_label,
                    colorbar_invert=spec.invert_colorbar,
                    hide_zero_values=False,
                    min_display_value=None,
                    alpha_scale=False,
                    bathymetry=False,
                    bathymetry_log_scale=False,
                    show_reference_map=False,
                ),
            )
        )

    return out_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create H3 maps for static bathymetric and coastal features."
    )
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Optional comma-separated static feature columns. "
            "Defaults to slope and dist_coast_km."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run static feature map plotting."""
    args = parse_args()
    specs = selected_specs(args.features)
    df = load_static_features(specs)
    out_files = plot_static_feature_maps(df=df, specs=specs)

    for out_file in out_files:
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
