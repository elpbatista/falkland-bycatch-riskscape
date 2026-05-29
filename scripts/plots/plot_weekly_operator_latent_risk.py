"""Plot weekly latent-risk operator climatology maps."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path
import sys
from typing import cast

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.base_map import MapBounds, load_reference_layers
from riskscape.visualization.maps import (
    MapStyle,
    color_norm,
    draw_h3_value_panel,
    draw_prediction_colorbar,
)
from riskscape.visualization.weekly_maps import (
    add_weekly_colorbar_axis,
    create_weekly_frame,
    create_weekly_map_grid,
    encode_mp4,
    format_week_panel,
    format_weekly_frame,
    save_weekly_map,
    weekly_axes,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_prediction_maps import (  # noqa: E402
    AGG as PREDICTION_MAP_AGG,
    YEAR as PREDICTION_MAP_YEAR,
    load_aggregated_predictions,
    prediction_path,
    shared_styles,
)


MODEL_NAME = "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30"
PRODUCT_NAME = "joint"
START_YEAR = 2014
END_YEAR = 2023
SEQUENCE_YEAR = 2022
SPECIES = ("BBAL", "SAFS")
REPRESENTATIVE_WEEKS = (12, 24, 36, 48)
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


def sequence_path(
    model_name: str,
    product_name: str,
    sequence_year: int,
) -> Path:
    """Return weekly sequence parquet path."""
    return (
        paths["data"]
        / "modeling"
        / "weekly_operator"
        / model_name
        / product_name
        / f"latent_risk_iso_week_sequence_{sequence_year}.parquet"
    )


def load_weekly_climatology(path: Path) -> pd.DataFrame:
    """Load weekly latent-risk climatology."""
    if not path.exists():
        raise FileNotFoundError(f"Weekly climatology not found: {path}")

    out = pd.read_parquet(path)
    out["h3"] = out["h3"].astype("uint64")
    return out


def load_weekly_sequence(path: Path) -> pd.DataFrame:
    """Load weekly latent-risk sequence."""
    if not path.exists():
        raise FileNotFoundError(f"Weekly sequence not found: {path}")

    out = pd.read_parquet(path)
    out["h3"] = out["h3"].astype("uint64")
    return out


def risk_style(
    model_name: str,
    product_name: str,
    species_values: list[str],
) -> MapStyle:
    """Return the same binned latent-risk style used by prediction maps."""
    path = prediction_path(PREDICTION_MAP_YEAR, model_name, product_name)
    predictions = load_aggregated_predictions(
        path=path,
        species_values=species_values,
        agg=PREDICTION_MAP_AGG,
        plausibility_agg=PREDICTION_MAP_AGG,
    )
    _, _, hazard_style, _ = shared_styles(
        predictions=predictions,
        species=species_values,
        agg=PREDICTION_MAP_AGG,
    )
    return hazard_style


def risk_norm(style: MapStyle) -> colors.BoundaryNorm:
    """Return the binned latent-risk color scale."""
    if style.colorbar_boundaries is None:
        raise ValueError("Weekly latent-risk style requires colorbar boundaries")
    norm = color_norm(
        pd.Series(style.colorbar_boundaries, dtype="float64"),
        style,
    )
    if not isinstance(norm, colors.BoundaryNorm):
        raise TypeError("Weekly latent-risk style requires BoundaryNorm")
    return norm


def plot_week_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    climatology: pd.DataFrame,
    species: str,
    week: int,
    norm: colors.BoundaryNorm,
    style: MapStyle,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one weekly climatology panel."""
    format_week_panel(ax, species=species, week=week, bounds=bounds)

    mask = (
        cast(pd.Series, climatology["species"]).eq(species)
        & cast(pd.Series, climatology["iso_week"]).eq(week)
    )
    values = climatology.loc[
        mask,
        ["h3", "display_latent_risk_log_pred_mean"],
    ]
    draw_h3_value_panel(
        ax=ax,
        grid=grid,
        values=values,
        value_col="display_latent_risk_log_pred_mean",
        norm=norm,
        style=style,
        bounds=bounds,
        land=land,
        coast=coast,
        draw_bathymetry=True,
    )


def add_risk_colorbar(
    ax: Axes,
    norm: colors.BoundaryNorm,
    style: MapStyle,
    cax: Axes | None = None,
) -> None:
    """Add the same labeled latent-risk colorbar used by prediction maps."""
    draw_prediction_colorbar(
        ax=ax,
        value_col="display_latent_risk_log_pred_mean",
        norm=norm,
        style=style,
        cax=cax,
    )


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


def animation_output_path(
    model_name: str,
    product_name: str,
    sequence_year: int,
    species: str,
) -> Path:
    """Return MP4 output path for one species weekly sequence."""
    return (
        OUTPUT_ROOT
        / (
            f"{model_name}_{product_name}_latent_risk_iso_week_"
            f"sequence_{sequence_year}_{species}.mp4"
        )
    )


def frame_output_dir(
    model_name: str,
    product_name: str,
    sequence_year: int,
    species: str,
) -> Path:
    """Return local frame directory for one species weekly sequence."""
    return (
        OUTPUT_ROOT
        / "frames"
        / f"{model_name}_{product_name}_latent_risk_iso_week_sequence_{sequence_year}_{species}"
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
    style = risk_style(model_name, product_name, species_values)
    norm = risk_norm(style)

    fig, axes_array = create_weekly_map_grid(
        species_values=species_values,
        weeks=weeks,
        title=f"Weekly Latent-Risk Climatology — {start_year}-{end_year}",
    )

    for ax, species, week in weekly_axes(axes_array, species_values, weeks):
        plot_week_panel(
            ax=ax,
            grid=grid,
            climatology=climatology,
            species=species,
            week=week,
            norm=norm,
            style=style,
            bounds=bounds,
            land=land,
            coast=coast,
        )

    cax = add_weekly_colorbar_axis(fig)
    add_risk_colorbar(
        ax=cast(Axes, axes_array[0, -1]),
        norm=norm,
        style=style,
        cax=cast(Axes, cax),
    )

    out_file = output_path(
        model_name=model_name,
        product_name=product_name,
        start_year=start_year,
        end_year=end_year,
    )
    return save_weekly_map(fig, out_file)


def render_animation_frames(
    sequence: pd.DataFrame,
    species: str,
    weeks: list[int],
    grid: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
    bounds: MapBounds,
    norm: colors.BoundaryNorm,
    style: MapStyle,
    out_dir: Path,
    sequence_year: int,
    dpi: int,
) -> list[Path]:
    """Render one PNG frame per week using the weekly climatology map style."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("week_*.png"):
        old_frame.unlink()

    frames: list[Path] = []

    for index, week in enumerate(weeks, start=1):
        fig, ax = create_weekly_frame()
        plot_week_panel(
            ax=ax,
            grid=grid,
            climatology=sequence,
            species=species,
            week=week,
            norm=norm,
            style=style,
            bounds=bounds,
            land=land,
            coast=coast,
        )
        ax.set_title(f"{species} — {sequence_year} ISO week {week:02d}", fontsize=13)
        format_weekly_frame(fig)
        add_risk_colorbar(
            ax=ax,
            norm=norm,
            style=style,
        )

        frame_path = out_dir / f"week_{index:03d}_iso_week_{week:02d}.png"
        save_weekly_map(fig, frame_path, dpi=dpi)
        frames.append(frame_path)

    return frames


def plot_animations(
    sequence: pd.DataFrame,
    species_values: list[str],
    model_name: str,
    product_name: str,
    sequence_year: int,
    fps: int,
    dpi: int,
) -> list[Path]:
    """Render and encode weekly MP4 animations for each species."""
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    style = risk_style(model_name, product_name, species_values)
    norm = risk_norm(style)
    weeks = sorted(int(week) for week in sequence["iso_week"].dropna().unique())
    out_files: list[Path] = []

    for species in species_values:
        frames_dir = frame_output_dir(
            model_name=model_name,
            product_name=product_name,
            sequence_year=sequence_year,
            species=species,
        )
        render_animation_frames(
            sequence=sequence,
            species=species,
            weeks=weeks,
            grid=grid,
            land=land,
            coast=coast,
            bounds=bounds,
            norm=norm,
            style=style,
            out_dir=frames_dir,
            sequence_year=sequence_year,
            dpi=dpi,
        )
        out_file = animation_output_path(
            model_name=model_name,
            product_name=product_name,
            sequence_year=sequence_year,
            species=species,
        )
        encode_mp4(frame_dir=frames_dir, out_file=out_file, fps=fps)
        out_files.append(out_file)

    return out_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot weekly latent-risk climatology and animation maps.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--product-name", default=PRODUCT_NAME)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--sequence-year", type=int, default=SEQUENCE_YEAR)
    parser.add_argument("--species", nargs="+", default=list(SPECIES))
    parser.add_argument("--make-small-multiples", action="store_true")
    parser.add_argument("--make-animation", action="store_true")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=180)
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
    if not args.make_small_multiples and not args.make_animation:
        args.make_small_multiples = True

    if args.make_small_multiples:
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

    if args.make_animation:
        path = sequence_path(
            model_name=args.model_name,
            product_name=args.product_name,
            sequence_year=args.sequence_year,
        )
        sequence = load_weekly_sequence(path)
        for out_file in plot_animations(
            sequence=sequence,
            species_values=list(args.species),
            model_name=args.model_name,
            product_name=args.product_name,
            sequence_year=args.sequence_year,
            fps=args.fps,
            dpi=args.dpi,
        ):
            print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
