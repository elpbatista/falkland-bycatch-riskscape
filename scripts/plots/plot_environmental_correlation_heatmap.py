"""Plot a correlation heatmap for environmental predictors."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths


YEAR = "2022"
METHOD = "spearman"
SAMPLE_SIZE = 250_000
INPUT_ROOT = paths["data"] / "features" / "environmental"
OUTPUT_ROOT = paths["plots"] / "environmental_correlation"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "environmental_correlation"


@dataclass(frozen=True)
class PredictorSpec:
    """Display settings for one environmental predictor."""

    column: str
    label: str


# Comment predictors in or out here to choose what appears in the heatmap.
PREDICTOR_SPECS = [
    PredictorSpec("sst", "SST"),
    PredictorSpec("ssh", "SSH"),
    PredictorSpec("wind_speed", "Wind speed"),
    PredictorSpec("chl_log", "Log-CHL"),
    PredictorSpec("sst_anom", "SST anomaly"),
    PredictorSpec("ssh_anom", "SSH anomaly"),
    PredictorSpec("wind_speed_anom", "Wind-speed anomaly"),
    PredictorSpec("chl_log_anom", "Log-CHL anomaly"),
    PredictorSpec("sst_grad", "SST gradient"),
    PredictorSpec("ssh_grad", "SSH gradient"),
    PredictorSpec("chl_log_grad", "Log-CHL gradient"),
]


def environmental_path(year: int) -> Path:
    """Return one yearly environmental feature path."""
    return INPUT_ROOT / f"year={year}" / "part.parquet"


def years_to_load(year: str) -> list[int]:
    """Return years to load from the CLI year argument."""
    if year.lower() == "all":
        years: list[int] = []
        for year_dir in sorted(INPUT_ROOT.glob("year=*")):
            years.append(int(year_dir.name.split("=", maxsplit=1)[1]))
        return years

    return [int(year)]


def selected_specs(variable_names: str | None) -> list[PredictorSpec]:
    """Return predictor specs after applying an optional CLI filter."""
    if variable_names is None:
        return PREDICTOR_SPECS

    requested = {
        name.strip()
        for name in variable_names.split(",")
        if name.strip()
    }
    specs = [spec for spec in PREDICTOR_SPECS if spec.column in requested]
    missing = requested - {spec.column for spec in specs}

    if missing:
        raise ValueError(
            "Unknown or commented-out predictors requested: "
            + ", ".join(sorted(missing))
        )

    return specs


def load_year_frame(year: int, specs: list[PredictorSpec]) -> pd.DataFrame:
    """Load selected predictor columns for one year."""
    path = environmental_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Environmental feature file not found: {path}")

    return pd.read_parquet(path, columns=[spec.column for spec in specs])


def sample_frame(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Return a deterministic sample, or the full frame if sample_size is zero."""
    if sample_size <= 0 or len(df) <= sample_size:
        return df

    return df.sample(n=sample_size, random_state=random_state)


def load_environmental_predictors(
    year: str,
    specs: list[PredictorSpec],
    sample_size: int,
) -> pd.DataFrame:
    """Load and optionally sample environmental predictors."""
    frames: list[pd.DataFrame] = []

    for index, selected_year in enumerate(years_to_load(year)):
        frame = load_year_frame(selected_year, specs)
        frames.append(
            sample_frame(
                frame,
                sample_size=sample_size,
                random_state=selected_year + index,
            )
        )

    if not frames:
        raise FileNotFoundError(f"No environmental year partitions found: {INPUT_ROOT}")

    return pd.concat(frames, ignore_index=True)


def compute_correlation(
    df: pd.DataFrame,
    specs: list[PredictorSpec],
    method: str,
) -> pd.DataFrame:
    """Compute the predictor correlation matrix."""
    columns = [spec.column for spec in specs]
    labels = [spec.label for spec in specs]
    corr = df[columns].dropna().corr(method=method)
    corr.index = labels
    corr.columns = labels

    return corr


def save_matrix(corr: pd.DataFrame, out_file: Path) -> None:
    """Save the correlation matrix as CSV."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_file)


def annotation_color(value: float) -> str:
    """Return readable annotation text color for a heatmap cell."""
    return "white" if abs(value) >= 0.55 else "#222222"


def plot_heatmap(
    corr: pd.DataFrame,
    title: str,
    out_file: Path,
) -> Path:
    """Plot and save an annotated correlation heatmap."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    values = corr.to_numpy(dtype="float64")
    labels = list(corr.columns)
    fig_size = max(8.0, len(labels) * 0.72)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    image = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r")
    cbar = fig.colorbar(image, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("correlation", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="both", length=0)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=annotation_color(value),
                fontsize=7,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def output_stem(year: str, method: str) -> str:
    """Return a stable output stem."""
    return f"environmental_predictors_{method}_correlation_{year}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a correlation heatmap for environmental predictors."
    )
    parser.add_argument(
        "--year",
        default=YEAR,
        help="Year to use, or 'all' for all environmental partitions.",
    )
    parser.add_argument(
        "--method",
        default=METHOD,
        choices=("pearson", "spearman", "kendall"),
        help="Correlation method.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Rows sampled per year. Use 0 to use all rows.",
    )
    parser.add_argument(
        "--variables",
        default=None,
        help=(
            "Optional comma-separated predictor columns. "
            "By default, uncommented PREDICTOR_SPECS entries are used."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run environmental correlation heatmap plotting."""
    args = parse_args()
    specs = selected_specs(args.variables)
    df = load_environmental_predictors(
        year=args.year,
        specs=specs,
        sample_size=args.sample_size,
    )
    corr = compute_correlation(df, specs=specs, method=args.method)

    stem = output_stem(year=args.year, method=args.method)
    matrix_path = DATA_OUTPUT_ROOT / f"{stem}.csv"
    figure_path = OUTPUT_ROOT / f"{stem}.png"

    save_matrix(corr, matrix_path)
    plot_heatmap(
        corr,
        title=f"Environmental Predictor Correlation ({args.method.title()})",
        out_file=figure_path,
    )

    print(f"Saved: {matrix_path}")
    print(f"Saved: {figure_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
