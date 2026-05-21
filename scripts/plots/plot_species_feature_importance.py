"""Plot feature importance for the selected species-use model."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import paths


INPUT_FILE = paths["data"] / "modeling" / "models" / "species_feature_importance.csv"
OUTPUT_ROOT = paths["plots"] / "species_use_diagnostics"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "species_use_diagnostics"
TOP_N = 18

DISPLAY_NAMES = {
    "depth_m": "Bathymetry",
    "dist_coast_m": "Distance to coast",
    "ssh_anom": "SSH anomaly",
    "ssh": "SSH",
    "slope": "Seafloor slope",
    "sst": "SST",
    "chl_log_grad": "CHL gradient",
    "sst_grad": "SST gradient",
    "ssh_grad": "SSH gradient",
    "chl_log_anom": "CHL anomaly",
    "sst_anom": "SST anomaly",
    "chl_log": "CHL",
    "wind_speed_anom": "Wind-speed anomaly",
    "wind_speed": "Wind speed",
    "doy_sin": "Seasonality, sine",
    "doy_cos": "Seasonality, cosine",
    "species_SAFS": "Species: SAFS",
    "species_BBAL": "Species: BBAL",
}


def load_importance(input_file: Path, top_n: int) -> pd.DataFrame:
    """Load and prepare feature-importance values."""
    df = pd.read_csv(input_file)
    required = {"feature", "importance"}

    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required)} in {input_file}")

    out = df.sort_values("importance", ascending=False).head(top_n).copy()
    out["feature_label"] = out["feature"].map(DISPLAY_NAMES).fillna(out["feature"])

    return out.sort_values("importance", ascending=True)


def save_plot(df: pd.DataFrame, out_file: Path) -> None:
    """Save a horizontal feature-importance plot."""
    fig_height = max(4.4, 0.27 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))

    ax.barh(
        df["feature_label"],
        df["importance"],
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel("Relative importance", fontsize=8)
    ax.set_ylabel("")
    ax.set_title("Selected Species-Use Model — Feature Importance", fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot selected species-use model feature importance.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=INPUT_FILE,
        help="CSV file with feature and importance columns.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for the generated figure.",
    )
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=DATA_OUTPUT_ROOT,
        help="Directory for the generated CSV export.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Number of ranked features to include.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the plotting workflow."""
    args = parse_args()

    df = load_importance(args.input_file, args.top_n)
    out_csv = args.data_output_root / "species_feature_importance_top_features.csv"
    out_png = args.output_root / "species_feature_importance_top_features.png"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("importance", ascending=False).to_csv(out_csv, index=False)
    save_plot(df, out_png)

    print("Saved:", out_csv)
    print("Saved:", out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
