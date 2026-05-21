"""Export raw GFW fishing effort by H3/date/gear/flag for examples."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.utils.dates import normalize_date_column


YEARS = "2022"
INPUT_ROOT = paths["data"] / "raw" / "gfw"
OUTPUT_ROOT = paths["data"] / "plot_exports" / "fishing_activity"
OUTPUT_COLUMNS = [
    "h3",
    "date",
    "gear_type",
    "flag",
    "fishing_hours",
    "vessel_count",
]


def fishing_effort_path(year: int) -> Path:
    """Return the raw fishing-effort partition path for one year."""
    return INPUT_ROOT / f"year={year}" / "fishing_effort.parquet"


def available_years() -> list[int]:
    """Return available raw fishing-effort years."""
    years: list[int] = []

    for year_dir in sorted(INPUT_ROOT.glob("year=*")):
        years.append(int(year_dir.name.split("=", maxsplit=1)[1]))

    if not years:
        raise FileNotFoundError(f"No raw fishing-effort partitions found: {INPUT_ROOT}")

    return years


def parse_years(years: str) -> list[int]:
    """Parse all, a single year, ranges, or comma-separated years."""
    if years.lower() == "all":
        return available_years()

    parsed: set[int] = set()
    for part in years.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            parsed.update(range(int(start_text), int(end_text) + 1))
        else:
            parsed.add(int(item))

    if not parsed:
        raise ValueError("No years selected")

    return sorted(parsed)


def year_label(years: list[int]) -> str:
    """Return display-safe selected-year text."""
    if len(years) == 1:
        return str(years[0])
    if years == list(range(min(years), max(years) + 1)):
        return f"{min(years)}-{max(years)}"
    return "_".join(str(year) for year in years)


def normalize_category(series: pd.Series, missing_value: str) -> pd.Series:
    """Return clean string categories for grouping."""
    out = series.fillna("").astype(str).str.strip()
    return out.mask(out.eq("") | out.str.upper().eq("NA"), missing_value)


def load_year(year: int) -> pd.DataFrame:
    """Load one raw fishing-effort year."""
    path = fishing_effort_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Raw fishing-effort file not found: {path}")

    return pd.read_parquet(
        path,
        columns=["date", "hours", "lat", "lon", "gear_type", "flag", "vessel_id"],
    )


def aggregate_year(
    year: int,
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Aggregate one raw GFW year to H3/date/gear/flag."""
    df = load_year(year)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["gear_type"] = normalize_category(df["gear_type"], "UNKNOWN_GEAR")
    df["flag"] = normalize_category(df["flag"], "UNKNOWN_FLAG")

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        grid[["h3", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = (
        joined.groupby(["h3", "date", "gear_type", "flag"], as_index=False)
        .agg(
            fishing_hours=("hours", "sum"),
            vessel_count=("vessel_id", "nunique"),
        )
        .sort_values(["date", "h3", "gear_type", "flag"])
        .reset_index(drop=True)
    )
    out["h3"] = out["h3"].astype("uint64")
    out = normalize_date_column(out)
    out["gear_type"] = out["gear_type"].astype("string")
    out["flag"] = out["flag"].astype("string")
    out["fishing_hours"] = out["fishing_hours"].astype("float32")
    out["vessel_count"] = out["vessel_count"].astype("uint16")

    return out[OUTPUT_COLUMNS]


def export_gear_flag_table(
    years: list[int],
    output_root: Path,
) -> Path:
    """Export selected years as one gear/flag example table."""
    grid = load_grid(uint64=True)
    frames = [aggregate_year(year, grid) for year in years]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "h3", "gear_type", "flag"]).reset_index(drop=True)

    out_file = output_root / f"fishing_effort_by_gear_flag_{year_label(years)}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False, compression="zstd")

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export an example H3/date/gear/flag fishing-effort table from raw GFW."
        ),
    )
    parser.add_argument(
        "--years",
        default=YEARS,
        help="Use 'all', one year, a range like 2014-2023, or a comma list.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for the exported example table.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the gear/flag export."""
    args = parse_args()
    years = parse_years(args.years)
    out_file = export_gear_flag_table(years, args.output_root)
    print("Saved:", out_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
