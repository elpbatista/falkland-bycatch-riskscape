#!/usr/bin/env python3
"""Export local dashboard weekly risk CSV files.

Each output file contains one species and one ISO week with the dashboard
attribute schema:

- h3
- species_use_log_pred
- plausibility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEEKLY_DIR = (
    PROJECT_ROOT
    / "data"
    / "plot_exports"
    / "dashboard"
    / "mapbox"
    / "risk_weekly"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "plot_exports" / "dashboard" / "risk_weekly_csv"
)
SPECIES = ("BBAL", "SAFS")
COLUMNS = ["h3", "species_use_log_pred", "plausibility"]


def export_species(
    species: str,
    weekly_dir: Path,
    output_dir: Path,
    weeks: range,
) -> list[Path]:
    """Export one weekly CSV per requested ISO week for one species."""
    path = weekly_dir / f"{species}_weekly.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    table = pd.read_parquet(path)
    missing = {"h3", "iso_week", *COLUMNS} - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for week in weeks:
        week_table = table.loc[table["iso_week"] == week, COLUMNS].copy()
        if week_table.empty:
            raise ValueError(f"No rows found for {species} ISO week {week:02d}")

        week_table["h3"] = week_table["h3"].astype("uint64").astype(str)
        week_table = week_table.sort_values("h3")

        output_path = output_dir / f"{species}_w{week:02d}.csv"
        week_table.to_csv(output_path, index=False, float_format="%.6g")
        outputs.append(output_path)

    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-dir", type=Path, default=DEFAULT_WEEKLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-week", type=int, default=1)
    parser.add_argument("--end-week", type=int, default=52)
    return parser.parse_args()


def main() -> None:
    """Run the export."""
    args = parse_args()
    if args.start_week < 1 or args.end_week > 53 or args.start_week > args.end_week:
        raise ValueError("Expected 1 <= start-week <= end-week <= 53")

    weeks = range(args.start_week, args.end_week + 1)
    outputs: list[Path] = []
    for species in SPECIES:
        outputs.extend(
            export_species(
                species=species,
                weekly_dir=args.weekly_dir,
                output_dir=args.output_dir,
                weeks=weeks,
            )
        )

    print(f"Exported {len(outputs)} weekly CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
