#!/usr/bin/env python3
"""Build the static dashboard folder for GitHub Pages."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DEFAULT_OUTPUT_DIR = DASHBOARD_DIR / "dist"
GRID_SOURCE = PROJECT_ROOT / "data" / "grids" / "h3_res6_falkland_islands.geojson"
RISK_CSV_SOURCE = (
    PROJECT_ROOT / "data" / "plot_exports" / "dashboard" / "risk_weekly_csv"
)
APP_PATH_REPLACEMENTS = {
    '../data/grids/h3_res6_falkland_islands.geojson': (
        './data/grids/h3_res6_falkland_islands.geojson'
    ),
    '../data/plot_exports/dashboard/risk_weekly_csv': './data/risk_weekly_csv',
}


def copy_dashboard_shell(output_dir: Path) -> None:
    """Copy dashboard HTML, CSS, JS, and local dashboard data."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    shutil.copytree(
        DASHBOARD_DIR,
        output_dir,
        ignore=shutil.ignore_patterns("dist", "README.md"),
    )
    (output_dir / ".nojekyll").write_text("")


def copy_data(output_dir: Path) -> None:
    """Copy only the data files required by the browser dashboard."""
    grid_dir = output_dir / "data" / "grids"
    risk_dir = output_dir / "data" / "risk_weekly_csv"
    grid_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)

    if not GRID_SOURCE.exists():
        raise FileNotFoundError(GRID_SOURCE)
    if not RISK_CSV_SOURCE.exists():
        raise FileNotFoundError(RISK_CSV_SOURCE)

    shutil.copy2(GRID_SOURCE, grid_dir / GRID_SOURCE.name)

    csv_paths = sorted(RISK_CSV_SOURCE.glob("*.csv"))
    if len(csv_paths) != 104:
        raise ValueError(f"Expected 104 weekly CSV files, found {len(csv_paths)}")
    for path in csv_paths:
        shutil.copy2(path, risk_dir / path.name)


def rewrite_app_paths(output_dir: Path) -> None:
    """Rewrite source app paths for the self-contained published folder."""
    app_path = output_dir / "src" / "app.js"
    text = app_path.read_text()
    for old, new in APP_PATH_REPLACEMENTS.items():
        text = text.replace(old, new)
    app_path.write_text(text)


def build(output_dir: Path) -> None:
    """Build the static dashboard output folder."""
    copy_dashboard_shell(output_dir)
    copy_data(output_dir)
    rewrite_app_paths(output_dir)

    size_mb = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    print(f"Built {output_dir} ({size_mb / 1024 / 1024:.1f} MB)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the build."""
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    build(output_dir)


if __name__ == "__main__":
    main()
