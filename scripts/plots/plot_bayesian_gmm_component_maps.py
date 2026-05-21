"""Plot dominant Bayesian/GMM environmental component assignments."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path
import duckdb
import matplotlib

matplotlib.use("Agg")

import pandas as pd

from riskscape.config import paths
from riskscape.visualization.component_maps import save_monthly_categorical_matrix
from riskscape.visualization.component_maps import save_single_categorical_map


YEAR = 2022
MODEL_NAME = "bayesian_gmm_k30"
PRODUCT_NAME = "joint"
INPUT_ROOT = (
    paths["data"]
    / "modeling"
    / "environmental_regimes"
)
OUTPUT_ROOT = paths["plots"] / "plausibility"

COMPONENT_COLORS = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#e15759",
    3: "#76b7b2",
    4: "#59a14f",
    5: "#edc948",
    6: "#b07aa1",
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ac",
}

def model_label(model_name: str) -> str:
    """Return a compact display label for the component model."""
    return model_name.replace("_", " ").upper().replace("BAYESIAN GMM", "Bayesian/GMM")

def component_path(year: int, input_root: Path) -> Path:
    """Return component-assignment partition path for a year."""
    return input_root / f"year={year}" / "part.parquet"


def dominant_components(
    year: int,
    model_name: str,
    product_name: str,
    input_root: Path,
) -> pd.DataFrame:
    """Return dominant component per H3 cell from environmental features."""
    _ = (model_name, product_name)
    component_file = component_path(year, input_root)

    if not component_file.exists():
        raise FileNotFoundError(
            f"Component assignment partition not found: {component_file}"
        )

    query = """
        WITH environmental_rows AS (
            SELECT DISTINCT
                CAST(h3 AS UBIGINT) AS h3,
                date,
                bayesian_gmm_k30_component AS component,
                bayesian_gmm_k30_component_probability AS component_probability,
                bayesian_gmm_k30_component_entropy AS component_entropy
            FROM read_parquet(?)
        ),
        counts AS (
            SELECT
                h3,
                component,
                count(*) AS component_days,
                avg(component_probability) AS mean_component_probability,
                avg(component_entropy) AS mean_component_entropy
            FROM environmental_rows
            GROUP BY h3, component
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY h3
                    ORDER BY component_days DESC, mean_component_probability DESC, component
                ) AS rank
            FROM counts
        )
        SELECT
            h3,
            CAST(component AS INTEGER) AS dominant_component,
            component_days,
            mean_component_probability,
            mean_component_entropy
        FROM ranked
        WHERE rank = 1
        ORDER BY h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(
            query,
            [str(component_file)],
        ).df()


def monthly_dominant_components(
    year: int,
    model_name: str,
    product_name: str,
    input_root: Path,
) -> pd.DataFrame:
    """Return dominant component by month/H3 from environmental features."""
    _ = (model_name, product_name)
    component_file = component_path(year, input_root)

    if not component_file.exists():
        raise FileNotFoundError(
            f"Component assignment partition not found: {component_file}"
        )

    query = """
        WITH environmental_rows AS (
            SELECT DISTINCT
                CAST(h3 AS UBIGINT) AS h3,
                date,
                bayesian_gmm_k30_component AS component,
                bayesian_gmm_k30_component_probability AS component_probability,
                bayesian_gmm_k30_component_entropy AS component_entropy
            FROM read_parquet(?)
        ),
        counts AS (
            SELECT
                h3,
                CAST(month(date) AS INTEGER) AS month,
                component,
                count(*) AS component_days,
                avg(component_probability) AS mean_component_probability,
                avg(component_entropy) AS mean_component_entropy
            FROM environmental_rows
            GROUP BY h3, month, component
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY h3, month
                    ORDER BY component_days DESC, mean_component_probability DESC, component
                ) AS rank
            FROM counts
        )
        SELECT
            h3,
            month,
            CAST(component AS INTEGER) AS dominant_component,
            component_days,
            mean_component_probability,
            mean_component_entropy
        FROM ranked
        WHERE rank = 1
        ORDER BY month, h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(
            query,
            [str(component_file)],
        ).df()


def save_component_map(
    summary: pd.DataFrame,
    year: int,
    model_name: str,
    out_file: Path,
) -> None:
    """Save one dominant-component map."""
    save_single_categorical_map(
        values=summary,
        value_col="dominant_component",
        colorbar_label="Component",
        title=f"Dominant {model_label(model_name)} Environmental Components — {year}",
        out_file=out_file,
        base_colors=COMPONENT_COLORS,
    )


def save_monthly_component_matrix(
    monthly: pd.DataFrame,
    year: int,
    model_name: str,
    out_file: Path,
) -> None:
    """Save a 12-panel monthly dominant-component matrix."""
    save_monthly_categorical_matrix(
        monthly=monthly,
        value_col="dominant_component",
        colorbar_label="Component",
        title=f"Monthly Dominant {model_label(model_name)} Environmental Components — {year}",
        out_file=out_file,
        base_colors=COMPONENT_COLORS,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot dominant Bayesian/GMM environmental components.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--model", default=MODEL_NAME, help=argparse.SUPPRESS)
    parser.add_argument("--product", default=PRODUCT_NAME, help=argparse.SUPPRESS)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help="Directory containing component assignment year=*/part.parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated component map figures.",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Also generate 12-panel monthly component matrices.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the component mapping workflow."""
    args = parse_args()

    summary = dominant_components(
        year=args.year,
        model_name=args.model,
        product_name=args.product,
        input_root=args.input_root,
    )
    figure_file = args.output_root / f"dominant_{args.model}_components_{args.year}.png"
    save_component_map(
        summary=summary,
        year=args.year,
        model_name=args.model,
        out_file=figure_file,
    )
    print("Saved:", figure_file)

    if args.monthly:
        monthly = monthly_dominant_components(
            year=args.year,
            model_name=args.model,
            product_name=args.product,
            input_root=args.input_root,
        )
        figure_file = (
            args.output_root
            / f"monthly_dominant_{args.model}_components_{args.year}.png"
        )
        save_monthly_component_matrix(
            monthly=monthly,
            year=args.year,
            model_name=args.model,
            out_file=figure_file,
        )
        print("Saved:", figure_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
