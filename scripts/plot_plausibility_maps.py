"""Plot Bayesian GMM environmental plausibility maps."""

from __future__ import annotations

from riskscape.visualization.maps import (
    MapStyle,
    plausibility_path,
    plot_plausibility_map,
)


YEAR = 2022
MODEL_NAME = "bayesian_gmm"
VALUE_COL = "plausibility"
AGG = "non_zero_median"
CONFIDENCE_THRESHOLD = 0.1

PLAUSIBILITY_STYLE = MapStyle(
    title="Plausibility",
    cmap="viridis",
    color_quantile=0.99,
    show_reference_map=False,
)


def main() -> int:
    """Run plausibility map plots."""
    plot_products = [
        ("bbal", "BBAL"),
        ("safs", "SAFS"),
        ("joint", "BBAL"),
        ("joint", "SAFS"),
    ]

    for product_name, species in plot_products:
        input_path = plausibility_path(
            year=YEAR,
            model_name=MODEL_NAME,
            product_name=product_name,
        )
        print(f"Input: {input_path}")

        out_file = plot_plausibility_map(
            year=YEAR,
            model_name=MODEL_NAME,
            product_name=product_name,
            species=species,
            value_col=VALUE_COL,
            agg=AGG,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            style=PLAUSIBILITY_STYLE,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
