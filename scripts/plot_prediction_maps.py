"""Plot species prediction maps."""

from riskscape.visualization.maps import MapStyle, plot_hazard_map, plot_prediction_map


RISK_STYLE = MapStyle(
    color_scale="log",
    alpha_scale=False,
    min_display_value=0.01,
    colorbar_labels=("Low", "Mod", "High", "Xtrm"),
    colorbar_quantiles=(0.0, 0.50, 0.90, 0.98, 1.0),
)
# 0.5 vessel-hours per H3 cell/day
HAZARD_MINIMUM_EFFORT_UNIT = 0.5


def main() -> int:
    """Run prediction map plots."""
    year = 2022
    model_name = "extra_trees"

    prediction_products = [
        ("bbal", "BBAL"),
        ("safs", "SAFS"),
        ("joint", "BBAL"),
        ("joint", "SAFS"),
    ]

    for product_name, species in prediction_products:
        print(
            "Input: "
            f"data/modeling/predictions/{model_name}/{product_name}/"
            f"year={year}/part.parquet"
        )

        plot_prediction_map(
            year=year,
            model_name=model_name,
            product_name=product_name,
            value_col="species_use_log_pred",
            species=species,
            agg="mean",
            title="Species Use",
        )

        plot_prediction_map(
            year=year,
            model_name=model_name,
            product_name=product_name,
            value_col="risk_log_pred",
            species=species,
            agg="mean",
            title="Risk",
            style=RISK_STYLE,
        )

        plot_hazard_map(
            year=year,
            model_name=model_name,
            product_name=product_name,
            species=species,
            agg="mean",
            minimum_effort_unit=HAZARD_MINIMUM_EFFORT_UNIT,
            title="Hazard",
            style=RISK_STYLE,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
