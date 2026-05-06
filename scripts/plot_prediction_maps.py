"""Plot species prediction maps."""

from riskscape.visualization.maps import MapStyle, plot_hazard_map, plot_prediction_map


RISK_STYLE = MapStyle(
    color_scale="log",
    alpha_scale=False,
    min_display_value=0.01,
    colorbar_labels=("Low", "Mod", "High", "Xtrm"),
    colorbar_quantiles=(0.0, 0.50, 0.90, 0.98, 1.0),
)
# use 0.25 as the fraction of fishing effort to consider as "hazard" for the hazard map, which combines risk and a fraction of imagined fishing effort (for 2022 = 1 vessel * 1/2 hour per day)
HAZARD_FISHING_EFFORT_FRACTION = 0.25


def main() -> int:
    """Run prediction map plots."""
    year = 2022
    model_name = "extra_trees"
    product_name = "bbal"

    print("Example:")
    print(
        "plot_prediction_map("
        "year=2022, model_name='extra_trees', product_name='bbal', "
        "value_col='risk_log_pred', species='BBAL')"
    )
    print(
        "Input: "
        "data/modeling/predictions/extra_trees/bbal/year=2022/part.parquet"
    )

    for product_name, species in [("bbal", "BBAL"), ("safs", "SAFS")]:
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
            fishing_effort_fraction=HAZARD_FISHING_EFFORT_FRACTION,
            title="Hazard",
            style=RISK_STYLE,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
