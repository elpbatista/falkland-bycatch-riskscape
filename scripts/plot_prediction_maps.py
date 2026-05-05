"""Plot prediction maps."""

from riskscape.visualization.maps import MapStyle, plot_prediction_map


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
        )

        plot_prediction_map(
            year=year,
            model_name=model_name,
            product_name=product_name,
            value_col="risk_log_pred",
            species=species,
            agg="mean",
        )

    plot_prediction_map(
        year=year,
        model_name=model_name,
        product_name="bbal",
        value_col="fishing_activity",
        agg="mean",
        style=MapStyle(
            color_quantile=None,
            color_scale="log",
            alpha_scale=False,
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
