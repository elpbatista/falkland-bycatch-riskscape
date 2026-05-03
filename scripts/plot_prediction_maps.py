"""Plot prediction maps."""

from riskscape.visualization.maps import plot_prediction_map


def main() -> int:
    """Run prediction map plots."""
    year = 2022

    for species in ["BBAL", "SAFS"]:
        plot_prediction_map(
            year=year,
            value_col="species_use_log_pred",
            species=species,
            agg="mean",
        )

        plot_prediction_map(
            year=year,
            value_col="risk_log_pred",
            species=species,
            agg="mean",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())