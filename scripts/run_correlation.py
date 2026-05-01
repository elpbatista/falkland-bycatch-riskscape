"""Run correlation analysis."""

from riskscape.analysis.correlation import run_correlation_analysis


RUNS = {
    "env_features": {
        "source_tables": ["environmental"],
        "join_keys": ["h3", "date"],
        "features": [
            "sst",
            "chl",
            "chl_log",
            "ssh",
            "wind_speed",
        ],
        "method": "spearman",
        "plot": True,
    },
}


def main() -> int:
    for name, cfg in RUNS.items():
        run_correlation_analysis(name, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())