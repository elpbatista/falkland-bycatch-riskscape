"""Run grouped plotting scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import TypeAlias


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
PlotCommand: TypeAlias = str | tuple[str, ...]

PLOT_GROUPS: dict[str, list[PlotCommand]] = {
    "context": [
        "plot_study_area_map.py",
    ],
    "environmental": [
        "plot_environmental_histograms.py",
        "plot_environmental_correlation_heatmap.py",
        "plot_environmental_daily_timeseries.py",
        "plot_environmental_monthly_matrix.py",
        "plot_environmental_gradient_maps.py",
        "plot_environmental_single_date_maps.py",
    ],
    "fishing": [
        "plot_fishing_activity_map.py",
        "plot_fishing_activity_monthly_matrix.py",
        "plot_fishing_activity_monthly_timeseries.py",
    ],
    "species": [
        "plot_species_presence_maps.py",
        "plot_species_feature_importance.py",
        "plot_species_partial_dependence.py",
        "plot_species_use_observed_vs_predicted.py",
    ],
    "predictions": [
        "plot_prediction_maps.py",
        "plot_prediction_latent_risk_monthly_matrix.py",
        "plot_plausibility_maps.py",
        "plot_plausibility_monthly_climatology.py",
        "plot_plausibility_yearly_timeseries.py",
        "plot_plausibility_gate_sensitivity.py",
    ],
    "seascapes": [
        "plot_bayesian_gmm_component_maps.py",
        ("plot_bayesian_gmm_component_maps.py", "--monthly"),
        "plot_seascapes_maps.py",
        "plot_seascape_species_use_monthly_matrix.py",
        "plot_seascape_prediction_maps.py",
        (
            "plot_seascape_prediction_maps.py",
            "--monthly-matrix",
            "--matrix-values",
            "species_use_log_pred",
        ),
        (
            "plot_prediction_latent_risk_monthly_matrix.py",
            "--model-name",
            "seascape_som_15x15_hierarchical_k30",
            "--product-name",
            "joint",
            "--year",
            "2022",
            "--agg",
            "non_zero_mean",
            "--color-bin-source",
            "monthly_species",
            "--color-quantiles",
            "0",
            "0.55",
            "0.80",
            "0.95",
            "1.0",
        ),
    ],
    "weekly": [
        "plot_weekly_operator_latent_risk.py",
        "plot_weekly_operator_fisheries_grid_example.py",
        "plot_weekly_latent_risk_with_jigger_activity.py",
        "plot_weekly_gear_aware_risk_examples.py",
    ],
    "gear": [
        "plot_set_longline_bbal_risk_example.py",
        "plot_weekly_gear_aware_risk_examples.py",
    ],
    "videos": [
        ("plot_weekly_operator_latent_risk.py", "--make-animation"),
    ],
    "diagnostics": [
        "plot_relationship_diagnostics.py",
    ],
}


def all_groups() -> list[str]:
    """Return all concrete plot groups in execution order."""
    return list(PLOT_GROUPS)


def commands_for_groups(groups: list[str]) -> list[tuple[str, ...]]:
    """Return de-duplicated commands for selected groups."""
    selected_groups = all_groups() if "all" in groups else groups
    commands: list[tuple[str, ...]] = []

    for group in selected_groups:
        for command in PLOT_GROUPS[group]:
            if isinstance(command, str):
                commands.append((command,))
            else:
                commands.append(command)

    return list(dict.fromkeys(commands))


def script_env() -> dict[str, str]:
    """Return environment with project src on PYTHONPATH."""
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{current}" if current else str(SRC_DIR)
    )

    return env


def run_command(command: tuple[str, ...], env: dict[str, str]) -> None:
    """Run one plotting command."""
    script_name = command[0]
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Plot script not found: {script_path}")

    command_text = " ".join([script_name, *command[1:]])
    print(f"\n=== Running {command_text} ===", flush=True)
    subprocess.run(
        [sys.executable, str(script_path), *command[1:]],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run grouped riskscape plotting scripts.",
    )
    parser.add_argument(
        "--group",
        nargs="+",
        default=["context"],
        choices=[*PLOT_GROUPS, "all"],
        help="Plot group(s) to run. Defaults to context.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected scripts without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later scripts if one plot script fails.",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected plot groups."""
    args = parse_args()
    commands = commands_for_groups(args.group)

    if args.list:
        for command in commands:
            print(" ".join(command))
        return 0

    env = script_env()
    failures: list[str] = []

    for command in commands:
        try:
            run_command(command, env)
        except subprocess.CalledProcessError:
            failures.append(" ".join(command))
            if not args.continue_on_error:
                raise

    if failures:
        print("\nFailed plot scripts:")
        for script_name in failures:
            print(f"- {script_name}")
        return 1

    print("\nSelected plot scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
