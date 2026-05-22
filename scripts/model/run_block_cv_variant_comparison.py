"""Run BlockCV validation variants and consolidate report-ready metrics.

The selected production path remains the K=15 environmental seascape split.
This script is for manuscript support: it trains comparable validation variants
with the existing ``riskscape.model.block_cv_train`` entry point and writes a
single comparison table for discussion.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from riskscape.config import paths


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
METRICS_DIR = paths["data"] / "modeling" / "metrics"
DEFAULT_MODELS = "extra_trees"
DEFAULT_TEST_FRACTION = 0.12


@dataclass(frozen=True)
class Variant:
    """One validation design to run through block_cv_train."""

    name: str
    split: str
    description: str
    cv_folds: int = 1
    block_resolution: int = 4
    buffer_rings: int = 1
    seascape_table: str = "environmental_regimes"
    seascape_column: str = "seascape"
    component_table: str = "environmental_regimes"

    @property
    def run_label(self) -> str:
        """Return a metrics-safe label for this variant."""
        return f"variant_{self.name}"


VARIANTS: dict[str, Variant] = {
    "random12": Variant(
        name="random12",
        split="random",
        description="Row-level random 12% holdout baseline.",
    ),
    "spatial_h3r4": Variant(
        name="spatial_h3r4",
        split="spatial",
        description="Spatial H3 parent-block holdout at resolution 4.",
    ),
    "buffered_h3r4": Variant(
        name="buffered_h3r4",
        split="buffered",
        description=(
            "Spatial H3 parent-block holdout with one-ring buffer removed "
            "from training."
        ),
    ),
    "gmm_k30": Variant(
        name="gmm_k30",
        split="environmental_gmm",
        description="Bayesian/Gaussian mixture environmental-component holdout.",
    ),
    "gmm_k30_5fold": Variant(
        name="gmm_k30_5fold",
        split="environmental_gmm",
        cv_folds=5,
        description="Bayesian/Gaussian mixture environmental-component grouped 5-fold CV.",
    ),
    "som_k30": Variant(
        name="som_k30",
        split="environmental_seascape",
        description="SOM-hierarchical K=30 environmental seascape holdout.",
    ),
    "som_k30_5fold": Variant(
        name="som_k30_5fold",
        split="environmental_seascape",
        cv_folds=5,
        description="SOM-hierarchical K=30 environmental seascape grouped 5-fold CV.",
    ),
}

VARIANT_GROUPS: dict[str, tuple[str, ...]] = {
    "selected": ("som_k30_5fold",),
    "core": ("random12", "spatial_h3r4", "gmm_k30", "som_k30_5fold"),
    "som": ("som_k30",),
    "som_5fold": ("som_k30_5fold",),
    "som_all": ("som_k30", "som_k30_5fold"),
    "all": tuple(VARIANTS),
}


def expand_variants(values: list[str]) -> list[Variant]:
    """Resolve variant names and aliases preserving first occurrence order."""
    names: list[str] = []

    for value in values:
        if value in VARIANT_GROUPS:
            names.extend(VARIANT_GROUPS[value])
        elif value in VARIANTS:
            names.append(value)
        else:
            valid = sorted(set(VARIANTS) | set(VARIANT_GROUPS))
            raise ValueError(f"Unknown variant '{value}'. Valid values: {valid}")

    seen: set[str] = set()
    out: list[Variant] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(VARIANTS[name])

    return out


def command_for_variant(
    variant: Variant,
    models: str,
    model_type: str,
    balance: str,
    test_fraction: float,
    diagnostics_only: bool,
) -> list[str]:
    """Return a block_cv_train command for one validation variant."""
    cmd = [
        sys.executable,
        "-m",
        "riskscape.model.block_cv_train",
        "--split",
        variant.split,
        "--model-type",
        model_type,
        "--models",
        models,
        "--test-fraction",
        str(test_fraction),
        "--block-resolution",
        str(variant.block_resolution),
        "--buffer-rings",
        str(variant.buffer_rings),
        "--cv-folds",
        str(variant.cv_folds),
        "--seascape-table",
        variant.seascape_table,
        "--seascape-column",
        variant.seascape_column,
        "--component-table",
        variant.component_table,
        "--balance",
        balance,
        "--run-label",
        variant.run_label,
    ]

    if diagnostics_only:
        cmd.append("--diagnostics-only")

    return cmd


def run_variant(
    variant: Variant,
    models: str,
    model_type: str,
    balance: str,
    test_fraction: float,
    diagnostics_only: bool,
    dry_run: bool,
) -> None:
    """Run one validation variant."""
    cmd = command_for_variant(
        variant=variant,
        models=models,
        model_type=model_type,
        balance=balance,
        test_fraction=test_fraction,
        diagnostics_only=diagnostics_only,
    )
    print("\n#", variant.name)
    print("#", variant.description)
    print(" ".join(cmd))

    if dry_run:
        return

    env = os.environ.copy()
    pythonpath = str(SRC_ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def metrics_path(variant: Variant) -> Path:
    """Return the expected metrics path for a variant."""
    return (
        METRICS_DIR
        / f"species_model_{variant.split}_{variant.run_label}_block_cv_metrics.csv"
    )


def cv_summary_path(variant: Variant) -> Path:
    """Return the expected CV summary path for a variant."""
    return (
        METRICS_DIR
        / f"species_model_{variant.split}_{variant.run_label}_block_cv_summary.csv"
    )


def load_variant_results(variant: Variant) -> pd.DataFrame:
    """Load one variant's metrics or grouped-CV summary."""
    if variant.cv_folds > 1 and cv_summary_path(variant).exists():
        df = pd.read_csv(cv_summary_path(variant))
        df.insert(0, "source_file", cv_summary_path(variant).name)
    else:
        df = pd.read_csv(metrics_path(variant))
        df.insert(0, "source_file", metrics_path(variant).name)

    df.insert(0, "validation_variant", variant.name)
    df.insert(1, "validation_description", variant.description)
    df.insert(2, "requested_cv_folds", variant.cv_folds)
    return df


def summarize_results(variants: list[Variant], out_prefix: str) -> Path:
    """Write consolidated CSV and Markdown comparison tables."""
    frames = []
    missing = []

    for variant in variants:
        path = cv_summary_path(variant) if variant.cv_folds > 1 else metrics_path(variant)
        if not path.exists():
            missing.append(path)
            continue
        frames.append(load_variant_results(variant))

    if missing:
        print("\nMissing metric files:")
        for path in missing:
            print(" -", path)

    if not frames:
        raise FileNotFoundError("No variant metric files found to summarize")

    out = pd.concat(frames, ignore_index=True, sort=False)
    out_file = METRICS_DIR / f"{out_prefix}.csv"
    md_file = METRICS_DIR / f"{out_prefix}.md"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)

    display_cols = [
        col
        for col in [
            "validation_variant",
            "validation_description",
            "model",
            "model_type",
            "species",
            "split",
            "cv_folds",
            "r2",
            "r2_mean",
            "r2_std",
            "rmse",
            "rmse_mean",
            "mae",
            "mae_mean",
            "r2_log",
            "r2_log_mean",
            "rmse_log",
            "rmse_log_mean",
            "mae_log",
            "mae_log_mean",
            "train_rows",
            "test_rows",
            "actual_test_fraction_mean",
            "excluded_buffer_rows",
        ]
        if col in out.columns
    ]
    try:
        out[display_cols].to_markdown(md_file, index=False)
    except ImportError:
        md_file.write_text(
            out[display_cols].to_csv(index=False),
            encoding="utf-8",
        )
    print("\nSaved comparison CSV:", out_file)
    print("Saved comparison Markdown:", md_file)
    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["core"],
        help=(
            "Variant names or aliases. Aliases: selected, core, all. "
            f"Variants: {', '.join(VARIANTS)}."
        ),
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--model-type",
        choices=("joint", "single", "both"),
        default="joint",
    )
    parser.add_argument(
        "--balance",
        choices=("before", "after", "none"),
        default="before",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
    )
    parser.add_argument(
        "--out-prefix",
        default="species_model_block_cv_variant_comparison",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only consolidate existing metric files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running or summarizing.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Create diagnostics for each variant without fitting models.",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected validation variants and summarize their metrics."""
    args = parse_args()
    variants = expand_variants(args.variants)

    if args.model_type != "joint" and any(variant.cv_folds > 1 for variant in variants):
        raise ValueError("Grouped CV variants currently support --model-type joint only")

    if not args.skip_run:
        for variant in variants:
            run_variant(
                variant=variant,
                models=args.models,
                model_type=args.model_type,
                balance=args.balance,
                test_fraction=args.test_fraction,
                diagnostics_only=args.diagnostics_only,
                dry_run=args.dry_run,
            )

    if args.dry_run:
        return 0

    if not args.diagnostics_only:
        summarize_results(variants, args.out_prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
