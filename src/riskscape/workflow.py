"""High-level workflow orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable

from riskscape.downloads.pipeline import download_configured_datasets
from riskscape.downloads.reference_data import download_reference_datasets
from riskscape.features import (
    build_environmental_features,
    build_fishing_effort_features,
    build_species_presence_features,
    build_static_features,
    process_environmental,
)
from riskscape.grid import build_h3_grid, convert_grid_to_uint64
from riskscape.indices import (
    build_h3_lookup,
    build_neighbor_index_table,
    build_neighbor_table,
    build_seasonal_lookup,
)
from riskscape.logs import stage_context
from riskscape.model import (
    build_model_datasets,
    evaluate_models,
    predict_models,
    train_models,
)
from riskscape.qa import run_feature_qa_summary, run_quick_validation


logger = logging.getLogger(__name__)

StageFn = Callable[[], None | int | list | object]


def run_step(name: str, func: StageFn) -> None:
    """Run one workflow step inside a logging stage."""
    logger.info("Running workflow step: %s", name)

    with stage_context(name):
        result = func()

    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"Workflow step failed: {name}")


def restore_reference_data() -> None:
    """Restore public reference datasets."""
    status = download_reference_datasets()
    if status != 0:
        raise RuntimeError("Reference data restoration did not complete")


def download_source_data() -> None:
    """Download configured provider-backed source datasets."""
    status = download_configured_datasets()
    if status != 0:
        raise RuntimeError("Source data download did not complete")


def build_spatial_framework() -> None:
    """Build grid, static features, and spatial lookup tables."""
    run_step("build_grid", build_h3_grid)
    run_step("convert_grid_to_uint64", convert_grid_to_uint64)
    run_step("build_h3_lookup", build_h3_lookup)
    run_step("build_neighbor_table", build_neighbor_table)
    run_step("build_neighbor_index_table", build_neighbor_index_table)
    run_step("build_seasonal_lookup", build_seasonal_lookup)
    run_step("build_static_features", build_static_features)


def build_features() -> None:
    """Build primary and derived feature tables."""
    run_step("build_environmental_features", build_environmental_features)
    run_step("build_fishing_effort_features", build_fishing_effort_features)
    run_step("build_species_presence_features", build_species_presence_features)
    run_step("build_derived_features", process_environmental)


def build_model_tables() -> None:
    """Build model-ready datasets."""
    run_step("build_model_datasets", build_model_datasets)


def train_predict_evaluate() -> None:
    """Train models, generate predictions, and evaluate results."""
    run_step("train_models", train_models)
    run_step("predict_models", predict_models)
    run_step("evaluate_models", evaluate_models)


def run_checks() -> None:
    """Run lightweight QA and validation checks."""
    run_step("feature_qa_summary", run_feature_qa_summary)
    run_step("quick_validation", run_quick_validation)


STAGES: dict[str, StageFn] = {
    "reference": restore_reference_data,
    "downloads": download_source_data,
    "spatial": build_spatial_framework,
    "features": build_features,
    "model-tables": build_model_tables,
    "modeling": train_predict_evaluate,
    "checks": run_checks,
}

DEFAULT_SEQUENCE = [
    "spatial",
    "features",
    "model-tables",
    "modeling",
    "checks",
]

FULL_SEQUENCE = [
    "reference",
    "downloads",
    *DEFAULT_SEQUENCE,
]


def run_workflow(stage: str) -> None:
    """Run a named workflow stage."""
    if stage == "all":
        sequence = DEFAULT_SEQUENCE
    elif stage == "all-with-downloads":
        sequence = FULL_SEQUENCE
    else:
        sequence = [stage]

    for stage_name in sequence:
        run_step(stage_name, STAGES[stage_name])
