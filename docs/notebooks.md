# Notebook Inventory

Notebooks are supporting material for workflow demonstrations and inspection.
They are not the canonical pipeline orchestrator. Public execution should live
in `scripts/` and reusable logic should live in `src/riskscape/`.

## Public Notebook Roles

- `workflow-demo`: polished walkthrough of a major workflow stage.
- `inspection`: diagnostic or validation notebook for checking intermediate
  products.
- `review`: useful material, but stale or not ready for public documentation.
- `defer`: exploratory or private scratch material that should not be promoted
  without deliberate cleanup.

Before release, public notebooks should have markdown context, no credentials,
no large stale embedded outputs, and should not depend on obsolete orchestration
files.

## Current Notebooks

- `workflow-demo` `notebooks/00_project_overview.ipynb`: Introduces the
  repository structure, workflow roles, generated-data policy, and the
  separation between species use, fishing exposure, plausibility, realized risk,
  and latent risk.
- `workflow-demo` `notebooks/01_reference_layers_and_study_area.ipynb`:
  Documents the study-area frame, reference layers, fisheries grid, conservation
  zones, Natural Earth layers, and H3 spatial framework.
- `inspection` `notebooks/02_data_sources_and_inputs.ipynb`: Summarizes input
  dataset families, provider roles, credentials, and redistribution cautions
  from `config.yaml` and the workflow documentation.
- `workflow-demo` `notebooks/03_feature_engineering.ipynb`: Documents the
  engineering logic and equations for raster-to-H3 lookups, wind speed,
  chlorophyll transformation, seasonal encodings, H3-neighbor gradients,
  anomalies, fishing exposure, and telemetry-derived species-use support.
- `workflow-demo` `notebooks/04_model_design_and_decisions.ipynb`: Explains the
  response variable, joint-species design, balancing, validation design, Extra
  Trees learner selection, and Bayesian/GMM plausibility role.
- `workflow-demo` `notebooks/05_predictions_and_risk_interpretation.ipynb`:
  Explains species-use predictions, environmental plausibility, realized risk,
  latent risk, and the tuned seascape latent-risk matrix scale.
- `workflow-demo` `notebooks/06_operational_outputs.ipynb`: Documents weekly
  latent-risk climatology, fisheries-grid aggregation, gear-aware examples,
  vessel-activity overlays, and animation-oriented outputs as operational
  planning and communication products.
- `inspection` `notebooks/07_quality_checks_and_diagnostics.ipynb`: Summarizes
  QA checks, validation diagnostics, known limitations, and diagnostic script
  entry points.

The previous notebooks `00_pipeline_execution.ipynb`,
`01_data_sources_and_scope.ipynb`, and `51_study_area.ipynb` were replaced
because they were code-only and depended on the obsolete `config.run.yaml`
notebook-orchestration pattern.
