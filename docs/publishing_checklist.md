# Publishing Checklist

This checklist tracks cleanup decisions for preparing the repository as a
reusable riskscape workflow/template.

## Completed

- Remove generated `outputs/` artifacts from Git and ignore future outputs.
- Remove `scripts/dev/` from Git tracking and keep development scripts local.
- Add `scripts/data/download_reference_data.py` for public reference-layer downloads.
- Stop tracking downloaded `reference/` geospatial files.
- Keep `reference/README.md` as the public entry point for restoring reference
  layers.
- Start `docs/script_inventory.md` for scripts that should be kept and
  documented.
- Move the GFW token convention to local `.env` / `GFW_TOKEN` and remove token
  values from committed YAML files.
- Remove duplicate `docs/authentication copy.md`.
- Confirm generated packaging metadata such as `src/riskscape.egg-info/` are
  not tracked and remain ignored.
- Rewrite the root `README.md` for the reusable workflow/template audience.
- Prune stale docs and keep only the current public documentation set.
- Add BSD-3-Clause license for code reuse with non-endorsement protection.
- Add `CITATION.cff` with provisional repository citation metadata.
- Expand `pyproject.toml` with package metadata and dependency groups.
- Retire obsolete `config.run.yaml`; use `config.yaml` plus `.env` as the
  current public configuration model.
- Simplify `config.yaml` by removing obsolete path aliases and old `layer1`
  settings, and make dataset roles/lookup behavior explicit.
- Add a GEBCO/CEDA OPeNDAP bathymetry provider and credential path; live
  download testing is currently blocked by CEDA `503 Service Unavailable`.
- Reconcile `docs/script_inventory.md` with all current top-level
  `scripts/*.py` entry points.

## Credential and Configuration Policy

- Credentials must not be committed to source control.
- Local credentials should live in provider-specific credential stores or in an
  ignored `.env` file.
- `.env.example` is committed only as a template.
- `config.yaml` is the canonical public workflow configuration.
- Notebooks are presentation, tutorial, and inspection surfaces; they are not
  the canonical pipeline orchestrator.

## External Archive Policy

- Zenodo will be the external archive for versioned data, plot, and model-output
  bundles that should not live in Git.
- GitHub remains the source-code, workflow, and documentation repository.
- Future release bundles should include a manifest, checksums, and a short
  README describing what can be regenerated and what depends on restricted
  source data.

## Deferred Design Decisions

- Decide whether a future `config.example.yaml` is useful after the public
  config settles.
- Decide whether future ignored `runs/<run-id>/config.yaml` snapshots are useful
  for reproducibility.
- Review notebooks as tutorial/presentation/inspection material before release.
- Decide which `data/` subfolders, plots, and model outputs are safe/useful to
  package for Zenodo.
- Review `docs/code_style_guide.md` for whether it should remain public or be
  replaced with a shorter contributor note.
- Decide whether to remove the `scripts/riskscape/` import shim before release.
  It is probably redundant once public scripts either run after
  `pip install -e .` or bootstrap `src/` explicitly.
- Retest `python scripts/data/download_data.py --dataset bathymetry` when CEDA's
  `dap.ceda.ac.uk` service is available.

## Still To Do
