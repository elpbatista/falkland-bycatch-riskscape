# Publishing Checklist

This checklist tracks cleanup decisions for preparing the repository as a
reusable riskscape workflow/template.

## Completed

- Remove generated `outputs/` artifacts from Git and ignore future outputs.
- Remove `scripts/dev/` from Git tracking and keep development scripts local.
- Add `scripts/download_reference_data.py` for public reference-layer downloads.
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

## Credential and Configuration Policy

- Credentials must not be committed to source control.
- Local credentials should live in provider-specific credential stores or in an
  ignored `.env` file.
- `.env.example` is committed only as a template.
- `config.run.yaml` needs a design decision before publishing. It comes from an
  older notebook-reproduction workflow and may mix runtime state, date ranges,
  local execution choices, and reproducibility metadata.

## External Archive Policy

- Zenodo will be the external archive for versioned data, plot, and model-output
  bundles that should not live in Git.
- GitHub remains the source-code, workflow, and documentation repository.
- Future release bundles should include a manifest, checksums, and a short
  README describing what can be regenerated and what depends on restricted
  source data.

## Deferred Design Decisions

- Decide whether the public configuration model should use:
  - `config.yaml` as the default reusable workflow configuration,
  - `config.example.yaml` as the commented public template,
  - ignored `config.local.yaml` or `.env` overrides for local settings,
  - optional ignored `runs/<run-id>/config.yaml` snapshots for reproducibility.
- Decide whether notebooks are public tutorials, execution records, or archived
  development material.
- Decide which `data/` subfolders, plots, and model outputs are safe/useful to
  package for Zenodo.
- Review `docs/code_style_guide.md` for whether it should remain public or be
  replaced with a shorter contributor note.

## Still To Do

- Add or confirm project license.
- Add citation guidance.
- Expand `pyproject.toml` with package metadata and dependencies.
