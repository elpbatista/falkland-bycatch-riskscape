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

## Credential and Configuration Policy

- Credentials must not be committed to source control.
- Local credentials should live in provider-specific credential stores or in an
  ignored `.env` file.
- `.env.example` is committed only as a template.
- `config.run.yaml` needs a design decision before publishing. It comes from an
  older notebook-reproduction workflow and may mix runtime state, credentials,
  date ranges, local execution choices, and reproducibility metadata.

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
- Review `docs/authentication copy.md` and other duplicate/internal docs before
  publishing.
- Review the tracked Postman collection and docs for credential placeholders,
  outdated endpoints, and public relevance.

## Still To Do

- Remove generated packaging metadata such as `src/riskscape.egg-info/` if still
  present in Git history or the working tree.
- Add or confirm project license.
- Add citation guidance.
- Rewrite the root `README.md` around the reusable workflow/template audience.
- Expand `pyproject.toml` with package metadata and dependencies.
