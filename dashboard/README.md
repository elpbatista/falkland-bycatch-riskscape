# Falkland Bycatch Riskscape Dashboard

Static MapLibre dashboard prototype for the 2022 latent plausible risk surfaces.

Run it from the repository root so the dashboard can fetch files under `data/`:

```bash
python3 -m http.server 8765
```

Then open:

```text
http://localhost:8765/dashboard/
```

The dashboard uses:

- Esri Ocean Basemap XYZ tiles for the bathymetry/ocean background:
  `https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}`.
- `dashboard/data/falkland_islands_land.geojson` for Falkland Islands land polygons.
- `data/grids/h3_res6_falkland_islands.geojson` for H3 geometry.
- `data/plot_exports/dashboard/risk_weekly_csv/BBAL_w01.csv` through
  `BBAL_w52.csv` for BBAL weekly risk inputs.
- `data/plot_exports/dashboard/risk_weekly_csv/SAFS_w01.csv` through
  `SAFS_w52.csv` for SAFS weekly risk inputs.

Regenerate the 52-week CSV set with:

```bash
python3 scripts/dashboard/export_weekly_risk_csv.py
```

Build the publishable static site with:

```bash
python3 scripts/dashboard/build_static_dashboard.py
```

The build writes `dashboard/dist/`, including only the dashboard files and the
data required by the browser. Preview the built site from the repository root:

```bash
python3 -m http.server 8765
```

Then open:

```text
http://localhost:8765/dashboard/dist/
```

GitHub Pages deployment is configured in
`.github/workflows/dashboard-pages.yml`. The workflow publishes `dashboard/dist/`,
so rebuild and commit that folder before pushing to `main`.

Risk is computed in the browser as:

```text
latent_risk_log_pred =
  log1p(max(expm1(species_use_log_pred), 0) * (1 - c_s * (1 - plausibility)))
  + log1p(minimum_fishing_effort_unit)
```

`minimum_fishing_effort_unit` is fixed in `src/app.js`; `c_s` and the plausibility
threshold are dashboard controls. Cells below the selected plausibility threshold
remain visible as outlines only when the risk cell is filled.

Cells are filled only when the gated `species_use_log_pred` remains above the
project display cut of `0.1`.

Layer order is ocean basemap, H3 risk cells, then Falkland Islands land polygons.
