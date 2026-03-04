# Layer 1 — Seascapes (Physical Layer)

Strictly following **Kavanaugh et al. (2014)**

> Continuous implementation, daily, 5 km equal-area grid.  
> 2014–2023 (10-year) training dataset, with 2024–2025 (2-year) forecast period.

## Data Sources

> Resolution mismatch is expected.  
> All variables will be harmonized to 5 km grid.

```Text
data/
   raw/
      sst/
      chlorophyll/
      adt/
```

### Sea Surface Temperature (SST)

Temperature of the ocean’s uppermost layer, representing the thermal state of the surface ocean and a key indicator of water mass structure and stratification.

### Chlorophyll-a (Chl-a)

Satellite-derived proxy for phytoplankton biomass, representing the biogeochemical state and surface ocean productivity.

### Sea Surface Height / Absolute Dynamic Topography (SSH / ADT)

Height of the sea surface relative to a geoid; reflects ocean circulation, mesoscale structure (eddies, fronts), and dynamic pressure gradients.
