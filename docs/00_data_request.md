# Data Request for Bycatch Risk Modeling

![alt text](image.png)

***Figure 1. Study region and spatial framework.** H3 grid (resolution 6), study bounding box, and 50 km buffer. The map also shows the fisheries grid, the Falkland Islands Conservation Zone (FICZ), and the Falkland Islands Outer Conservation Zone (FOCZ).*

## Project Overview

This project develops a spatial–temporal model to estimate bycatch risk for seabirds and marine mammals in the Falkland Islands region. The model integrates environmental conditions, species presence, and fishing activity to identify conditions under which bycatch occurs, with potential for forecasting.

## Spatial and Temporal Scope

- Study region: Falkland Islands and surrounding marine area
  - Bounding box:
    - Longitude: -64 to -51
    - Latitude: -57 to -47
  - Buffer: 50 km

- Spatial representation:
  - H3 discrete global grid
  - Resolution: 6

- Temporal range:
  - Target period: 2014–present (flexible)
  - Temporal resolution: daily

## Data Processing and Structure

All datasets will be:

- Aggregated to the H3 grid (resolution 6)
- Aligned to daily temporal resolution
- Structured as cell × day observations
- Standardized for consistent scaling

## Data Completeness

- Complete spatial and temporal coverage is preferred but not required
- Partial datasets are acceptable if spatial and temporal references are preserved
- Data gaps will be handled during processing and modeling

## Data Requested

### Species Presence (Seabirds and Marine Mammals)

- Tracking / telemetry data (preferred)
- Observation records (if available)

Minimum fields:

- Latitude
- Longitude
- Timestamp
- Species identifier

### Fishing Activity

- Spatial–temporal fishing effort data

Preferred:

- VMS or AIS-based effort
- Gear type (if available)

Minimum fields:

- Latitude
- Longitude
- Timestamp
- Effort proxy (e.g., fishing hours or presence)

### Bycatch Events

- Observed bycatch records

Minimum fields:

- Location (lat/lon or grid)
- Timestamp
- Species (if available)

## Outcome

The project will produce spatially explicit daily bycatch risk estimates to support risk assessment and forecasting.

## Notes

I am happy to adapt to available data formats and discuss any constraints or data access conditions.

All data will be processed as described in the **Data Processing and Structure** section, resulting in aggregated and anonymized representations at the cell × day level. No raw or identifiable data will be used directly in modeling or included in any project outputs.
