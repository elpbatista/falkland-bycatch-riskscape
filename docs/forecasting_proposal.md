# Forecasting Proposal — Bycatch Riskscape

## Overview

Because direct bycatch observations are not currently available, forecasting is framed as:

> **Prediction of potential bycatch risk as the overlap between species-use (hazard) and fishing activity (exposure), not as observed bycatch probability.**

The system remains consistent with the current architecture:

- **Hazard** → species use (from SDM)
- **Exposure** → fishing activity
- **Risk** → hazard × exposure

---

## Core Forecasting Framework

### 1. Species-Use Forecast (Hazard)

Use forecasted or near-real-time environmental variables:

- SST  
- SSH  
- Chlorophyll (CHL)  
- Wind speed  
- Anomalies (SST, SSH, CHL, wind)  
- Gradients (SST, SSH, CHL)  
- DOY (sin/cos)  
- Static features (depth, slope, distance to coast, lat/lon encoding)

Apply trained species models:

environment forecast → species_use_pred

This produces:

> Spatial prediction of where species are likely to concentrate under expected ocean conditions.

---

### 2. Fishing Exposure Forecast

Since future fishing activity is unknown, three approaches are proposed:

#### Option A — Persistence (short-term)

Use recent observed fishing activity as forecast.

- Assumes short-term continuity  
- Suitable for near-real-time systems  

#### Option B — Climatology (seasonal)

Use historical fishing activity for the same period (month/day-of-year).

- Captures seasonal fishing patterns  
- Stable baseline forecast  

#### Option C — Model-based (future work)

Train fishing model using:

- Past fishing activity  
- Environmental variables  

- Requires additional modeling  
- Not required for current capstone  

---

### 3. Risk Forecast

Compute:

forecast_risk = species_use × fishing_activity

In current implementation:

forecast_risk_log = species_use_log_pred + fishing_activity_log_forecast

---

## Interpretation (Critical)

This is NOT a forecast of observed bycatch events.

It is:

> A forecast of potential bycatch risk based on ecological hazard (species use) and operational exposure (fishing activity).

---

## Hazard vs Exposure Separation (Conceptual Clarification)

Instead of only analyzing:

risk = hazard × exposure

explicitly retain all components:

- hazard   = species_use  
- exposure = fishing_activity  
- risk     = hazard × exposure  

This allows interpretation of *why* risk is high:

- High hazard + low exposure → ecological hotspot  
- Low hazard + high exposure → fishing-driven risk  
- High hazard + high exposure → true high-risk zone  

---

## Proposed CNN-LSTM Forecasting Extension

### Concept

- CNN → captures spatial patterns (fronts, gradients, structure)  
- LSTM → captures temporal dynamics (movement, persistence)  

Combined:

CNN-LSTM → spatiotemporal forecasting of species use

---

## Input Design

### Spatial Representation

- H3 resolution 6 (~36 km²)
- Converted to:
  - 2D raster grid (preferred), or  
  - Local spatial patches centered on each cell  

---

### Temporal Window

- Sequence length: 7–30 days  
- Recommended: 14 days  

---

## Input Tensor Structure

Per timestep:

[channels, height, width]

---

### Dynamic Features (channels)

- SST  
- SSH  
- CHL (log)  
- Wind speed  
- SST anomaly  
- SSH anomaly  
- CHL anomaly  
- Wind anomaly  
- SST gradient  
- SSH gradient  
- CHL gradient  

---

### Static Features (broadcasted across grid)

- Depth  
- Slope  
- Distance to coast  
- Lat_sin / Lat_cos  
- Lon_sin / Lon_cos  

---

## Full Input Tensor

[time_steps, channels, height, width]

Example:

[14, ~15, H, W]

---

## Model Architecture (Conceptual)

Input sequence  
→ CNN (applied per timestep)  
→ Spatial feature maps  
→ LSTM (across time)  
→ Temporal encoding  
→ Dense / Conv layer  
→ Species-use prediction map  

---

## Output

Predicted species use per cell

Then:

Risk = predicted species use × forecast fishing activity

---

## Advantages

- Captures spatial structure (fronts, gradients)  
- Captures temporal evolution (movement, persistence)  
- Avoids independent cell assumption  
- Better aligned with dynamic ocean systems  

---

## Limitations

- Requires H3 → grid transformation  
- Higher computational cost  
- More complex training and tuning  

---

## Final Positioning (for Capstone)

> The forecasting component estimates potential bycatch-risk hotspots by projecting species-use distributions under forecasted environmental conditions and combining them with expected fishing activity. These outputs represent exposure-based risk surfaces rather than direct predictions of bycatch events.

---

## Summary

- Current system → valid baseline forecasting framework  
- Short-term → persistence or climatology for fishing exposure  
- Future work → CNN-LSTM for spatiotemporal species forecasting  
- Interpretation → risk represents overlap potential, not observed bycatch  
