# Conceptual Architecture of Bycatch Riskscape

## Architecture

The bycatch riskscape framework is structured into a layered architecture (Figure 1):

```text
┌─────────────────────────────────────────────────────────────┐
│                        Layer 0                              │
│               Data & Spatial Reference Framework            │
│                                                             │
│   Raw Environmental Datasets:                               │
│     • MUR SST (PO.DAAC)                                     │
│     • Chlorophyll-a (Copernicus Marine)                     │
│     • SSH / ADT (Copernicus Marine)                         │
│                                                             │
│   Spatial Context:                                          │
│     • H3 Grid (Resolution 6, Falkland Islands region)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Layer 1                              │
│        Continuous Physical Seascape State (H3 indexed)      │
│                                                             │
│   Daily ocean state per H3 cell (10-year time series)       │
│   • SST                                                     │
│   • Chlorophyll-a                                           │
│   • SSH                                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Layer 2                              │
│              Derived Dynamic Features                       │
│                                                             │
│   • Thermal fronts (∇SST)                                   │
│   • Productivity fronts (∇Chl)                              │
│   • Eddy structure (∇SSH)                                   │
│   • SST anomalies                                           │
│   • (Planned) Wind persistence                              │
│   • (Planned) Wind × Front interaction terms                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Layer 3                              │
│            Biological & Fishing Activity Layer              │
│                                                             │
│   • Species movement (telemetry)                            │
│   • Fishing effort (AIS / VMS)                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Layer 4                              │
│                   Bycatch Risk Model                        │
│                                                             │
│   • Spatiotemporal overlap                                  │
│   • Dynamic risk surfaces                                   │
│   • Forecasting capability                                  │
└─────────────────────────────────────────────────────────────┘
```

### Figure 1. Conceptual architecture of the dynamic bycatch riskscape framework

*Layer 0 defines the data and spatial reference framework, including raw environmental datasets (MUR SST, chlorophyll-a, and sea surface height) and the H3 grid used as a harmonized spatial scaffold for the Falkland Islands region. Layer 1 constructs a continuous, H3-indexed physical seascape state at daily resolution across a 10-year period. Layer 2 derives dynamic oceanographic features from this state, including thermal and productivity fronts, eddy structure, and anomalies; wind-derived metrics will be incorporated at this stage as external forcing. Layer 3 integrates biological movement data and fishing effort to quantify spatiotemporal overlap. Layer 4 produces dynamic bycatch risk surfaces and enables forecasting. The layered structure separates raw data ingestion, physical state representation, ecological feature engineering, and predictive modeling, ensuring modularity, interpretability, and scalability.*

## Bycatch Riskscape Framework

### Layer 0 — Data & Spatial Framework *(Completed!)*

- [x] SST, Chlorophyll-a, SSH datasets
- [x] H3 grid (Resolution 6, Falkland Islands)
- [x] Spatial and temporal consistency

### Layer 1 — Continuous Physical Seascape *(Completed!)*

- [x] Daily SST, Chl-a, SSH (H3 indexed)
- [x] 10-year time series
- [x] Spatial and temporal validation

### Layer 1b — Classified Seascapes *(Optional)*

- [ ] Classification strategy
- [ ] Daily classified maps
- [ ] Continuous vs classified comparison

### Layer 2 — Derived Features (Ecological Feature Engineering)

- [ ] Gradients (thermal, productivity, SSH)
- [ ] Anomalies
- [ ] Wind-derived metrics

### Layer 3 — Biological & Fishing Layer (Integration Layer)

- [ ] Species movement (telemetry)
- [ ] Fishing effort (AIS / VMS)

### Layer 4 — Bycatch Risk Model (Prediction Layer)

- [ ] Baseline model
- [ ] Dynamic risk maps
- [ ] Predictive evaluation
