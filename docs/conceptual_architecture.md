# Conceptual Architecture of Bycatch Riskscape

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
