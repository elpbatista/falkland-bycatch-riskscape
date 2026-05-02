# Note: Chlorophyll Log Transformation (Layer 2)

## Data Inspection

Raw Copernicus chlorophyll was inspected before defining the log transformation.

Results:

- Total valid values inspected: 317,023,954
- Total zero values: 0
- Minimum positive chlorophyll: 0.0145849 mg m⁻³

No true zero values exist in the dataset.

---

## Decision

Chlorophyll will be transformed using:

log10(chl)

No epsilon term will be added.

NaN values will be preserved.

---

## Rationale

- No zero values are present, so log10 is numerically safe.
- Adding epsilon would unnecessarily distort real measurements.
- log10 provides ecologically interpretable order-of-magnitude contrasts.
- Gradients computed on log10(chl) represent multiplicative habitat contrast.

---

## Status

Final transformation for Layer 2 chlorophyll gradients: confirmed.
