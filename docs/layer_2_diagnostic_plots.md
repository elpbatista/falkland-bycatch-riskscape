# Layer 2A — Diagnostic Plots

The following four plots were generated to evaluate the structural integrity
and ecological relevance of the spatial gradient features derived from the
10-year Layer 1 dataset.

---

## 1. 10-Year Mean SST

Represents the time-averaged sea surface temperature field across the H3 grid.

Purpose:

- Verify spatial coherence of extraction
- Confirm large-scale thermal gradients
- Validate correct geospatial alignment

Interpretation:

- Smooth latitudinal gradient
- Shelf-break temperature structure visible
- No spatial artifacts or striping

---

## 2. 10-Year Mean SST Gradient

Represents the time-averaged gradient magnitude defined as:

$$
G_i = \sqrt{\frac{1}{n} \sum_{j \in N(i)} (X_i - X_j)^2}
$$

Where:

- $N(i)$ = set of ring-1 neighboring H3 cells of cell $i$
- $n$ = number of valid neighboring cells
- $X_i$ = value of the variable at cell $i$
- $X_j$ = value of the same variable at neighboring cell $j$
- $G_i$ = root-mean-square spatial contrast at cell $i$

Purpose:

- Identify persistent thermal front intensity
- Detect mesoscale eddy corridors
- Validate neighbor-based gradient computation

Interpretation:

- Strong shelf-break front
- Northern boundary structure
- Central mesoscale eddy signal
- Structured, not noisy

---

## 3. SST Front Frequency (Top 25%)

Binary front defined using threshold:

$$
\tau = \text{75th percentile of } sst\_grad
$$

Front indicator:

$$
F_{i,t} =
\begin{cases}
1 & \text{if } sst\_grad_{i,t} > \tau \\
0 & \text{otherwise}
\end{cases}
$$

Front frequency:

$$
P_i = \frac{1}{T} \sum_t F_{i,t}
$$

Purpose:

- Identify persistent frontal zones
- Distinguish stable vs intermittent fronts

Interpretation:

- Shelf-break region highly persistent
- Open ocean fronts more variable
- Strong spatial coherence

---

## 4. Joint SST × Chlorophyll Front Frequency

Binary joint event defined as:

$$
J_{i,t} = F^{sst}_{i,t} \cdot F^{chl}_{i,t}
$$

Joint frequency:

$$
P_i = \frac{1}{T} \sum_t J_{i,t}
$$

Purpose:

- Identify biologically active physical fronts
- Detect convergence zones relevant for foraging ecology
- Filter purely physical gradients

Interpretation:

- More selective than SST-only fronts
- Strong coupling near shelf-break
- Mesoscale eddy corridors highlighted
- Northern thermal band reduced (less biological coupling)

---

## Summary

Layer 2A gradients show:

- Physically coherent spatial structure
- Persistent frontal systems
- Biologically relevant coupled fronts
- No extraction artifacts

The system is ready for Layer 2B anomaly computation.
