# Time Representation in the Riskscape Pipeline: Date vs DOY

## 1. Purpose

This document defines how time is represented in the riskscape workflow, including:

- why **date** is required
- how **day of year (DOY)** is used
- how to handle **leap years (Option 4)**
- how these choices affect model consistency

---

## 2. Core Principle

The system is **time-explicit**, not climatological.

All layers (environment, species, fishing, bycatch) must align in real time.

---

## 3. Use of `date`

The variable `date` is the **primary temporal index**.

### Roles of `date`

- aligns all datasets:
  - environmental predictors
  - telemetry (species)
  - fishing effort
  - observer data

- preserves:
  - interannual variability
  - anomalies
  - real temporal sequences

### Example

2015-06-12 is not the same as 2020-06-12.

Even if DOY is the same, environmental conditions may differ significantly.

---

## 4. Use of DOY (Day of Year)

DOY is **not a replacement for date**.

It is used as an **additional feature** to encode seasonality.

### Why DOY is needed

Many ecological and oceanographic processes are **cyclical**:

- primary productivity cycles
- seabird foraging behavior
- migration patterns

---

## 5. Problem with Raw DOY

DOY is a **linear variable**, but time is **cyclical**.

Day 1 and Day 365 are close in reality, but far apart numerically.

This creates discontinuities in models.

---

## 6. Cyclical Encoding of DOY

To resolve this, DOY is transformed into circular coordinates:

    doy = df["date"].dt.dayofyear

    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

### Interpretation

- maps DOY onto a **unit circle**
- ensures:
  - continuity between Dec 31 and Jan 1
  - smooth seasonal transitions

---

## 7. Leap Year Problem

Leap years contain **366 days**, which introduces inconsistency.

DOY ranges from 1 to 366 instead of 1 to 365.

If untreated:

- Dec 31 shifts position across years
- seasonal alignment is distorted

---

## 8. Selected Solution: Option 4 (Adjusted DOY)

To maintain a consistent 365-day cycle, DOY is adjusted in leap years.

### Rule

For leap years:

- if DOY > 59 (after Feb 28), subtract 1

### Implementation

    doy = df["date"].dt.dayofyear
    is_leap = df["date"].dt.is_leap_year

    adjusted_doy = doy.copy()
    adjusted_doy[(is_leap) & (doy > 59)] -= 1

Then compute:

    angle = 2 * np.pi * adjusted_doy / 365.0

    df["doy_sin"] = np.sin(angle)
    df["doy_cos"] = np.cos(angle)

---

## 9. Effect on Feb 29

Under this transformation:

| Date   | Adjusted DOY | Result               |
|--------|--------------|----------------------|
| Feb 28 | 59           | unique               |
| Feb 29 | 59           | merged with Feb 28   |
| Mar 1  | 60           | aligned across years |

Feb 29 is treated as an additional observation of Feb 28.

---

## 10. Justification

This approach is selected because it:

### 1. Preserves seasonal consistency

Same calendar day maps to the same position on the seasonal cycle.

### 2. Avoids temporal drift

Mar 1 does not shift across leap and non-leap years.

### 3. Minimizes distortion

- only one day is compressed
- the effect is negligible at dataset scale

### 4. Separates concepts

Calendar time is not the same as seasonal phase.

The model learns **seasonal phase**, not calendar artifacts.

---

## 11. Final Design Decision

The pipeline uses:

### Required

- `date` as the primary temporal axis

### Optional but recommended

- `doy_sin`
- `doy_cos`

### With

- DOY adjusted using **Option 4 (leap-year correction)**

---

## 12. Summary

- `date` ensures temporal alignment across all layers
- `doy_sin` and `doy_cos` encode cyclical seasonality
- Option 4 ensures consistent seasonal phase across leap and non-leap years

This combination provides:

- temporal accuracy
- seasonal continuity
- model stability
- ecological interpretability

---

## 13. Status

This decision is **final and consistent** with the riskscape architecture.
