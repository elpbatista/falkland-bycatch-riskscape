# Bycatch Risk Modeling and Data Pipeline Overview

## Purpose

This document provides an overview of the modeling approach and data pipeline developed to analyze bycatch risk. It is intended to support discussion around the integration of telemetry and fisheries observer datasets.

---

## Modeling Approach

The objective is to understand bycatch risk by separating three independent components:

- Environmental conditions  
- Species presence  
- Fishing activity  

Key principle:

> Species can be present in an area even if there are no vessels.  
> Bycatch only occurs when species presence and fishing activity overlap.

---

## Components

### Environmental Conditions

Environmental variables describe ocean conditions at a given location and time. These variables have been processed, validated, and standardized across the full study period.

---

### Species Presence

Species presence represents the probability of encountering a species under certain environmental conditions.

This will be derived from:

- Telemetry (animal tracking data)

This allows estimation of:

P(species presence | environmental conditions)

This component is independent of fishing activity.

---

### Bycatch Observations

Observed bycatch events provide information on when and where interactions have occurred.

This will be derived from:

- Fisheries observer data  

These observations are used to identify:

- Conditions under which bycatch occurs  
- Locations where interactions have been recorded  

---

### Fishing Activity

Fishing effort determines where interactions can occur.

This will be derived independently using:

- VMS / AIS data  

---

## Data Pipeline

### Spatial Framework

All data are processed using a discrete global grid system:

- H3 hexagonal grid  
- Resolution: 6  

Each record represents:

- A spatial cell (H3 index)  
- A specific day  

---

### Temporal Structure

The dataset is organized as:

- Daily time steps  
- Multi-year period (2014–2023)  

Data are stored per year for scalability.

---

### Data Storage

- Format: Parquet  
- Enables efficient storage and scalable processing  

---

### Processing Workflow

The pipeline follows a consistent process:

1. Data acquisition  
2. Spatial aggregation to H3 grid  
3. Temporal alignment (daily)  
4. Feature generation:
   - Gradients  
   - Anomalies  
5. Validation:
   - Range checks  
   - Distribution summaries  
6. Standardization:
   - Mean and standard deviation computed across full period  
   - Z-score transformation applied  

---

### Standardization

All predictor variables are standardized using full-period statistics:

- Mean and standard deviation computed across all years  
- Applied consistently to each yearly dataset  

This ensures comparability across time.

---

### Automation

The pipeline is fully reproducible:

- Configuration-driven  
- Batch processing by year  
- Modular structure  
- Logging of outputs and summaries  

---

## Environmental Variables Summary

The following variables are currently included in the dataset:

| Variable Type           | Variables                                       |
|-------------------------|-------------------------------------------------|
| Core physical variables | `sst`, `chl`, `ssh`, `wind`                     |
| Gradients               | `sst_grad`, `chl_grad`, `ssh_grad`, `wind_grad` |
| Anomalies               | `sst_anom`, `chl_anom`, `ssh_anom`, `wind_anom` |
| Standardized variables  | All variables above with `_z` suffix            |

Notes:

- Gradients represent local spatial variability  
- Anomalies represent deviations from baseline conditions  
- Standardized variables are used for modeling  

---

## Time Representation (Planned Addition)

In addition to using calendar dates, a cyclical representation of time will be incorporated.

This will use **day-of-year (DOY)** transformed into circular features:

- doy_sin = sin(2π × DOY / 365)  
- doy_cos = cos(2π × DOY / 365)  

This allows the model to represent seasonal cycles continuously.

Leap year handling:

- February 29 will share the same position as February 28 in the annual cycle  

This avoids discontinuities in seasonal representation.

---

## Current Status

Completed:

- Environmental dataset preparation  
- Validation and quality control  
- Standardization across full period  
- H3 spatial framework  
- Automated processing pipeline  

Pending:

- Integration of telemetry data (species presence)  
- Integration of observer data (bycatch observations)  

---

## Summary

The project currently provides:

- A fully processed environmental dataset  
- A consistent spatial-temporal structure (H3, daily resolution)  
- A reproducible and automated pipeline  

The next step is to integrate:

- Telemetry data to represent species presence  
- Observer data to represent bycatch events  

These components will enable analysis of where and when bycatch risk occurs based on the overlap between species presence and fishing activity.
