# Bycatch Risk Model Specification <!-- omit in toc -->

This project develops a workflow for modeling spatial and temporal bycatch risk involving seabirds and marine mammals in the Falkland Islands region.

The approach integrates species movement, fishing effort, and environmental conditions to represent their spatial overlap and identify conditions under which bycatch occurs, with the potential to support risk estimation and forecasting.

## Objective <!-- omit in toc -->

- Develop a workflow to model spatial and temporal bycatch risk involving seabirds and marine mammals in the Falkland Islands region.
- Support the characterization of bycatch risk under varying environmental conditions and fishing activity, with potential for forecasting.
- Maintain the model as a generalizable and extensible framework applicable to other regions and species.

---

TOC goes here

---

## 1. Definitions

- **Environmental Conditions**  
  The physical state of the ocean across space and time, consistent with the concept of dynamic seascapes (Kavanaugh).

- **Species Presence**  
  The occurrence of seabirds and marine mammals across space and time.

- **Fishing Activity**  
  Fishing operations characterized by their spatial distribution and associated effort.

- **Bycatch Events**  
  Interactions between fishing activity and species resulting in capture or mortality.

- **Gradients**  
  Local spatial variation of environmental variables, representing changes between neighboring locations.

- **Anomalies**  
  Deviations of environmental variables from their typical seasonal pattern at a given location and time of year.

---

## 2. Scientific Framework

### 2.1. Core Logic

- Environmental conditions and species presence define ecological hazard.
- Ecological hazard and fishing activity determine the occurrence of bycatch events.

### 2.2. Conceptual Structure

The system is conceptualized as a five-component structure:

- **Seascapes (Physical Layer)**
- **Species Presence (Ecological Layer)**
- **Hazard (Ecological Potential)** → defined by environmental conditions and species presence, independent of fishing activity.
- **Exposure (Fishing Activity)**
- **Bycatch Events (Outcome Layer)** → resulting from the combination of hazard and exposure.

### 2.3. Assumptions

- The system is spatially and temporally continuous.
- Environmental conditions, species presence, and fishing activity vary across space and time.

---

## 3. Data Engineering Specification

### 3.1. Data Sources

- **Environmental Conditions**
  - Sea Surface Temperature (SST)
  - Chlorophyll-a concentration (Chl-a)
  - Sea Surface Height (SSH)
  - Wind

- **Species Presence**
  - Telemetry data

- **Fishing Activity**
  - Fishing effort data

- **Bycatch Events**
  - Observational records

### 3.2. Spatial Representation

- H3 discrete global grid (resolution 6)
- Hexagonal cells define the spatial unit of analysis
- All variables and system components are represented at the cell level

### 3.3. Temporal Representation

- Calendar date as primary temporal reference
- Daily temporal resolution
- Seasonal variability represented using day-of-year (DOY)
- Leap years handled through adjusted DOY representation

### 3.4. Derived Variables

- **Gradients**
  - Computed from environmental variables
  - Represent local spatial variation between neighboring cells

- **Anomalies**
  - Computed from environmental variables
  - Represent deviations from the typical seasonal pattern at each location

### 3.5. Data Structure

- Unit of analysis: H3 cell × day
- Each record represents a single cell at a given date
- Variables are stored as attributes of each cell–day record
- All system components are represented within the same structure

### 3.6. Processing Steps

- Data ingestion
- Data cleaning and filtering
- Spatial aggregation (H3)
- Temporal alignment
- Derived variable computation
- Variable standardization

### 3.7. Validation

- Range validation
- Missing data validation
- Temporal consistency validation
- Spatial consistency validation
- Statistical summaries

## 4. Modeling Specification

### 4.1. Modeling Objective

- Define prediction target

### 4.2. Predictors

- Environmental variables
- Derived variables (gradients, anomalies)
- Temporal features (e.g., DOY encoding)

### 4.3. Target Variables

- Bycatch observations (if available)
- Proxy variables (if applicable)

### 4.4. Feature Representation

- Standardized variables (_z)
- Temporal encoding (e.g., sin/cos DOY)

### 4.5. Data Assembly

- Unit of analysis (e.g., cell × day)
- Join logic across datasets

### 4.6. Training Dataset

- Time range
- Inclusion criteria
- Handling missing data

### 4.7. Validation Strategy

- Temporal split
- Spatial considerations

### 4.8. Outputs

- Predicted risk
- Intermediate model outputs

## 5. Notes

- No mixing between sections
- Scientific logic must not depend on file structure
- Data engineering must not redefine scientific meaning
- Modeling must only consume validated data products
