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

## 1. Scientific Framework

### 1.1. Core Logic

The system can be summarized through two primary relationships:

- Environmental conditions and species presence together define ecological hazard.
- Ecological hazard and exposure to fishing activity together determine the occurrence of bycatch events.

### 1.2. System Components

- Environmental Conditions
- Species Presence
- Fishing Activity
- Bycatch Events

### 1.3. Definitions

- **Environmental Conditions**  
  The physical state of the ocean varying in space and time, consistent with the concept of dynamic seascapes as described by Maria Kavanaugh.

- **Species Presence**  
  The occurrence of seabirds and marine mammals in space and time.

- **Fishing Activity**  
  Human fishing operations characterized by their spatial distribution and associated effort.

- **Bycatch Events**  
  Interactions between fishing activity and species that result in capture or mortality.

- **Gradients**  
  Local spatial variation of a variable, representing changes between neighboring locations.

- **Anomalies**  
  Deviations of a variable from its typical seasonal pattern at a given location and time of year.

### 1.4. Conceptual Structure

The system is conceptualized as a five-component structure:

- **Seascapes (Physical Layer)**  
  Environmental conditions describing the spatial–temporal physical state of the ocean.

- **Species Presence (Ecological Layer)**  
  Spatial–temporal presence of species.

- **Hazard (Ecological Potential)**  
  Conditions under which species presence and environmental factors create the potential for interaction, independent of fishing activity.

- **Exposure (Fishing Activity)**  
  Spatial–temporal occurrence of fishing activity representing the potential for species to be exposed to fishing operations.

- **Bycatch Events (Outcome Layer)**  
  Outcomes resulting from the combination of hazard and exposure, leading to capture or mortality of species.

### 1.5. Assumptions

- Environmental conditions influence species presence.
- Species presence and environmental conditions define ecological hazard.
- Fishing activity represents exposure independent of ecological processes.
- Bycatch events arise from the combination of ecological hazard and exposure to fishing activity.
- The system is spatially and temporally continuous, with processes varying across both dimensions.

---

## 2. Data Engineering Specification

### 2.1. Data Sources

- SST dataset
- Chlorophyll dataset
- SSH dataset
- Wind dataset
- Telemetry data (if available)
- Observer data (if available)
- Fishing effort data (if available)

### 2.2. Spatial Representation

- H3 grid definition
- Resolution
- Spatial indexing rules

### 2.3. Temporal Representation

- Date field
- DOY / adjusted DOY (if implemented)
- Handling of leap years

### 2.4. Derived Variables

- Gradients
- Anomalies
- Standardized variables

### 2.5. Processing Steps

- Raw data ingestion
- Cleaning and filtering
- Spatial aggregation to H3
- Temporal alignment
- Variable derivation
- Standardization

### 2.6. Data Structure

- File formats (e.g., Parquet)
- Partitioning strategy (e.g., by year)
- Column schema

### 2.7. Validation

- Range checks
- Missing values
- Consistency checks across years
- Statistical summaries

### 2.8. Reproducibility

- Scripts
- Configuration files
- Logging of sources and parameters

## 3. Modeling Specification

### 3.1. Modeling Objective

- Define prediction target

### 3.2. Predictors

- Environmental variables
- Derived variables (gradients, anomalies)
- Temporal features (e.g., DOY encoding)

### 3.3. Target Variables

- Bycatch observations (if available)
- Proxy variables (if applicable)

### 3.4. Feature Representation

- Standardized variables (_z)
- Temporal encoding (e.g., sin/cos DOY)

### 3.5. Data Assembly

- Unit of analysis (e.g., cell × day)
- Join logic across datasets

### 3.6. Training Dataset

- Time range
- Inclusion criteria
- Handling missing data

### 3.7. Validation Strategy

- Temporal split
- Spatial considerations

### 3.8. Outputs

- Predicted risk
- Intermediate model outputs

## 4. Notes

- No mixing between sections
- Scientific logic must not depend on file structure
- Data engineering must not redefine scientific meaning
- Modeling must only consume validated data products
