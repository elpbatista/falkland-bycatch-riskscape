# Bycatch Risk Model Specification <!-- omit in toc -->

*This is a draft specification document for the bycatch risk model, outlining the scientific framework, data engineering requirements, and modeling approach.*

---

## TOC <!-- omit in toc -->

- [1. Definitions](#1-definitions)
- [2. Scientific Framework](#2-scientific-framework)
  - [2.1. Core Logic](#21-core-logic)
  - [2.2. Conceptual Structure](#22-conceptual-structure)
  - [2.3. Assumptions](#23-assumptions)
- [3. Data Engineering Specification](#3-data-engineering-specification)
  - [3.1. Data Sources](#31-data-sources)
    - [3.1.1. Environmental Conditions](#311-environmental-conditions)
    - [3.1.2. Species Presence](#312-species-presence)
    - [3.1.3. Fishing Activity](#313-fishing-activity)
    - [3.1.4. Bycatch Events](#314-bycatch-events)
  - [3.2. Spatial Representation](#32-spatial-representation)
  - [3.3. Temporal Representation](#33-temporal-representation)
  - [3.4. Derived Variables](#34-derived-variables)
  - [3.5. Data Structure](#35-data-structure)
  - [3.6. Processing Steps](#36-processing-steps)
  - [3.7. Validation](#37-validation)
- [4. Modeling Specification](#4-modeling-specification)
  - [4.1. Modeling Components](#41-modeling-components)
  - [4.2. Model Relationships](#42-model-relationships)
  - [4.3. Model Outputs](#43-model-outputs)
  - [4.4. Model Assumptions](#44-model-assumptions)

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
- **Hazard (Ecological Potential)**
  Defined by environmental conditions and species presence, independent of fishing activity.
- **Exposure (Fishing Activity)**
- **Bycatch Events (Outcome Layer)**
  Resulting from the combination of hazard and exposure.

### 2.3. Assumptions

- The system is spatially and temporally continuous.
- Environmental conditions, species presence, and fishing activity vary across space and time.

---

## 3. Data Engineering Specification

### 3.1. Data Sources

#### 3.1.1. Environmental Conditions

- Sea Surface Temperature (SST)
  - Provider: [NASA's Physical Oceanography Distributed Active Archive Center (PO.DAAC)](https://podaac.jpl.nasa.gov/)
  - Product: `MUR-JPL-L4-GLOB-v4.1`
  - Variable: `analysed_sst`

- Chlorophyll-a (Chl-a)
  - Provider: [Copernicus Marine Service](https://marine.copernicus.eu/)
  - Product: `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D`
  - Version: 202311
  - Variable: `CHL`

- Sea Surface Height (SSH)
  - Provider: [Copernicus Marine Service](https://marine.copernicus.eu/)
  - Product: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
  - Variable: `adt`

- Wind
  - Provider: [Climate Data Store](https://cds.climate.copernicus.eu/)
  - Product: `derived-era5-single-levels-daily-statistics`
  - Variables:
    - `10m_u_component_of_wind`
    - `10m_v_component_of_wind`

#### 3.1.2. Species Presence

- Telemetry data — [source]

#### 3.1.3. Fishing Activity

- Fishing effort data — [source]

#### 3.1.4. Bycatch Events

- Observational records — [source]

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

Fully automated data processing pipeline that includes:

- Data ingestion
- Data cleaning and filtering
- Spatial aggregation (H3)
- Temporal alignment
- Derived variable computation
- Variable standardization

### 3.7. Validation

Fully automated validation procedures to ensure data quality and integrity, including:

- Range validation
- Missing data validation
- Temporal consistency validation
- Spatial consistency validation
- Statistical summaries

## 4. Modeling Specification

### 4.1. Modeling Components

- Seascapes (Environmental Representation)
- Species Presence Model
- Ecological Hazard Model
- Exposure (Fishing Activity)
- Bycatch Risk

### 4.2. Model Relationships

Ecological hazard is defined as the probability of interaction given environmental conditions and species presence:

$$
H = P(\text{interaction} \mid E, S)
$$

Bycatch risk is defined as the probability of bycatch given ecological hazard and fishing activity:

$$
R = P(\text{bycatch} \mid H, F)
$$

Species presence is represented as a probability conditioned on environmental conditions:

$$
S = P(\text{presence} \mid E)
$$

Where:

- $E$ → Environmental Conditions (Seascapes)
- $S$ → Species Presence
- $H$ → Hazard
- $F$ → Fishing Activity
- $R$ → Bycatch Risk

### 4.3. Model Outputs

- Species presence probability ($S$)
- Ecological hazard probability ($H$)
- Bycatch risk probability ($R$)

All outputs are defined at the spatial–temporal resolution of the data (H3 cell × day).

### 4.4. Model Assumptions

- Species presence is conditionally dependent on environmental conditions.
- Ecological hazard is conditionally dependent on environmental conditions and species presence.
- Bycatch risk is conditionally dependent on ecological hazard and fishing activity.
- Fishing activity represents exposure independent of ecological processes.
- All variables are defined consistently at the same spatial–temporal resolution (H3 cell × day).
