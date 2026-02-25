# Bycatch Riskscape Workflow for Falkland Islands Fisheries <!-- omit in toc -->

## A Proof-of-Concept MS Capstone Project <!-- omit in toc -->

- [1. Problem Statement](#1-problem-statement)
- [2. Summary](#2-summary)
- [3. Research Questions](#3-research-questions)
- [4. Objectives](#4-objectives)
- [5. Target Specimen Species](#5-target-specimen-species)
  - [5.1. Seabirds](#51-seabirds)
  - [5.2. Marine Mammals](#52-marine-mammals)
- [6. Data Availability](#6-data-availability)
- [7. Scope and Limitations](#7-scope-and-limitations)
- [8. Risks and Mitigation Strategies](#8-risks-and-mitigation-strategies)
- [9. Methods](#9-methods)
  - [9.1. Data Sources](#91-data-sources)
    - [9.1.1. Species Tracking Data](#911-species-tracking-data)
    - [9.1.2. Fishing Effort Data](#912-fishing-effort-data)
    - [9.1.3. Environmental Data](#913-environmental-data)
    - [9.1.4. Ancillary Data](#914-ancillary-data)
  - [9.2. Workflow Design](#92-workflow-design)
    - [9.2.1. Stage 1: Data Acquisition and Preprocessing](#921-stage-1-data-acquisition-and-preprocessing)
    - [9.2.2. Stage 2: Feature Engineering](#922-stage-2-feature-engineering)
    - [9.2.3. Stage 3: Model Development](#923-stage-3-model-development)
    - [9.2.4. Stage 4: Validation and Sensitivity Analysis](#924-stage-4-validation-and-sensitivity-analysis)
    - [9.2.5. Stage 5: Visualization and Documentation](#925-stage-5-visualization-and-documentation)
  - [9.3. Modeling Approach](#93-modeling-approach)
    - [9.3.1. Species Distribution Modeling](#931-species-distribution-modeling)
    - [9.3.2. Fishing Effort Mapping](#932-fishing-effort-mapping)
    - [9.3.3. Risk Index Calculation](#933-risk-index-calculation)
    - [9.3.4. Incorporating Wind](#934-incorporating-wind)
- [10. Deliverable Format](#10-deliverable-format)
- [11. Stakeholder Engagement](#11-stakeholder-engagement)
- [12. Deliverables](#12-deliverables)
- [13. Timeline (Estimated ~200 hours)](#13-timeline-estimated-200-hours)
- [14. Budget Summary (Estimate)](#14-budget-summary-estimate)
- [15. Significance and Future Directions](#15-significance-and-future-directions)

## 1. Problem Statement

Seabird and marine mammal bycatch remains a persistent management challenge in the Falkland Islands fisheries. Existing spatial tools tend to be static, retrospective, or limited in their ability to incorporate dynamic environmental drivers such as wind, eddies, or seasonal seascape structures. Fisheries managers lack operational, data-driven workflows that can integrate species behavior, environmental variability, and fishing effort into a unified predictive system. This project addresses that gap by developing a repeatable workflow for generating dynamic bycatch “riskscapes,” enabling more adaptive and informed management decisions.

## 2. Summary

This MS capstone project develops and tests a repeatable workflow for creating dynamic “bycatch riskscapes” that integrate species movement data, fishing effort, and oceanographic conditions to forecast spatial overlap and bycatch risk. Focusing on one to two specimen species with robust data availability—southern giant petrel (Macronectes giganteus) and South American fur seal (Arctocephalus australis)—the project will demonstrate a proof-of-concept system for the Falkland Islands fisheries zone. The workflow emphasizes technical rigor, reproducibility, and scalability, using satellite-derived seascapes, AIS/VMS tracking data, species telemetry, and wind data to model spatiotemporal risk zones. This foundational work establishes the data processing pipeline, modeling framework, and validation approach necessary for broader application across additional species and regions.

## 3. Research Questions

- How do dynamic oceanographic features (e.g., seascapes, eddies, fronts) and wind patterns influence the spatial distribution of seabirds and marine mammals in the Falkland Islands region?
- To what extent do species–fishery interactions vary seasonally, and can riskscapes reliably capture these patterns?
- Which environmental and behavioral predictors most strongly influence bycatch risk?
- How well can machine learning and time-series approaches forecast spatial bycatch risk when integrating fishing effort, telemetry, and environmental variables?

## 4. Objectives

### Objective 1: Develop a repeatable workflow for constructing dynamic bycatch riskscapes <!-- omit in toc -->

Design and document a standardized, modular workflow for integrating multi-source data (species tracking, fishing effort, oceanographic variables, wind patterns) into spatially and temporally explicit bycatch risk models. The workflow will include data acquisition, preprocessing, feature engineering, model training, validation, and visualization protocols. Emphasis will be placed on reproducibility, code documentation, and version control to enable future applications to additional species and study areas.

### Objective 2: Model bycatch risk for specimen species using satellite seascapes, wind, telemetry, and fishing effort data <!-- omit in toc -->

Apply the riskscape workflow to southern giant petrel and South American fur seal, leveraging available GPS/PTT telemetry data, observer records, AIS/VMS fishing vessel tracking, and environmental covariates (sea surface temperature, chlorophyll-a, sea surface height, eddies, and wind speed/direction). Integrate wind as a key environmental driver influencing seabird foraging range, flight behavior, and habitat accessibility. Use machine learning approaches (Random Forest, LSTM neural networks) to forecast high-risk zones based on species presence probability, fleet overlap, and environmental conditions. Stratify models by season and behavioral state (e.g., foraging vs. commuting) to capture temporal variability in risk.

### Objective 3: Validate riskscape predictions and assess model performance <!-- omit in toc -->

Evaluate model accuracy using independent datasets, including fisheries observer bycatch records, cross-validation techniques, and spatial holdout tests. Assess model sensitivity to data resolution, environmental variables, and temporal scale. Document model performance metrics (AUC, precision-recall, spatial correlation) and identify key predictors of bycatch risk for each specimen species. Compare predicted risk zones with known bycatch hotspots to refine the modeling approach and inform data needs for future expansion.

## 5. Target Specimen Species

The following species were selected based on data availability, documented bycatch interactions, and their utility as representative taxa for proof-of-concept modeling.

### 5.1. Seabirds

| Common Name           | Scientific Name         | Status        | Justification                                                                                                                                                                  |
|-----------------------|-------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Southern giant petrel | *Macronectes giganteus* | Least Concern | Documented scavenger behind trawlers; robust telemetry datasets available from Falklands breeding colonies; wide-ranging forager suitable for testing dynamic seascape models. |

### 5.2. Marine Mammals

| Common Name             | Scientific Name           | Status        | Interaction Risk                                                                                                                                                |
|-------------------------|---------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| South American fur seal | *Arctocephalus australis* | Least Concern | Occasional bycatch in trawl gear and depredation near vessels; telemetry and sightings data available; coastal-offshore gradient suitable for habitat modeling. |

*Species selection is contingent on final data availability. Substitution may occur if needed.*

## 6. Data Availability

To complete the workflow and modeling tasks, the following data sources will be used:

- **Telemetry data**: Access coordinated with SAERI partners (GPS/PTT tracking of selected specimen species). Preliminary confirmation of availability has been established.
- **AIS/VMS fishing effort data**: Access coordinated through the Falkland Islands Fisheries Department. Expected to be available under standard research-use agreements.
- **Environmental data**: Publicly available sources including Copernicus Marine Service, GEBCO, ERA5, and NOAA GFS.
- **Observer records**: Access expected through SAERI/FIFD collaboration for validation purposes.

Where delays occur, placeholder datasets or publicly available analogues will be used to continue pipeline and model development.

## 7. Scope and Limitations

This project is designed as a focused proof-of-concept with the following boundaries:

- Models will be developed for **one to two species**, not a full ecosystem assessment.
- The project will use **historical environmental and fishing data**, not real-time prediction systems.
- Dashboard or web service implementation is **beyond scope**, though documented for future work.
- Species substitution may occur if data access is delayed.
- The workflow will not include socio-economic or regulatory decision modeling.
- Spatial resolution may be constrained by data volume and computational requirements.

These limitations ensure the project remains feasible within the ~200-hour capstone timeframe.

## 8. Risks and Mitigation Strategies

| Risk                                                      | Mitigation Strategy                                                                                                                 |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Delay in receiving telemetry or AIS/VMS data              | Begin development using publicly available datasets; substitute species if necessary; structure workflow for easy data replacement. |
| Environmental datasets too large or computationally heavy | Use coarser-resolution composites; subset spatial domain; run processing in cloud or HPC environments when available.               |
| LSTM or ML models underperform                            | Fall back to simpler, interpretable models (Random Forest, GAM, MaxEnt); refine feature engineering.                                |
| Difficulty integrating wind or dynamic seascapes          | Implement simplified wind indices first; incrementally add complexity; rely on seascape composites where necessary.                 |
| Overly broad scope for limited time                       | Prioritize workflow development and one species; defer multi-species expansions to future work.                                     |

## 9. Methods

### 9.1. Data Sources

#### 9.1.1. Species Tracking Data

- GPS/PTT telemetry from Falklands breeding colonies (SAERI, BirdLife International, BAS archives)
- Seabird-at-sea surveys and observer records
- Acoustic detections and visual sightings (marine mammals)

#### 9.1.2. Fishing Effort Data

- AIS/VMS vessel tracking data (Falkland Islands Fisheries Department)
- Observer logbooks and bycatch incident reports
- Fishing effort layers (trawl, jig, longline by month/season)

#### 9.1.3. Environmental Data

- Sea surface temperature (SST), chlorophyll-a concentration, sea surface height (SSH), eddy kinetic energy (Copernicus Marine Service)
- Satellite-derived seascape classifications (Kavanaugh et al. approach)
- Wind speed and wind direction (ERA5 reanalysis or NOAA GFS forecasts)

#### 9.1.4. Ancillary Data

- Bathymetry (GEBCO), distance to shelf break, distance to colony
- Sea ice extent (seasonal, if applicable)
- Frontal zones and mesoscale features

### 9.2. Workflow Design

The workflow will follow a modular, open-source structure to ensure repeatability and transparency. Key stages include:

#### 9.2.1. Stage 1: Data Acquisition and Preprocessing

- Automate download of environmental data from public repositories (Copernicus, NOAA, NASA)
- Process AIS/VMS data to extract fishing effort layers (vessel density, speed-based fishing activity)
- Clean and standardize telemetry data; filter for quality flags, interpolate missing positions
- Generate daily/weekly composite seascapes and wind fields

#### 9.2.2. Stage 2: Feature Engineering

- Distance to fronts, eddy proximity, chlorophyll gradients
- Wind components: headwind/tailwind, crosswind relative to movement direction
- Temporal features: day of year, month, breeding phenology
- Aggregated fishing effort by grid cell and time window

#### 9.2.3. Stage 3: Model Development

- Random Forest and/or MaxEnt for species habitat suitability
- LSTM neural networks for dynamic time-series forecasting
- Risk index = species presence probability × fishing effort intensity

#### 9.2.4. Stage 4: Validation and Sensitivity Analysis

- Spatial and temporal cross-validation
- Validation using independent observer bycatch records
- Sensitivity tests (variable removal, grid resolution tests)

#### 9.2.5. Stage 5: Visualization and Documentation

- Monthly or seasonal risk maps
- Time-series animations
- Reproducible notebooks (Jupyter/R Markdown)
- GitHub repository with workflow and documentation

### 9.3. Modeling Approach

#### 9.3.1. Species Distribution Modeling

- Ensemble ML models (Random Forest, BRT)
- Incorporation of wind as environmental covariate
- Telemetry-driven model training; validation with independent sightings

#### 9.3.2. Fishing Effort Mapping

- Grid-based aggregation of AIS/VMS data using speed filters (< 3 knots)
- Gear-type differentiation (trawl vs. jig)

#### 9.3.3. Risk Index Calculation

- Species presence × fishing effort intensity
- Thresholds to classify low, moderate, high risk
- Seasonal comparisons

#### 9.3.4. Incorporating Wind

- Wind-driven accessibility modeling
- Behavioral thresholds (e.g., gliding vs. active flight)
- Interaction between wind and foraging distribution

## 10. Deliverable Format

At minimum, the following deliverables will be produced:

- A written capstone project report (PDF), suitable as a thesis chapter  
- A complete GitHub repository containing:  
  - Workflow documentation  
  - All scripts (Python + R as needed)  
  - Reproducible Jupyter notebooks  
  - Example inputs and outputs (as permitted by data agreements)  
- Figures, maps, and risk surfaces (PDF, PNG, GeoTIFF)
- A final oral or poster presentation

## 11. Stakeholder Engagement

This MS capstone project is designed as a technical proof-of-concept and will involve limited stakeholder engagement focused on data access and validation feedback. Key stakeholders include:

- SAERI (South Atlantic Environmental Research Institute)
- Falkland Islands Fisheries Department (FIFD)
- Academic advisors at OSU

Future expansion may integrate fishers, NGOs, and regulatory agencies.

## 12. Deliverables

| Deliverable                         | Description                                                                                                                                   |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Workflow Documentation              | Step-by-step protocols for data acquisition, processing, modeling, and validation; published as a GitHub repository with README and tutorials |
| Species-Specific Risk Maps          | Monthly or seasonal bycatch risk maps for southern giant petrel and South American fur seal (GeoTIFF and PDF formats)                         |
| Model Performance Report            | Technical report documenting model accuracy, validation results, sensitivity analyses, and key predictor variables                            |
| Code Repository                     | Fully documented Python and R scripts for all workflow stages; Jupyter notebooks with inline explanations                                     |
| MS project                          | Written thesis chapter describing methods, results, and discussion for submission                                                             |
| Peer-Reviewed Manuscript (optional) | Draft manuscript suitable for submission to a journal                                                                                         |
| Presentation                        | Oral or poster presentation                                                                                                                   |

## 13. Timeline (Estimated ~200 hours)

| Phase   | Activities                                       | Duration |
|---------|--------------------------------------------------|----------|
| Phase 1 | Data access, preprocessing, exploratory analysis | 50 hours |
| Phase 2 | Workflow development and initial modeling        | 70 hours |
| Phase 3 | Validation and sensitivity analysis              | 40 hours |
| Phase 4 | Visualization and documentation                  | 30 hours |
| Phase 5 | Writing and presentation                         | 10 hours |

Total time: ~200 hours.

## 14. Budget Summary (Estimate)

| Category                                              | Estimated Cost |
|-------------------------------------------------------|----------------|
| Satellite data access (if proprietary sources needed) | ?              |
| Conference travel and registration (optional)         | ?              |
| Software licenses (if required)                       | ?              |
| Miscellaneous (printing, publication fees)            | ?              |
| **Total**                                             | $              |

## 15. Significance and Future Directions

This project will provide the first fully documented, dynamic bycatch riskscape workflow tailored to Falkland Islands fisheries. By integrating species behavior, environmental variability, and fishing patterns, the project supports adaptive management and conservation planning. The workflow’s modular design enables future expansion to additional species, regions, and operational tools such as dashboards or early-warning systems. Its deliverables will directly support ongoing SAERI research and build a foundation for peer-reviewed publication and continued academic advancement.
