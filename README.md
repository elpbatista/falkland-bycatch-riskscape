# MS Capstone Project

## Bycatch Riskscape Workflow for Falkland Islands Fisheries: A Proof-of-Concept

This MS capstone project develops and tests a repeatable workflow for creating dynamic “bycatch riskscapes” that integrate species movement data, fishing effort, and oceanographic conditions to forecast spatial overlap and bycatch risk. Focusing on one to two specimen species with robust data availability—southern giant petrel (Macronectes giganteus) and South American fur seal (Arctocephalus australis)—the project will demonstrate a proof-of-concept system for the Falkland Islands fisheries zone. The workflow emphasizes technical rigor, reproducibility, and scalability, using satellite-derived seascapes, AIS/VMS tracking data, species telemetry, and wind data to model spatiotemporal risk zones. This foundational work establishes the data processing pipeline, modeling framework, and validation approach necessary for broader application across additional species and regions.

## Objectives

### Objective 1: Develop a repeatable workflow for constructing dynamic bycatch riskscapes <!-- omit in toc -->

Design and document a standardized, modular workflow for integrating multi-source data (species tracking, fishing effort, oceanographic variables, wind patterns) into spatially and temporally explicit bycatch risk models. The workflow will include data acquisition, preprocessing, feature engineering, model training, validation, and visualization protocols. Emphasis will be placed on reproducibility, code documentation, and version control to enable future applications to additional species and study areas.

### Objective 2: Model bycatch risk for specimen species using satellite seascapes, wind, telemetry, and fishing effort data <!-- omit in toc -->

Apply the riskscape workflow to southern giant petrel and South American fur seal, leveraging available GPS/PTT telemetry data, observer records, AIS/VMS fishing vessel tracking, and environmental covariates (sea surface temperature, chlorophyll-a, sea surface height, eddies, and wind speed/direction). Integrate wind as a key environmental driver influencing seabird foraging range, flight behavior, and habitat accessibility. Use machine learning approaches (Random Forest, LSTM neural networks) to forecast high-risk zones based on species presence probability, fleet overlap, and environmental conditions. Stratify models by season and behavioral state (e.g., foraging vs. commuting) to capture temporal variability in risk.

### Objective 3: Validate riskscape predictions and assess model performance <!-- omit in toc -->

Evaluate model accuracy using independent datasets, including fisheries observer bycatch records, cross-validation techniques, and spatial holdout tests. Assess model sensitivity to data resolution, environmental variables, and temporal scale. Document model performance metrics (AUC, precision-recall, spatial correlation) and identify key predictors of bycatch risk for each specimen species. Compare predicted risk zones with known bycatch hotspots to refine the modeling approach and inform data needs for future expansion.
