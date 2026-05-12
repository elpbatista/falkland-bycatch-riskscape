# Notes

The current system estimates a relative bycatch risk index derived from
the spatial and temporal co-occurrence of species use and fishing activity.
This index is not a direct estimate of bycatch probability, as it does not
incorporate observed bycatch events.

Future work will integrate observed bycatch data to calibrate the risk index,
enabling the estimation of probabilistic bycatch outcomes and improving the
operational relevance of the system.

```text
predict.py takes the environmental grid,
asks the species model “how much species use here?”,
asks the fishing model “how much fishing activity here?”,
multiplies them,
and saves the resulting risk surface.
```

Bycatch risk is primarily structured by:

1) spatial fishing patterns constrained by bathymetry and distance to coast
2) dynamic species distributions driven by environmental variability

Fishing model is used for forecasting and scenario simulation, but observed fishing data is used for historical risk estimation.

The species-use model produced very similar spatial predictions for BBAL and SAFS. This suggests that, given the available telemetry data and predictor set, the model captured broad habitat-use structure but did not strongly differentiate species-specific habitat responses. Therefore, species-specific risk maps should be interpreted cautiously; the strongest inference is the shared spatial pattern of seabird-fishing overlap.

## Prediction

```text
1. Load separate species models
   species_model_bbal.joblib
   species_model_safs.joblib

2. Predict species_use_log_pred separately per species

3. Use observed fishing_activity

4. Compute:
   fishing_activity_log = log1p(fishing_activity) only where fishing > 0

5. Compute:
   risk_log_pred = species_use_log_pred + fishing_activity_log

6. Force:
   risk_log_pred = 0 where fishing_activity == 0

```

## Three modeling approaches

1. Machine learning (Extra Trees), which directly learns nonlinear relationships between environmental conditions and species-use intensity.

2. Density-based models (Gaussian Mixture Models), which estimate the environmental distribution of observed species-use locations.

3. A Bayesian likelihood framework, where the probability of species use is inferred from the likelihood of environmental conditions given observed use, optionally combined with ecological priors.

This comparison highlights the trade-off between predictive performance (ML) and interpretability and theoretical grounding (Bayesian likelihood).

`Final hazard = α *ML + (1 - α)* Bayesian`

combine ML + Bayesian into a single hybrid model

## For the report

Risk was modeled as the spatial-temporal overlap between
predicted species use intensity and observed fishing effort,
representing a relative encounter-risk surface rather than
an explicit estimate of bycatch probability.

Bayesian mixture models were used to estimate the ecological
plausibility of environmental feature combinations, providing
a probabilistic measure of support for model predictions under
observed and novel conditions.

At H3 resolution 6, a minimum operational exposure unit of 0.5 vessel-hours per cell-day was used, corresponding approximately to one vessel operating within or traversing a grid cell for about 30 minutes at fishing speed.

Although likelihood-based models performed poorly as direct predictors of residence intensity, they were retained as probabilistic environmental-support models. Their purpose was to characterize the multivariate feature space associated with observed species use and to identify cell-days whose environmental conditions resemble known-use conditions.

The Bayesian/GMM model was not used as a direct residence-intensity predictor. Instead, its normalized likelihood was used as an environmental-support weight applied to the machine-learning species-use prediction.

Likelihood-based models were evaluated differently,
because their purpose was to estimate environmental
support rather than directly predict residence intensity.

The likelihood-based model was used as an ecological support filter. It did not replace the machine-learning prediction; instead, it discounted predicted species use in environmental conditions weakly supported by observed tracking data.

Which environmental combinations define plausible species-use conditions?  
Extra Trees Importance: Which variables help prediction accuracy?  
GMM/Bayesian plausibility: Which environmental states are characteristic of observed species presence?  

latent risk = predicted species use under a minimum fishing-effort unit

if plausibility < threshold:
    latent plausible risk = 0
else:
    latent plausible risk = latent risk

The nuance is that low plausibility does not prove absence. It means:

This prediction is outside, or weakly supported by, the environmental conditions represented in the plausibility model.

we do not report latent risk there because the species-use prediction is environmentally unsupported under the plausibility threshold

---

## Species-Use Modeling

Species-use models were trained to predict relative species use from environmental and static spatial predictors. The training dataset was built from the H3/date/species species-presence table joined to the model feature grid. The response variable was `residence_index`, implemented as the product of telemetry record count and individual count for each observed H3/date/species group:

$$
\mathrm{ResidenceIndex}(h,t,s)
=
\mathrm{PresenceCount}(h,t,s)
\times
\mathrm{IndividualCount}(h,t,s)
$$

where $h$ is an H3 cell, $t$ is date, and $s$ is species. For each observed species/date combination, all H3 cells in the feature grid were included. Cells without telemetry observations were assigned `residence_index = 0`, allowing the model to learn both observed-use and zero-use examples within the same daily environmental domain. The modeling target was transformed using:

$$
y = \log(1 + \mathrm{ResidenceIndex})
$$

Predictor variables included dynamic environmental features, derived environmental features, seasonal terms, and static spatial features. Dynamic predictors were SST, SSH, wind speed, and log-transformed chlorophyll-a. Derived predictors included SST, SSH, wind speed, and chlorophyll anomalies, as well as H3-neighbor gradients for SST, SSH, and log-transformed chlorophyll-a. Seasonal predictors were `doy_sin` and `doy_cos`. Static predictors included bathymetric depth, bathymetric slope, distance to coast, and sine/cosine encodings of H3 centroid latitude and longitude.

The implemented training workflow used a joint-species modeling approach. Species identity was included as a categorical predictor using one-hot encoding, and the encoded species indicators were concatenated with the numerical feature matrix. This allowed a single model to learn shared environmental structure while retaining species-specific offsets through the species indicator variables.

Because zero-use rows greatly outnumbered positive-use rows, the training set was balanced before model fitting. All positive rows were retained, and an equal number of zero-use rows was randomly sampled using a fixed random seed (`RANDOM_STATE = 42`). Models were then trained using a random 75/25 train-test split. Sample weights were applied during fitting to give greater influence to higher-use observations:

$$
w = 1 + \mathrm{ResidenceIndex}^{0.75}
$$

Four model classes were included in the model comparison workflow: histogram gradient boosting, random forest, extra trees, and a Bayesian/Gaussian mixture model. The histogram gradient boosting model used 300 boosting iterations, a learning rate of 0.05, a maximum of 31 leaf nodes, and L2 regularization of 0.1. The random forest and extra trees models each used 300 trees, a maximum depth of 20, a minimum leaf size of 5, parallel fitting, and the fixed random seed.

The Bayesian/Gaussian mixture model was implemented as a custom estimator. It fitted a Gaussian mixture model to positive-use observations in standardized feature space, using 10 full-covariance components and a covariance regularization value of $10^{-6}$. The resulting environmental likelihood was normalized using the 1st and 99th percentiles of the training log-density distribution and scaled to the 99th percentile of the target. A histogram gradient boosting prior was also fitted to the full training set, and final predictions from this estimator were calculated as an equal-weighted combination of the likelihood-based estimate and the prior model prediction.

Model outputs were stored as serialized joblib payloads containing the fitted model, species encoder, predictor list, target name, log-transform flag, model name, and model type. Prediction outputs from the species-use models were expressed as `species_use_log_pred`, representing predicted species use on the log-transformed scale. Model comparison metrics were computed after back-transforming predictions to the original target scale using `expm1`, and included $R^2$, root mean squared error, and mean absolute error.

---

## Risk Estimation

Risk estimation was implemented as a relative spatial-temporal overlap index rather than as a direct prediction of observed bycatch probability. The prediction workflow generated species-use and risk surfaces for each H3 cell, date, and species using the model feature grid and observed fishing activity. The active implementation used a hybrid prediction strategy that combined an Extra Trees species-use model with a Bayesian/Gaussian mixture environmental plausibility model.

### Environmental plausibility and support filtering

The Bayesian/Gaussian mixture model was used to estimate environmental plausibility rather than as the primary species-use predictor. For each H3/date/species combination, the model calculated the log density of the environmental feature vector under the fitted Gaussian mixture model. Log densities were normalized to a bounded plausibility score using the fitted 1st and 99th percentile density limits:

$$
\mathrm{Plausibility}(h,t,s)
=
\mathrm{clip}
\left(
\frac{
\ell(h,t,s) - \ell_{\min}
}{
\ell_{\max} - \ell_{\min}
},
0,
1
\right)
$$

where $\ell(h,t,s)$ is the Gaussian mixture log density for cell $h$, date $t$, and species $s$, and $\ell_{\min}$ and $\ell_{\max}$ are the lower and upper normalization limits learned during model fitting. Plausibility values near 1 indicate environmental conditions similar to those associated with observed species use, while values near 0 indicate weak environmental support relative to the fitted use-space distribution.

This plausibility score was used as a support filter for machine-learning predictions. The Extra Trees model produced the primary species-use prediction, and the Bayesian/GMM plausibility score scaled the prediction downward where environmental support was weak. In the implemented presence-gate strategy, predictions were first converted from log space to the original target scale, multiplied by a species-specific gate, and then transformed back to log space:

$$
\mathrm{Gate}(h,t,s)
=
1 - c_s \left(1 - \mathrm{Plausibility}(h,t,s)\right)
$$

$$
\mathrm{HybridUse}(h,t,s)
=
\mathrm{Use}_{ML}(h,t,s)
\times
\mathrm{Gate}(h,t,s)
$$

where $c_s$ is the maximum proportional reduction allowed for species $s$. The implemented maximum reductions were 0.10 for BBAL and 0.50 for SAFS. This means that low plausibility did not force predicted species use to zero, but it reduced the prediction where environmental conditions were weakly supported by the telemetry-informed plausibility model.

### Fishing exposure and realized risk

Observed fishing activity was used for historical risk estimation. For each H3/date combination, fishing activity was calculated from the processed fishing table as:

$$
\mathrm{FishingActivity}(h,t)
=
\mathrm{FishingHours}(h,t)
\times
\mathrm{VesselCount}(h,t)
$$

Fishing activity was transformed using `log1p` only where fishing activity was greater than zero:

$$
\mathrm{FishingActivityLog}(h,t)
=
\log(1 + \mathrm{FishingActivity}(h,t))
$$

with zero retained where no fishing activity was recorded. The realized risk index was then calculated additively in log space:

$$
\mathrm{RiskLogPred}(h,t,s)
=
\mathrm{SpeciesUseLogPred}(h,t,s)
+
\mathrm{FishingActivityLog}(h,t)
$$

This is equivalent to modeling risk as a multiplicative overlap between species use and fishing exposure on the original scale. Cells with no fishing activity therefore received no realized fishing-exposure contribution, even if predicted species use was high.

### Latent risk

In addition to realized risk under observed fishing activity, latent risk maps were generated to represent species-use risk under a standardized minimum fishing exposure. A minimum operational exposure unit of 0.5 vessel-hours per H3 cell-day was used, corresponding to approximately one vessel operating within or traversing a grid cell for about 30 minutes at fishing speed. Latent risk therefore represents where species-use predictions imply potential vulnerability if fishing were present, while realized risk represents where predicted species use overlapped with observed fishing activity.

For plausibility-filtered latent risk, low-plausibility cell-days were interpreted as environmentally weakly supported rather than as confirmed absences. Where plausibility fell below the selected support threshold, latent plausible risk was not reported for that cell-day. This retained the distinction between high predicted species use in well-supported environmental conditions and predictions that occurred outside the environmental domain represented by the telemetry-informed plausibility model.

The final prediction outputs included H3 cell, date, species, hybrid species-use prediction, observed fishing exposure on the log scale, risk prediction on the log scale, and hybrid support variables including plausibility and the effective gate value.

---

## Validation

Model validation was conducted at several levels, including data-quality checks, model-performance evaluation, and plausibility assessment. During preprocessing, feature tables were checked for required columns, consistent `h3` and `date` keys, duplicate records, missing values, and expected data types. Environmental features were inspected after aggregation and transformation to confirm that yearly partitions retained the expected H3/date structure and that derived variables, including gradients and anomalies, were generated without row inflation.

Species-use models were evaluated using a random train-test split, with 25% of rows withheld for testing. Predictions were evaluated after back-transforming from the log scale to the original residence-index scale. Model comparison metrics included coefficient of determination ($R^2$), root mean squared error (RMSE), and mean absolute error (MAE). These metrics were used to compare candidate species-use models and select the primary machine-learning predictor used in the risk workflow.

Additional diagnostic plots were produced to compare predicted and observed species-use values, inspect residuals, and assess model behavior in log space. Feature importance and permutation-based importance were used to evaluate which predictors contributed most strongly to model predictions. These diagnostics supported interpretation of the fitted models but were not treated as independent ecological validation.

Environmental plausibility was evaluated separately from direct species-use prediction. The Bayesian/Gaussian mixture model was used to identify `H3`/`date`/`species` combinations whose environmental conditions were similar to those associated with observed telemetry locations. Plausibility values were therefore interpreted as environmental support diagnostics rather than as direct validation of species presence or absence. Risk maps were interpreted alongside plausibility maps to distinguish well-supported predictions from areas of environmental extrapolation.

Several additional validation steps were not implemented in the current workflow but would strengthen future versions of the analysis. First, spatial or spatiotemporal block cross-validation should be used to test whether models generalize to withheld regions or time periods, rather than only to randomly withheld rows. Second, telemetry tracks could be split by individual, trip, colony, or year to evaluate transferability across animals and sampling periods. Third, sensitivity analysis should be conducted for the plausibility-gate parameter to quantify how risk maps change under different levels of environmental-support filtering. Fourth, predicted risk surfaces should be compared with independent bycatch, observer, or fisheries interaction records when such data become available. Finally, uncertainty should be summarized across model classes, plausibility thresholds, and aggregation choices to identify areas where risk estimates are robust versus areas where conclusions depend strongly on modeling assumptions.

---

## Data Summary

The final study grid contained 37,209 H3 resolution 6 cells covering the Falkland Islands fisheries grid plus a 50 km buffer. Across the 2014-2023 analysis period, this produced 3,652 daily time steps and 135,887,268 H3 cell-day records in the environmental feature grid. The environmental table contained no duplicate `h3`/`date` keys and provided daily values for sea surface temperature, sea surface height, wind speed, log-transformed chlorophyll-a, seasonal terms, spatial gradients, and temporal anomalies.

The raw Global Fishing Watch dataset contained 2,297,069 fishing activity records from 2,011 unique vessels, representing 3,094,974.5 fishing hours between 2014 and 2023. After spatial aggregation to the H3 study grid, the processed fishing-effort table contained 849,818 active `h3`/`date` records across 17,218 H3 cells and all 3,652 dates in the analysis period. These processed records retained 3,086,036.2 fishing hours and were expanded by adding 0 values to the cells without fishing observations into the full 135,887,268-row modeling table so that fishing exposure could be represented for every cell-day.

The cleaned SAERI telemetry dataset contained 59,182 valid records from 42 tracked individuals and 76 trips during 2022-2023. Black-browed albatrosses (BBAL) accounted for 33,425 raw telemetry records from 27 individuals and 58 trips, while South American fur seals (SAFS) accounted for 25,757 records from 15 individuals and 18 trips. After aggregation within the H3 study grid, the species-presence feature table contained 10,268 `h3`/`date`/`species` records across 6,763 H3 cells and 146 observed dates, with 40,824 total telemetry detections retained as H3-level presence counts.

Processed species observations were concentrated in 2022 for BBAL and spanned 2022-2023 for SAFS. BBAL contributed 4,552 `h3`/`date`/`species` rows across 3,270 H3 cells and 16 dates, with 21,329 aggregated presence counts. SAFS contributed 5,716 `h3`/`date`/`species` rows across 4,024 H3 cells and 146 dates, with 19,495 aggregated presence counts.

The resulting modeling products were substantially larger than the raw biological observations because the workflow evaluated species use and risk across the full study grid. The species-training table contained 6,027,858 rows for observed species-date combinations, and the final joint plausibility, prediction, and cube-component tables each contained 257,916,862 species-cell-day records across 2014-2023.

---

## Environmental Feature Generation

### Environmental Coverage and Completeness

The environmental feature-generation workflow produced a continuous daily feature grid for all 37,209 H3 cells across the full 2014-2023 analysis period. The resulting environmental table contained 135,887,268 H3 cell-day records, with one record for each cell on each of 3,652 dates. No duplicate `h3`/`date` keys were present.

Coverage was highest for sea surface temperature, which was complete across the full grid. Sea surface height was available for 99.5% of cell-days, wind speed for 98.4%, and log-transformed chlorophyll-a for 96.6%. Static spatial predictors were complete for all H3 cells, including bathymetric depth, slope, distance to coast, and trigonometric encodings of latitude and longitude.

Across the full environmental grid, sea surface temperature ranged from 271.4 to 293.5 K, with a median of 279.6 K. Sea surface height ranged from -1.32 to 1.25 m, with a median of 0.16 m. Wind speed had a median of 8.1 m/s and a 95th percentile of 13.3 m/s. Log-transformed chlorophyll-a was right-skewed, with a median of 0.22 and a 95th percentile of 0.95.

<!-- Suggested figure: Example environmental layers for a representative date showing SST, SSH, CHL, and wind speed aggregated to the H3 grid. -->

### Derived Environmental Features

The final feature grid expanded the raw environmental inputs into a richer spatiotemporal representation. For each H3 cell-day, the workflow produced base oceanographic variables, cyclic seasonal predictors, static spatial predictors, local spatial gradients, and temporal anomaly fields. The modeling feature grid therefore represented not only the environmental state at a location, but also its seasonal timing, coastal and bathymetric context, local spatial contrast, and departure from expected seasonal conditions.

Static features described broad spatial structure within the study area. Bathymetric depth ranged from shallow shelf and coastal cells to deep offshore waters, with a median depth of 1,581.7 m and a maximum depth of 6,261.6 m. Distance to coast ranged from 14 m to 789.5 km, with a median of 296.3 km. These static features provided persistent spatial context alongside the daily oceanographic variables.

The correlation structure among base environmental predictors showed moderate relationships among SST, chlorophyll-a, and SSH, while wind speed was more weakly correlated with the other variables. SST was moderately correlated with chlorophyll-a and SSH, with correlations of 0.57 and 0.49, respectively. Chlorophyll-a and SSH were also moderately correlated at 0.56. Wind speed had weak negative correlations with SST, chlorophyll-a, and SSH, ranging from -0.15 to -0.10.

<!-- Suggested figure: Correlation heatmap of environmental predictors. -->

### Spatial Gradients and Front-Like Structure

Spatial gradient features captured local environmental heterogeneity across neighboring H3 cells. These fields were generally sparse and right-skewed, indicating that most cell-days had relatively smooth local conditions while a smaller fraction contained stronger spatial contrasts. SST gradients had a median value of 0.080 K and a 95th percentile of 0.282 K. SSH gradients had a median value of 0.0087 m and a 95th percentile of 0.0382 m. Log-chlorophyll gradients had a median value of 0.0091 and a 95th percentile of 0.0967.

The gradient layers added information distinct from the base environmental values. Rather than describing whether a cell was warm, productive, or elevated in sea surface height, these layers identified locations of local spatial transitions and front-like structure. The environmental feature space therefore included both large-scale oceanographic state variables and local spatial heterogeneity.

<!-- Suggested figure: Example gradient layers for SST, SSH, and log-CHL highlighting frontal structure and shelf transitions. -->

### Seasonal and Anomaly Features

Seasonal structure was represented for every H3 cell-day using cyclic day-of-year terms. The sine and cosine seasonal predictors covered their expected range from -1 to 1 and remained balanced across the 10-year record, confirming that the feature grid preserved continuous annual seasonality rather than treating calendar time as a linear variable.

Anomaly features were centered close to zero across the full period, as expected for variables expressed relative to local seasonal conditions. SST anomalies had a standard deviation of 0.85 K and ranged from -6.00 to 7.00 K. SSH anomalies had a standard deviation of 0.11 m, while wind-speed anomalies had a standard deviation of 2.95 m/s. Log-chlorophyll anomalies had a standard deviation of 0.19, with a right-skewed upper tail indicating occasional positive departures from expected seasonal productivity.

Yearly mean anomalies showed that the feature space retained interannual variability after seasonal adjustment. Mean SST anomalies were negative in 2014-2016 and 2019, but positive in 2017-2018 and 2020-2023, with the largest positive yearly mean in 2020. Wind-speed anomalies also varied by year, ranging from negative mean conditions in 2021 and 2022 to positive mean conditions in 2015 and 2023. These results indicate that the generated environmental representation preserved daily, seasonal, spatial, and interannual structure for downstream species-use and risk modeling.

<!-- Suggested figure: Time series of yearly mean SST and wind-speed anomalies across the study area. -->

<!-- Suggested figure: Distribution plots or histograms of anomaly variables showing centered seasonal departures. -->

---

## Date-Key and Seascape Coverage Audit Note

Updated: 2026-05-11 17:09:17 PDT.

The project-wide `h3`/`date` key audit confirmed that the environmental feature grid and source environmental table are fully aligned after normalizing dates to timezone-free UTC calendar days. Across 2014-2023, every yearly `feature_grid` partition matched the corresponding environmental partition exactly on distinct `h3`/`date` keys, with no unmatched keys in either direction.

Audit output:

- `Repo/data/modeling/metrics/date_join_integrity/feature_grid_vs_environmental_join_integrity.csv`
- `Repo/data/modeling/metrics/date_join_integrity/seascapes_vs_feature_grid_join_integrity.csv`
- `Repo/data/modeling/metrics/date_join_integrity/kmeans_k15_feature_grid_unmatched_key_diagnostics.csv`

The seascape assignments do not cover the full feature grid because the KMeans classifier requires complete environmental predictor vectors. Unassigned H3-day cells occur primarily where one or more dynamic predictors are unavailable, especially chlorophyll-related fields over land or masked coastal cells, SSH over land or masked near-coast cells, and wind fields in the outer buffer. Gradient variables can further increase missingness because gradients require valid neighboring-cell context.

This incomplete seascape coverage is therefore a data-availability mask, not a date-key mismatch. The seascape tables are internally valid: all seascape-assigned `h3`/`date` keys match the feature grid, but some feature-grid keys are intentionally absent from the seascape products because the feature vector was incomplete.

Possible report wording:

> Environmental seascape assignments were restricted to H3-day cells with complete predictor vectors. Unassigned cells occurred primarily in land-masked, coastal, or outer-buffer areas where chlorophyll-a, sea-surface height, wind, or derived gradient fields were unavailable. These exclusions did not reflect date-key mismatches; join-integrity checks confirmed complete alignment between the environmental source table and the model feature grid.
