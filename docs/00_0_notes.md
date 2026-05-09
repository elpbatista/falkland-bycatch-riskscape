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
