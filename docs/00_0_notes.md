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
