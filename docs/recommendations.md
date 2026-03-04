# Recommendations for the Dynamic Bycatch Riskscape Model

This note summarizes recommendations to improve robustness, interpretability, and stability of the current five-layer bycatch riskscape framework.

## 1) Standardize variables before computing gradients

Layer 1 variables have different numerical scales (SST, log10(Chl), SSH). Compute gradients from standardized variables so one variable does not dominate.

Recommended workflow:

- Standardize:
  - z(SST)
  - z(log10(Chl))
  - z(SSH)
- Then compute gradients:
  - G(x,t) = grad(S(x,t))

## 2) Smooth environmental fields before gradient calculation

Daily products can be noisy and can produce unstable gradients. Apply temporal smoothing before computing gradients.

Recommended window (to test):

- 3-day, 5-day, or 7-day smoothing

Then compute gradients from the smoothed fields.

## 3) Represent wind using u/v components (Layer 2)

If wind is included to improve seabird SDMs, prefer wind components over direction angles:

- wind_u
- wind_v

This avoids circular-variable issues and improves model stability.

## 4) Consider temporal lags in the species SDM (Layer 2)

Species may respond to environmental changes with delays. Consider adding lagged predictors as SDM covariates, for example:

- G(x,t-1), G(x,t-2)
- or 3- to 7-day lag summaries

## 5) Maintain separation between species and hazard models

Both models depend on seascape gradients:

- P_species = f(G)
- P_hazard = g(G)

To avoid the two surfaces becoming redundant, ensure they remain identifiable by using different training data:

- Species SDM trained on tracking/telemetry
- Hazard model trained on bycatch observations

## 6) If possible, include gear type in the hazard model (Layer 3)

Bycatch risk varies strongly by gear. If gear data are available, consider:

- P_hazard = g(G, gear_type)

This can improve realism and interpretability.

## 7) Interpret outputs as three distinct products

Your architecture naturally yields three management-relevant surfaces:

- P_species(x,t): ecological presence
- P_hazard(x,t): environmental interaction hazard
- Impact(x,t) = P_species *P_hazard* Effort: realized operational impact

Keeping these outputs distinct strengthens interpretation and decision support.

## Summary of key recommendations

1. Standardize before gradients.
2. Smooth before gradients (3–7 day tests).
3. Wind as u/v components (Layer 2).
4. Add temporal lags for SDM (Layer 2).
5. Keep SDM vs hazard training datasets separate.
6. Add gear type to hazard model if available.
7. Treat P_species, P_hazard, and Impact as distinct outputs.
