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

Joint species model saved: /Users/pb/Work/OSU/CapstoneProject/bycatch-riskscape/data/modeling/models/species_model.joblib
{'r2': -0.3750467813453888, 'rmse': 10.792053080025694, 'mae': 2.84695716031283, 'train_rows': 7639, 'test_rows': 2546}
BBAL species model saved: /Users/pb/Work/OSU/CapstoneProject/bycatch-riskscape/data/modeling/models/species_model_bbal.joblib
{'r2': 0.7435731749852437, 'rmse': 20.34022408773636, 'mae': 3.711006998479471, 'species': 'BBAL', 'train_rows': 3370, 'test_rows': 1123}
SAFS species model saved: /Users/pb/Work/OSU/CapstoneProject/bycatch-riskscape/data/modeling/models/species_model_safs.joblib
{'r2': 0.718534843872827, 'rmse': 3.083809161861689, 'mae': 2.0280969776015296, 'species': 'SAFS', 'train_rows': 4269, 'test_rows': 1423}
pb@Pilar bycatch-riskscape % python3 scripts/predict_models.py
