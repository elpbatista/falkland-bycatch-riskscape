# BlockCV Validation Summary for the Joint Extra Trees Species-Use Model

Last verified: 2026-05-11 16:09:51 PDT

This note records the first BlockCV-style validation panel for the joint Extra Trees species-use model. The purpose is to compare the optimistic random holdout baseline with more spatially and environmentally structured validation designs.

All runs used a 12% holdout target and the same model implementation as the production species-use workflow. Only the train/test split design changed.

## Metric Verification Rule

Any metric copied from this note into the report must retain or be traceable to:

- the source CSV path,
- the verification date and time,
- the validation split,
- the model name,
- the target scale: raw back-transformed residence index or log-transformed residence index,
- the train and test row counts.

Authoritative comparison metrics source:

`Repo/data/modeling/metrics/species_model_random12_vs_kmeans_k15_block_cv_comparison.csv`

## Main Contender Comparison

The two main validation contenders are the random 12% holdout and the K = 15 environmental seascape holdout. The random split is the optimistic interpolation baseline used for production-style model fitting. The K = 15 environmental split is the selected BlockCV-style ecological transferability diagnostic.

| Validation design | Model | Raw R2 | Raw RMSE | Raw MAE | Log R2 | Log RMSE | Log MAE | Train rows | Test rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_12pct | extra_trees | 0.9408 | 8.653 | 1.731 | 0.5649 | 0.520 | 0.342 | 17,926 | 2,444 |
| random_12pct | hist_gradient_boosting | 0.4206 | 27.068 | 2.855 | 0.2534 | 0.681 | 0.523 | 17,926 | 2,444 |
| random_12pct | random_forest | 0.2272 | 31.260 | 2.585 | 0.5131 | 0.550 | 0.361 | 17,926 | 2,444 |
| random_12pct | bayesian_gmm | 0.1203 | 33.352 | 3.519 | 0.0544 | 0.766 | 0.568 | 17,926 | 2,444 |
| environmental_kmeans_k15 | extra_trees | 0.8313 | 67.921 | 4.907 | 0.4658 | 0.623 | 0.434 | 15,201 | 5,169 |
| environmental_kmeans_k15 | hist_gradient_boosting | 0.7302 | 85.898 | 6.604 | 0.2539 | 0.736 | 0.557 | 15,201 | 5,169 |
| environmental_kmeans_k15 | random_forest | 0.5262 | 113.821 | 7.341 | 0.4008 | 0.660 | 0.452 | 15,201 | 5,169 |
| environmental_kmeans_k15 | bayesian_gmm | 0.0406 | 161.972 | 9.323 | 0.1235 | 0.798 | 0.583 | 15,201 | 5,169 |

## Interpretation Notes

The random 12% holdout remains the optimistic interpolation baseline, with Extra Trees reaching raw-scale R2 = 0.9408. This split is useful for continuity with the existing production workflow, but it does not test transfer to new spatial or environmental conditions.

Spatial and buffered H3-parent blocking produce much lower R2 values than the random split, indicating that random row-level validation overstates spatial transferability. The buffered split is slightly stricter because neighboring H3 cells around the held-out blocks are removed from the training set.

The environmental GMM split is the harshest test. Holding out entire Bayesian/GMM environmental components requires the model to predict in environmental regimes not represented during training. The negative R2 and high error therefore indicate poor extrapolation to unseen GMM-defined environmental states.

The K = 15 k-means environmental split provides the selected interpretable validation extension. It holds out feature-only seascape classes rather than random rows, so it better tests transfer to withheld environmental regimes. Under this split, Extra Trees remains the strongest candidate, with raw-scale R2 = 0.8313 and log-scale R2 = 0.4658.

## Report Use

This table can be used in the report as a validation-extension result rather than as a replacement for the main production model. A concise report interpretation would be:

> Blocked validation reduced apparent model performance relative to the random holdout baseline, confirming that random row-level validation overestimates transferability across space and environmental regimes. Spatial and buffered blocks tested geographic transfer, while environmental GMM and k-means seascape blocks tested transfer to novel environmental conditions. The environmental GMM holdout was most stringent, whereas k-means seascape blocking provided a more interpretable intermediate diagnostic.

## Working Decision

The two clearest contenders from the initial validation panel are:

- `random_12pct`, as the optimistic production-style interpolation baseline.
- `environmental_kmeans_k15`, as the ecologically interpretable BlockCV-style extension.

The selected validation extension is `environmental_kmeans_k15`. This design is preferred for reporting because it is connected to the seascape literature, uses the visually interpretable K = 15 environmental-regime level, and tests transfer across withheld environmental seascape classes rather than random row-level interpolation. The random 12% split should be retained as the production baseline, while K = 15 environmental blocking should be used as the primary ecological transferability diagnostic.

## Split Viability Check

Before using the K = 15 environmental-blocking split for model training, a diagnostics-only check was run to verify that the training partition retained enough positive examples for both species.

Command:

```bash
PYTHONPATH=Repo/src python3 -m riskscape.model.block_cv_train \
  --split environmental_seascape \
  --seascape-table seascapes/kmeans_k15 \
  --run-label kmeans_k15_viability \
  --diagnostics-only
```

Diagnostics sources:

- `Repo/data/modeling/metrics/species_model_environmental_seascape_kmeans_k15_viability_split_diagnostics.csv`
- `Repo/data/modeling/metrics/species_model_environmental_seascape_kmeans_k15_viability_block_group_diagnostics.csv`

| Partition | Species | Rows | Positive rows | Zero rows | Positive fraction |
|---|---|---:|---:|---:|---:|
| train | BBAL | 3,500 | 2,729 | 771 | 0.780 |
| train | SAFS | 11,701 | 4,270 | 7,431 | 0.365 |
| test | BBAL | 2,008 | 1,764 | 244 | 0.878 |
| test | SAFS | 3,161 | 1,422 | 1,739 | 0.450 |

The split is viable for Extra Trees training because both species retain substantial positive examples in the training set. The held-out seascapes are biologically uneven, however: `seascape_11` and `seascape_6` contain no BBAL positive rows, while the BBAL test positives are concentrated in `seascape_5`. This does not invalidate the split, but it should be noted when interpreting ecological transferability.

## Literature Note on Environmental Blocking

The environmental-blocking behavior observed here is consistent with the `blockCV` guidance from Valavi et al. (2019). Environmental folds can be generated by clustering either the full environmental raster/domain or only the environmental values associated with species presence/absence or background points. Clustering the full environmental domain produces classes that are consistent across the whole region and across species, but it does not guarantee that every class contains species records. Consequently, the effective validation folds can be fewer than the specified number of clusters, or biologically uneven across species. Clustering only the species/background samples can force the folds to be represented in the modeling data, but the resulting classes are more sample-dependent and can vary between runs unless a random seed is fixed.

This project used clustering on the full environmental feature grid. That choice supports a region-wide, species-independent seascape interpretation, but it also explains why some K = 15 seascape classes contained few or no positive records for one species. For this reason, environmental-blocking splits should be accompanied by species-specific positive-count diagnostics before model fitting.

Report-ready wording:

> Environmental blocking was based on k-means classes fitted to the full environmental feature grid. This ensured region-wide, species-independent environmental regimes, but did not guarantee equal species representation within each class. Consequently, held-out environmental folds were biologically uneven, and fold composition was inspected using species-specific positive-count diagnostics before model fitting.

Reference to add to the report bibliography:

Valavi R, Elith J, Lahoz-Monfort JJ, Guillera-Arroita G. 2019. blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. Methods in Ecology and Evolution 10:225-232. https://doi.org/10.1111/2041-210X.13107

## Date-Key and Seascape Coverage Audit

Last verified: 2026-05-11 17:10:12 PDT

The project-wide date audit normalized H3/day products to timezone-free UTC calendar dates. The two key joins for the environmental-blocking workflow were then checked directly:

- `feature_grid` versus `environmental`
- KMeans seascape assignments versus `feature_grid`

Audit sources:

- `Repo/data/modeling/metrics/date_join_integrity/feature_grid_vs_environmental_join_integrity.csv`
- `Repo/data/modeling/metrics/date_join_integrity/seascapes_vs_feature_grid_join_integrity.csv`
- `Repo/data/modeling/metrics/date_join_integrity/kmeans_k15_feature_grid_unmatched_key_diagnostics.csv`

The `feature_grid` and source environmental table matched exactly on distinct `h3`/`date` keys for every year from 2014 through 2023. The seascape tables were also internally consistent: every seascape-assigned `h3`/`date` key matched the feature grid. However, the seascape tables covered approximately 94.9% of the feature-grid keys because KMeans assignments require complete predictor vectors.

The unassigned feature-grid keys are a data-availability mask rather than a date-key mismatch. Missing seascape assignments occur primarily where one or more dynamic predictors are unavailable, especially chlorophyll-related fields over land or masked coastal cells, SSH over land or masked near-coast cells, and wind fields in the outer buffer. Gradient predictors can further increase missingness because they require valid neighboring-cell context.

Report-ready wording:

> Environmental seascape assignments were restricted to H3-day cells with complete predictor vectors. Unassigned cells occurred primarily in land-masked, coastal, or outer-buffer areas where chlorophyll-a, sea-surface height, wind, or derived gradient fields were unavailable. These exclusions did not reflect date-key mismatches; join-integrity checks confirmed complete alignment between the environmental source table and the model feature grid.

## Candidate Model Ranking Under K = 15 Environmental Blocking

After selecting `environmental_kmeans_k15` as the preferred ecological validation extension, all joint species-use candidate models were evaluated under that split. The same three seascape classes were held out for each model: `seascape_11`, `seascape_6`, and `seascape_5`.

Metrics source:

`Repo/data/modeling/metrics/species_model_environmental_seascape_all_models_kmeans_k15_with_log_block_cv_metrics.csv`

| Model | Log R2 | Log RMSE | Log MAE | Raw R2 | Raw RMSE | Raw MAE | Train rows | Test rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extra_trees | 0.4658 | 0.623 | 0.434 | 0.8313 | 67.921 | 4.907 | 15,201 | 5,169 |
| random_forest | 0.4008 | 0.660 | 0.452 | 0.5262 | 113.821 | 7.341 | 15,201 | 5,169 |
| hist_gradient_boosting | 0.2539 | 0.736 | 0.557 | 0.7302 | 85.898 | 6.604 | 15,201 | 5,169 |
| bayesian_gmm | 0.1235 | 0.798 | 0.583 | 0.0406 | 161.972 | 9.323 | 15,201 | 5,169 |

The Extra Trees model remained the strongest candidate under the selected ecological blocking design, with the highest log-scale R2 and the lowest log-scale RMSE and MAE. This supports retaining Extra Trees as the primary species-use prediction model while using K = 15 environmental seascape blocking as the main transferability diagnostic.

## Risk Product Definitions for Weekly Operator Product

Last updated: 2026-05-11 18:52:25 PDT

Definitions to preserve for Methods, Results, and Discussion:

- **Latent risk** is the operational planning layer. It represents where risk would emerge if fishing activity occurred, using species-use predictions plus the fixed minimum-effort assumption. This is the appropriate week-by-week vessel-operator product because it is exposure-independent and supports forward-looking planning.
- **Latent plausible risk** is a scientific filtering or robustness layer. It constrains latent risk by environmental plausibility and is useful for evaluating whether predicted risk lies inside environmentally credible species-use space. It is not the main operator-facing product.
- **Realized risk** is the retrospective overlap layer. It combines species use with observed fishing activity and answers where fishing and predicted species use actually overlapped.

Report-ready wording:

> Latent risk was treated as the operational planning product because it identifies where risk would emerge under a minimum-effort exposure assumption, independent of the realized fishing distribution. Plausibility-filtered latent risk was retained as a scientific sensitivity layer that restricts latent risk to environmentally credible species-use space, while realized risk was interpreted as the retrospective overlap between predicted species use and observed fishing activity.

Advisor-recommended product:

- Week-of-year latent-risk climatology averaged across 2014-2023 for each species and H3 cell.
- Animated weekly latent-risk sequence for 2022, with 52 frames per species.
- Paper figure with four representative ISO weeks per species as small multiples.

Rationale:

The weekly latent-risk product better matches vessel-operator planning horizons than annual aggregate maps. It also demonstrates that the riskscape is dynamic on the daily H3 substrate rather than a static habitat-suitability surface.

## Plausibility Definition in the Hybrid Model

Last updated: 2026-05-11 23:15:03 PDT

Clarified interpretation:

Plausibility measures whether an H3/day environmental state lies within the set of environmental regimes supported by observed species use. It is species-specific, but it is not an estimate of residence intensity. The residence-index magnitude is modeled separately by the species-use model.

The Bayesian/Gaussian mixture components are environmental regimes: clusters of similar environmental feature vectors. Species observations identify a set of regimes that are environmentally supported for each species. Plausibility then evaluates whether a new H3/day is inside, near, or compatible with that species-supported regime set. It does not compare the cell only to the single most common regime, and it does not require the exact H3/day cell to have been observed.

Conceptual split:

- **Environmental components**: species-independent environmental regimes.
- **Plausibility**: species-specific environmental support within or near the supported regime set.
- **Species-use prediction**: expected log-transformed residence index.
- **Hybrid gated prediction**: species-use prediction adjusted by the environmental support/plausibility layer.

Report-ready wording:

> Plausibility was defined as species-specific environmental support, measuring whether an H3/day feature vector fell within the set of environmental regimes associated with observed species use. It did not estimate residence intensity directly; predicted residence magnitude was modeled separately by the Extra Trees species-use model. Thus, plausibility acted as an environmental support layer over the prediction space rather than as a substitute for species-use prediction.
