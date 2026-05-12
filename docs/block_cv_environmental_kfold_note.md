# Environmental k-fold BlockCV note

Generated: 2026-05-12 08:07:18

## Purpose

This note documents the revised environmental BlockCV validation strategy for species-use models. Instead of a single random row-level split, complete k-means k=15 environmental seascape classes are grouped into five folds. Each fold withholds complete environmental classes for testing and trains on the remaining classes, so validation evaluates transferability across environmental regimes rather than interpolation among randomly mixed rows.

## Fold construction

Rows were first assigned to k-means k=15 seascape classes through the compact environmental-regime table keyed by H3 cell and UTC date. Fold assignment used complete seascape classes and balanced three quantities: total rows, BBAL positive-use support, and SAFS positive-use support. This avoids the earlier issue where a fold could contain no positive BBAL test cases.

## Fold metrics

| cv_fold | heldout_groups                      | train_rows | test_rows | actual_test_fraction | r2      | rmse    | mae    | r2_log | rmse_log | mae_log |
| ------- | ----------------------------------- | ---------- | --------- | -------------------- | ------- | ------- | ------ | ------ | -------- | ------- |
| 1       | seascape_4,seascape_5,seascape_9    | 15349      | 5021      | 0.2465               | 0.8330  | 68.5643 | 5.1196 | 0.4832 | 0.6115   | 0.4166  |
| 2       | seascape_10,seascape_2,seascape_7   | 15622      | 4748      | 0.2331               | 0.9094  | 37.1353 | 2.1168 | 0.5498 | 0.4835   | 0.3090  |
| 3       | seascape_1,seascape_13,seascape_8   | 16568      | 3802      | 0.1866               | -1.6224 | 28.3070 | 2.7798 | 0.4540 | 0.6107   | 0.4313  |
| 4       | seascape_11,seascape_12,seascape_14 | 17475      | 2895      | 0.1421               | 0.7598  | 83.8339 | 4.5697 | 0.5256 | 0.5678   | 0.4135  |
| 5       | seascape_0,seascape_3,seascape_6    | 16466      | 3904      | 0.1917               | 0.6166  | 22.8884 | 2.0856 | 0.4441 | 0.5583   | 0.3824  |

## Test-fold species support

| cv_fold | species | rows | positive_rows | positive_fraction | target_mean | target_max |
| ------- | ------- | ---- | ------------- | ----------------- | ----------- | ---------- |
| 1       | BBAL    | 2009 | 1764          | 0.8780            | 16.1936     | 6852.0000  |
| 1       | SAFS    | 3012 | 1101          | 0.3655            | 4.6720      | 2002.0000  |
| 2       | BBAL    | 1516 | 1331          | 0.8780            | 10.9301     | 6296.0000  |
| 2       | SAFS    | 3232 | 952           | 0.2946            | 0.9127      | 28.0000    |
| 3       | BBAL    | 1179 | 982           | 0.8329            | 5.5411      | 894.0000   |
| 3       | SAFS    | 2623 | 1132          | 0.4316            | 1.4674      | 270.0000   |
| 4       | BBAL    | 463  | 315           | 0.6803            | 20.1015     | 8217.0000  |
| 4       | SAFS    | 2432 | 1123          | 0.4618            | 5.7196      | 2100.0000  |
| 5       | BBAL    | 341  | 101           | 0.2962            | 1.3724      | 29.0000    |
| 5       | SAFS    | 3563 | 1384          | 0.3884            | 3.0269      | 1236.0000  |

## Mean and standard deviation across folds

| model       | model_type    | species | cv_folds | r2_mean | r2_std | rmse_mean | rmse_std | mae_mean | mae_std | r2_log_mean | r2_log_std | rmse_log_mean | rmse_log_std | mae_log_mean | mae_log_std | actual_test_fraction_mean | actual_test_fraction_std |
| ----------- | ------------- | ------- | -------- | ------- | ------ | --------- | -------- | -------- | ------- | ----------- | ---------- | ------------- | ------------ | ------------ | ----------- | ------------------------- | ------------------------ |
| extra_trees | joint_species | all     | 5        | 0.2993  | 1.0796 | 48.1458   | 26.6613  | 3.3343   | 1.4197  | 0.4914      | 0.0455     | 0.5663        | 0.0523       | 0.3906       | 0.0490      | 0.2000                    | 0.0414                   |

## Interpretation for report

The five-fold environmental BlockCV result is more conservative and more scientifically defensible than a single random train/test split because each test fold represents environmental regimes withheld from training. Test fractions vary by fold because complete seascape classes are withheld. All final folds include positive BBAL and SAFS support in the test partition. Raw-scale R2 is variable across folds because residence-index magnitude and variance differ strongly among withheld environmental regimes; log-scale metrics are more stable and align with the log-transformed target used by the model. These diagnostics support using k-means k=15 environmental folds as the validation framework while training the final production model on all eligible training rows after validation.

## Modeling scale note

The species-use models are trained in log space using `log1p(residence_index)` as the target. Reported log-scale validation metrics therefore correspond directly to the fitted objective. Raw-scale metrics are computed after back-transforming predictions with `expm1` and are retained for interpretability, but they are more sensitive to extreme residence-index values and regime-specific variance.
