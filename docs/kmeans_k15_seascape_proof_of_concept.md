# k-means k15 Seascape Proof of Concept

This note preserves the development history for the earlier k-means k=15
seascape workflow. These details are useful for repository reconstruction and
audit trails, but they should not drive the final manuscript argument.

## Purpose

The k-means k=15 seascape layer was used as a pragmatic proof of concept for:

- environmental seascape-based BlockCV,
- seascape-conditioned species-use/risk aggregation,
- monthly seascape-risk matrix plotting,
- checking whether seascape classes could support a coarse risk surrogate.

It was later replaced as the data-informed validation design by:

```text
SOM-hierarchical k=30 + 5-fold grouped environmental BlockCV
```

## Original Evidence

Source:

```text
Repo/data/modeling/metrics/species_model_block_cv_variant_comparison.csv
```

Relevant rows:

| validation variant | validation design              | key result                           |
|--------------------|--------------------------------|--------------------------------------|
| kmeans_k15         | k-means k=15 seascape holdout  | r2 = 0.831, r2_log = 0.466           |
| kmeans_k15_5fold   | k-means k=15 grouped 5-fold CV | r2_mean = 0.299, r2_log_mean = 0.491 |

Interpretation:

The k-means k=15 layer performed better than the tested GMM-component blocking
alternative and helped demonstrate that environmental seascape blocking was
viable. However, it was a provisional direct k-means classification rather than
a final scientifically selected seascape framework.

## Original Seascape-Risk Product

Data product:

```text
Repo/data/modeling/predictions/seascape_kmeans_k15_predicted_q90
```

Accepted proof-of-concept matrices:

```text
Repo/plots/predictions/seascape_kmeans_k15_predicted_q90_joint_latent_risk_log_pred_non_zero_mean_BBAL_2022_monthly_matrix.png
Repo/plots/predictions/seascape_kmeans_k15_predicted_q90_joint_latent_risk_log_pred_non_zero_mean_SAFS_2022_monthly_matrix.png
```

These maps used the shared binned latent-risk matrix style:

```bash
--color-bin-source monthly_species \
--color-quantiles 0 0.40 0.75 0.95 1.0
```

## Current Status

The k-means k=15 work should be treated as development history and proof of
concept. The current manuscript direction should focus on SOM-hierarchical k=30
with 5-fold grouped environmental BlockCV.

Do not frame k-means k=15 as the final selected validation design in the
manuscript.
