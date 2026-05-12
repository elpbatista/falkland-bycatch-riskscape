# Bayesian/GMM Component-Count Selection

This note documents the selected component-count sweep for the Bayesian/Gaussian mixture environmental gate used in the hybrid species-use workflow. The goal was not to choose the component count that maximizes the species-use regression score alone, but to select a scientifically defensible environmental partition that preserves heterogeneous ocean states while remaining interpretable.

## Candidate Range

The sweep used six candidate component counts: 8, 10, 15, 18, 30, and 33.

These values were selected to span the range used in related dynamic seascape and environmental classification frameworks. The useful pattern from the seascape literature is that there is no universal best number of classes. Seascape studies usually choose a hierarchical level that balances ecological interpretability, likelihood or classification detail, and fragmentation.

- Kavanaugh et al. (2014) used hierarchical dynamic seascapes in the North Pacific and found that eight seascapes nested within three superseascapes captured substantial biogeochemical variance. This supports a coarse, stable regional classification level.
- NOAA/MBON global seascapes use 33 potential seascape categories generated from satellite/model variables and a probabilistic self-organizing map plus hierarchical agglomerative classification. This motivates the high-complexity end of the sweep.
- Montes et al. (2020) followed the Kavanaugh seascape approach for the Florida Keys. They started from a 15 x 15 probabilistic self-organizing map with 225 neuronal classes, then interpreted dominant seascape classes locally, especially classes around 10-18. This supports the regional middle of the sweep.
- A recent Eastern Tropical Pacific seascape study reported 15 recognized seascapes from physicochemical variables, further supporting the 15-class regional candidate.

For this project, the interpretation is:

- Eight components are consistent with a coarse, stable regional seascape level.
- Fifteen components are strongly defensible as a finer regional classification level.
- Thirty or more components are closer to global MBON-style class granularity, but in this smaller regional domain they may become over-partitioned unless explicitly interpreted as fine environmental micro-regimes.

References and source links:

- Kavanaugh, M. T., Hales, B., Saraceno, M., Spitz, Y. H., White, A. E., and Letelier, R. M. 2014. Hierarchical and dynamic seascapes: A quantitative framework for scaling pelagic biogeochemistry and ecology. Existing bibliography key: `kavanaughHierarchicalDynamicSeascapes2014`.
- NOAA CoastWatch / MBON seascape pelagic habitat classification: <https://oceanwatch.noaa.gov/cwn/products/seascape-pelagic-habitat-classification.html>
- MBON `seascapeR` 33-class seascape table: <https://marinebon.github.io/seascapeR/articles/seascapeR.html>
- Montes et al. 2020. Dynamic satellite seascapes as a biogeographic framework for understanding phytoplankton assemblages in the Florida Keys National Marine Sanctuary. Frontiers in Marine Science: <https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2020.00575/full>
- Eastern Tropical Pacific seascape classification study. Marine Environmental Research 2025: <https://www.sciencedirect.com/science/article/pii/S0141113625003514>

## Sweep Configuration

The sweep used the same joint Bayesian/GMM species-use model structure as the existing workflow, with a 12% random holdout to match the revised model-evaluation setting.

Command:

```bash
PYTHONPATH=Repo/src python3 Repo/scripts/compare_bayesian_gmm_components.py \
  --test-fraction 0.12 \
  --components 8,10,15,18,30,33 \
  --sweep-name bayesian_gmm_component_sweep_random12_selected \
  --metrics-path Repo/data/modeling/metrics/bayesian_gmm_component_comparison_random12_selected.csv
```

Metrics were saved to:

`Repo/data/modeling/metrics/bayesian_gmm_component_comparison_random12_selected.csv`

## Results

| Components | R2 | RMSE | MAE | Mean log likelihood | BIC | AIC | Min component share | <1% components | <2% components | Top-10 capture | Top-5 capture | Top-1 capture |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.1355 | 33.06 | 4.27 | 10.88 | -175032 | -190705 | 3.03% | 0 | 0 | 0.7387 | 0.7241 | 0.7479 |
| 10 | 0.1199 | 33.36 | 4.07 | 12.81 | -204737 | -224330 | 3.04% | 0 | 0 | 0.7368 | 0.7144 | 0.7492 |
| 15 | 0.0999 | 33.74 | 3.94 | 16.30 | -254642 | -284035 | 1.80% | 0 | 1 | 0.7133 | 0.6675 | 0.7479 |
| 18 | 0.1002 | 33.73 | 3.82 | 17.11 | -261620 | -296893 | 0.50% | 1 | 3 | 0.7201 | 0.6545 | 0.7479 |
| 30 | 0.1203 | 33.35 | 3.52 | 21.96 | -318452 | -377245 | 0.35% | 1 | 6 | 0.7408 | 0.6766 | 0.7479 |
| 33 | 0.1144 | 33.46 | 3.39 | 22.35 | -317932 | -382605 | 0.37% | 1 | 10 | 0.7295 | 0.6958 | 0.7479 |

## Interpretation

The eight-component model had the best regression R2 and RMSE, but this model represents a relatively coarse environmental partition. It is useful as a low-complexity benchmark, but it is less aligned with the purpose of the Bayesian/GMM component layer in the hybrid workflow, which is to represent environmental plausibility and distinguish recurring environmental states.

The 15- and 18-component models occupy the regional middle ground. They improve environmental likelihood relative to 8 and 10 components, but their regression metrics and hotspot-capture diagnostics were weaker than the 30-component model. The 18-component model also began to show stronger fragmentation, with one component below 1% of positive training records and three components below 2%.

The 33-component model was included because it is closest to the NOAA/MBON global dynamic seascape framework. It produced the highest mean log likelihood and lowest MAE, but it also produced the strongest fragmentation among the selected candidates, with ten components representing less than 2% of positive training records. It also had weaker BIC, R2, RMSE, and top-10 hotspot capture than the 30-component model.

## Final Decision

The selected model is the 30-component Bayesian/GMM.

This choice balances the main scientific and operational goals of the component layer:

- It retains a fine environmental partition close to the NOAA/MBON 33-class seascape framework.
- It has the best BIC among the selected candidates, indicating the strongest likelihood-complexity tradeoff in this sweep.
- It improves MAE substantially relative to lower component counts.
- It has better R2 and RMSE than the 33-component model.
- It has the strongest top-10 hotspot capture among the selected candidates.
- It accepts moderate fragmentation as a cost of preserving environmental heterogeneity and rare/high-use conditions.

The 30-component model should therefore be treated as the revised fine-scale Bayesian/GMM environmental component model. The 8-component result remains useful as a coarse benchmark, and the 33-component result can be referenced as a NOAA/MBON-aligned sensitivity point, but neither is selected as the primary component count for the revised workflow.
