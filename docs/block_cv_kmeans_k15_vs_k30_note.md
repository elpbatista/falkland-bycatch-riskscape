# BlockCV k-means k=15 vs k=30 diagnostic note

Generated: 2026-05-12 07:42:50

This note compares k-means environmental blocking options using saved seascape class summaries and the Extra Trees BlockCV metric outputs. The purpose is to choose a validation-blocking scheme, not to maximize seascape-map detail.

| k  | classes | extra_trees_r2 | extra_trees_rmse | extra_trees_mae | train_rows | test_rows | test_fraction_actual | heldout_classes | min_class_h3_day_rows | median_class_h3_day_rows | max_class_h3_day_rows | class_size_cv | max_class_share |
| -- | ------- | -------------- | ---------------- | --------------- | ---------- | --------- | -------------------- | --------------- | --------------------- | ------------------------ | --------------------- | ------------- | --------------- |
| 15 | 15      | 0.8313         | 67.9207          | 4.9066          | 15201      | 5169      | 0.2538               | 3               | 1557968               | 8795578                  | 12721529              | 0.3086        | 0.0986          |
| 30 | 30      | 0.7348         | 62.6341          | 3.9134          | 14076      | 6294      | 0.3090               | 10              | 889595                | 4403508                  | 7193801               | 0.3914        | 0.0558          |

Interpretation: k=15 gives stronger held-out Extra Trees performance than k=30 in the saved environmental BlockCV run, with fewer held-out environmental classes needed to reach the target test fraction. k=30 increases environmental fragmentation and produced lower R2 in the corresponding saved validation metric. This supports k=15 as the primary environmental blocking choice, while k=30 remains useful for finer Bayesian/GMM regime and plausibility diagnostics.
