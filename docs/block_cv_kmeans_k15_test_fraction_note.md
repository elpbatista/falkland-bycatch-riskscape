# BlockCV k-means k=15 target test-fraction note

Generated: 2026-05-12 07:52:55

Changing the requested environmental BlockCV holdout from 12% to 10% did not change the realized split. Because environmental blocking holds out whole k-means k=15 seascape classes, the same shuffled class sequence is selected until the target is reached. In this case, the same three classes are needed to pass both targets, so the actual holdout remains 25.38%.

| Scenario | Requested | Actual | Held-out groups | Train rows | Test rows | R2 | RMSE | MAE |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| k15 target 12% | 0.12 | 0.2538 | seascape_11,seascape_6,seascape_5 | 15201 | 5169 | 0.8313 | 67.9207 | 4.9066 |
| k15 target 10% | 0.10 | 0.2538 | seascape_11,seascape_6,seascape_5 | 15201 | 5169 | 0.8313 | 67.9207 | 4.9066 |
