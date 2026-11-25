# Results

Mean over tasks, with bootstrap 95% CI in brackets. Failures (no code, compile error, timeout) count as similarity 0. pass@1 = similarity >= 0.85.

| model | split | n | compile % | mean sim | pass@1 % | mean reward |
|---|---|---:|---:|---:|---:|---:|
| reference | test_unseen_params | 20 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] |
| reference | test_unseen_combo | 20 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] |
| reference | test_new_family | 150 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] |
| reference | test_iid | 60 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] |
