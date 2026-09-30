# M1 Reproducible Baseline

- Experiment: m1-reproducible-baseline-v1
- Training seeds: [7, 17, 29, 43, 59]
- Selection config: evaluation_selection
- Test config: evaluation_test

Binary and clearance intervals are 95% scenario-stratified bootstrap intervals. Wilson intervals are also available in summary.csv.

| Policy | Checkpoint | Shield | Episodes | Success 95% CI | Collision 95% CI | Timeout 95% CI | Mean clearance 95% CI (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| bc | fixed | False | 160 | 83.1% [80.0, 85.6] | 16.9% [14.4, 20.0] | 0.0% [0.0, 0.0] | 0.206 [0.192, 0.219] |
| bc | fixed | True | 160 | 77.5% [74.4, 80.6] | 12.5% [12.5, 12.5] | 10.0% [6.9, 13.1] | 0.211 [0.199, 0.223] |
| custom_stable_ppo | best | False | 800 | 85.4% [84.0, 86.5] | 14.6% [13.5, 16.0] | 0.0% [0.0, 0.0] | 0.208 [0.197, 0.218] |
| custom_stable_ppo | best | True | 800 | 83.0% [79.8, 86.1] | 12.5% [12.5, 12.5] | 4.5% [1.2, 7.8] | 0.210 [0.202, 0.220] |
| custom_stable_ppo | final | False | 800 | 77.2% [69.9, 84.2] | 22.8% [15.9, 29.8] | 0.0% [0.0, 0.0] | 0.194 [0.180, 0.209] |
| custom_stable_ppo | final | True | 800 | 79.9% [73.6, 86.0] | 12.5% [12.5, 12.5] | 7.6% [1.6, 13.6] | 0.201 [0.191, 0.212] |
| legacy_stable_ppo | legacy_best | False | 160 | 68.8% [63.7, 73.1] | 31.2% [26.9, 35.6] | 0.0% [0.0, 0.0] | 0.173 [0.162, 0.185] |
| legacy_stable_ppo | legacy_best | True | 160 | 72.5% [68.8, 76.2] | 12.5% [12.5, 12.5] | 15.0% [11.2, 18.8] | 0.184 [0.174, 0.194] |
| lite_rule_baseline | fixed | False | 160 | 86.2% [84.4, 87.5] | 13.8% [12.5, 15.6] | 0.0% [0.0, 0.0] | 0.202 [0.192, 0.213] |
| lite_rule_baseline | fixed | True | 160 | 87.5% [87.5, 87.5] | 12.5% [12.5, 12.5] | 0.0% [0.0, 0.0] | 0.204 [0.195, 0.214] |
