# Experiment 18: Statistical Analysis Report

## 1. Overview
Analyzed 29 metrics across 25 layers for groups G1-G4.

## 2. Top Metric-Layer Discriminators (by Cohen's d)

| Comparison | Layer | Metric | Cohen's d | p-value | Mean G1 | Mean G2 |
|:---|:---|:---|:---|:---|:---|:---|
| G2 vs G4 | 18 | dir_autocorr_lag2 | -7.510 | 1.07e-45 | -0.228 | -0.130 |
| G2 vs G4 | 7 | cosine_to_running_mean | -7.483 | 2.36e-33 | -0.111 | -0.040 |
| G2 vs G4 | 15 | cosine_to_running_mean | -7.453 | 5.93e-33 | -0.110 | -0.039 |
| G2 vs G4 | 13 | cosine_to_running_mean | -7.392 | 1.23e-32 | -0.111 | -0.040 |
| G2 vs G4 | 14 | cosine_to_running_mean | -7.359 | 6.60e-33 | -0.108 | -0.039 |
| G2 vs G4 | 12 | cosine_to_running_mean | -7.096 | 4.40e-32 | -0.110 | -0.041 |
| G2 vs G4 | 6 | cosine_to_running_mean | -7.031 | 2.83e-32 | -0.107 | -0.041 |
| G1 vs G4 | 0 | directional_consistency | 6.975 | 5.36e-104 | 0.410 | 0.033 |
| G2 vs G4 | 19 | dir_autocorr_lag2 | -6.974 | 4.34e-43 | -0.224 | -0.126 |
| G2 vs G4 | 17 | dir_autocorr_lag2 | -6.968 | 2.21e-38 | -0.229 | -0.129 |
| G2 vs G4 | 16 | cosine_to_running_mean | -6.951 | 6.75e-32 | -0.109 | -0.040 |
| G2 vs G4 | 17 | cosine_to_running_mean | -6.659 | 2.28e-30 | -0.122 | -0.043 |
| G2 vs G4 | 8 | cosine_to_running_mean | -6.613 | 1.95e-30 | -0.111 | -0.042 |
| G1 vs G4 | 17 | mean_attractor_dist | 6.548 | 1.71e-243 | 27.621 | 16.001 |
| G2 vs G4 | 1 | cosine_to_running_mean | -6.471 | 4.59e-29 | -0.110 | -0.041 |
| G2 vs G4 | 5 | cosine_to_running_mean | -6.438 | 2.19e-29 | -0.107 | -0.042 |
| G1 vs G4 | 18 | mean_attractor_dist | 6.343 | 5.92e-239 | 31.536 | 18.314 |
| G2 vs G4 | 11 | cosine_to_running_mean | -6.275 | 2.55e-29 | -0.104 | -0.042 |
| G1 vs G4 | 19 | mean_attractor_dist | 6.137 | 1.20e-235 | 35.216 | 20.727 |
| G2 vs G4 | 0 | radius_of_gyration | -6.055 | 1.13e-31 | 0.353 | 0.406 |

## 3. Layer-Wise Summary: Best Separation Layers
Across all comparisons, which layers show the highest mean absolute effect size?

| Layer | Avg |abs(Cohen's d)| |
|:---|:---|
| 17 | 1.386 |
| 18 | 1.374 |
| 19 | 1.371 |
| 14 | 1.328 |
| 16 | 1.327 |
| 15 | 1.318 |
| 12 | 1.283 |
| 7 | 1.274 |
| 9 | 1.273 |
| 10 | 1.270 |

## 4. Specific Regime Insights

### CoT Success (G4) vs CoT Failure (G3)
| Layer | Metric | Cohen's d | Description |
|:---|:---|:---|:---|
| 16 | distance_slope | 1.533 | |
| 16 | cosine_slope | -1.425 | |
| 23 | tortuosity | -1.411 | |
| 23 | mean_attractor_dist | 1.391 | |
| 17 | mean_attractor_dist | 1.375 | |
| 15 | distance_slope | 1.371 | |
| 16 | mean_attractor_dist | 1.361 | |
| 20 | mean_attractor_dist | 1.350 | |
| 15 | mean_attractor_dist | 1.349 | |
| 18 | cosine_slope | -1.349 | |

### Direct Success (G2) vs CoT Success (G4)
| Layer | Metric | Cohen's d | Description |
|:---|:---|:---|:---|
| 18 | dir_autocorr_lag2 | -7.510 | |
| 7 | cosine_to_running_mean | -7.483 | |
| 15 | cosine_to_running_mean | -7.453 | |
| 13 | cosine_to_running_mean | -7.392 | |
| 14 | cosine_to_running_mean | -7.359 | |
| 12 | cosine_to_running_mean | -7.096 | |
| 6 | cosine_to_running_mean | -7.031 | |
| 19 | dir_autocorr_lag2 | -6.974 | |
| 17 | dir_autocorr_lag2 | -6.968 | |
| 16 | cosine_to_running_mean | -6.951 | |
