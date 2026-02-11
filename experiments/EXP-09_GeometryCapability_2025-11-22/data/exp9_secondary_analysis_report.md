# Experiment 9: Secondary Analysis Report

**Generated**: 2026-02-02 14:20
**Model**: Qwen/Qwen2.5-0.5B
**N Problems**: 300

---

## 1. Group Sizes

| Group | Description | N |
|---|---|---|
| G1 | Direct Failure | 247 |
| G2 | Direct Success | 53 |
| G3 | CoT Failure | 77 |
| G4 | CoT Success | 223 |

---

## 2. Primary Comparison: G4 vs G1 (Reference)

| Layer | Metric | G4 Mean | G1 Mean | Cohen's d | p-value |
|---|---|---|---|---|---|
| 0 | speed | 0.5583 | 0.4166 | **2.68** | 0.0000 |
| 0 | dir_consistency | 0.0484 | 0.5087 | **-2.60** | 0.0000 |
| 0 | stabilization | 0.0009 | -0.1190 | **1.59** | 0.0000 |
| 0 | curvature_early | 1.4149 | 1.3254 | **0.81** | 0.0000 |
| 10 | speed | 13.6859 | 12.1395 | **3.11** | 0.0000 |
| 10 | dir_consistency | 0.0530 | 0.5104 | **-2.61** | 0.0000 |
| 10 | stabilization | -0.0177 | -0.6048 | **1.04** | 0.0000 |
| 10 | curvature_early | 1.3469 | 1.2928 | **1.13** | 0.0000 |
| 13 | speed | 14.2805 | 12.4863 | **3.26** | 0.0000 |
| 13 | dir_consistency | 0.0559 | 0.5094 | **-2.59** | 0.0000 |
| 13 | stabilization | -0.0117 | -0.6346 | **1.10** | 0.0000 |
| 13 | curvature_early | 1.3293 | 1.2744 | **1.16** | 0.0000 |
| 16 | speed | 17.2675 | 15.1971 | **2.98** | 0.0000 |
| 16 | dir_consistency | 0.0570 | 0.5182 | **-2.57** | 0.0000 |
| 16 | stabilization | 0.0085 | -0.9328 | **1.15** | 0.0000 |
| 16 | curvature_early | 1.3054 | 1.2495 | **1.09** | 0.0000 |
| 24 | speed | 190.8886 | 130.8658 | **4.25** | 0.0000 |
| 24 | dir_consistency | 0.0524 | 0.5310 | **-2.63** | 0.0000 |
| 24 | stabilization | 0.6636 | -24.0137 | **2.03** | 0.0000 |
| 24 | curvature_early | 1.3259 | 1.1820 | **1.74** | 0.0000 |

---

## 3. Secondary Comparisons

### 3.1 Success Effect Within Prompting Conditions

**G4 vs G3** (CoT Success vs CoT Failure) — Isolates outcome effect within CoT

| Layer | Metric | G4 Mean | G3 Mean | Cohen's d | p-value | Sig? |
|---|---|---|---|---|---|---|
| 0 | speed | 0.5583 | 0.5568 | 0.10 | 0.4350 |  |
| 0 | dir_consistency | 0.0484 | 0.0427 | 0.73 | 0.0000 | ✓ |
| 0 | stabilization | 0.0009 | 0.0007 | 0.18 | 0.1590 |  |
| 0 | curvature_early | 1.4149 | 1.4173 | -0.08 | 0.5150 |  |
| 10 | speed | 13.6859 | 13.8619 | -0.82 | 0.0000 | ✓ |
| 10 | dir_consistency | 0.0530 | 0.0540 | -0.18 | 0.1710 |  |
| 10 | stabilization | -0.0177 | -0.0122 | -0.36 | 0.0090 |  |
| 10 | curvature_early | 1.3469 | 1.3560 | -0.49 | 0.0000 |  |
| 13 | speed | 14.2805 | 14.4491 | -0.71 | 0.0000 | ✓ |
| 13 | dir_consistency | 0.0559 | 0.0576 | -0.29 | 0.0280 |  |
| 13 | stabilization | -0.0117 | -0.0069 | -0.28 | 0.0360 |  |
| 13 | curvature_early | 1.3293 | 1.3351 | -0.30 | 0.0270 |  |
| 16 | speed | 17.2675 | 17.3132 | -0.19 | 0.1310 |  |
| 16 | dir_consistency | 0.0570 | 0.0594 | -0.37 | 0.0050 |  |
| 16 | stabilization | 0.0085 | 0.0083 | 0.01 | 0.9440 |  |
| 16 | curvature_early | 1.3054 | 1.3112 | -0.25 | 0.0520 |  |
| 24 | speed | 190.8886 | 192.9256 | -0.40 | 0.0070 |  |
| 24 | dir_consistency | 0.0524 | 0.0522 | 0.04 | 0.7900 |  |
| 24 | stabilization | 0.6636 | 0.9975 | -0.63 | 0.0000 | ✓ |
| 24 | curvature_early | 1.3259 | 1.3353 | -0.36 | 0.0060 |  |

**G2 vs G1** (Direct Success vs Direct Failure) — Isolates outcome effect within Direct

> [!NOTE]
> Small N in G2 limits power. Interpret with caution.

| Layer | Metric | G2 Mean | G1 Mean | Cohen's d | p-value | Sig? |
|---|---|---|---|---|---|---|
| 0 | speed | 0.4994 | 0.4166 | 1.26 | 0.0000 | ✓ |
| 0 | dir_consistency | 0.0860 | 0.5087 | -1.89 | 0.0000 | ✓ |
| 0 | stabilization | -0.0039 | -0.1190 | 1.22 | 0.0000 | ✓ |
| 0 | curvature_early | 1.3926 | 1.3254 | 0.49 | 0.0020 |  |
| 10 | speed | 13.0705 | 12.1395 | 1.53 | 0.0000 | ✓ |
| 10 | dir_consistency | 0.0941 | 0.5104 | -1.88 | 0.0000 | ✓ |
| 10 | stabilization | -0.0375 | -0.6048 | 0.80 | 0.0000 | ✓ |
| 10 | curvature_early | 1.3245 | 1.2928 | 0.55 | 0.0010 | ✓ |
| 13 | speed | 13.6836 | 12.4863 | 1.78 | 0.0000 | ✓ |
| 13 | dir_consistency | 0.0952 | 0.5094 | -1.87 | 0.0000 | ✓ |
| 13 | stabilization | -0.0266 | -0.6346 | 0.85 | 0.0000 | ✓ |
| 13 | curvature_early | 1.3099 | 1.2744 | 0.62 | 0.0000 | ✓ |
| 16 | speed | 16.5110 | 15.1971 | 1.52 | 0.0000 | ✓ |
| 16 | dir_consistency | 0.0873 | 0.5182 | -1.90 | 0.0000 | ✓ |
| 16 | stabilization | 0.0484 | -0.9328 | 0.96 | 0.0000 | ✓ |
| 16 | curvature_early | 1.2802 | 1.2495 | 0.50 | 0.0030 |  |
| 24 | speed | 162.1330 | 130.8658 | 1.75 | 0.0000 | ✓ |
| 24 | dir_consistency | 0.0841 | 0.5310 | -1.94 | 0.0000 | ✓ |
| 24 | stabilization | 1.8427 | -24.0137 | 1.69 | 0.0000 | ✓ |
| 24 | curvature_early | 1.3138 | 1.1820 | 1.27 | 0.0000 | ✓ |

### 3.2 Prompting Effect Within Outcome Conditions

**G3 vs G1** (CoT Failure vs Direct Failure) — Isolates prompting effect, both failed

| Layer | Metric | G3 Mean | G1 Mean | Cohen's d | p-value | Sig? |
|---|---|---|---|---|---|---|
| 0 | speed | 0.5568 | 0.4166 | 2.22 | 0.0000 | ✓ |
| 0 | dir_consistency | 0.0427 | 0.5087 | -2.18 | 0.0000 | ✓ |
| 0 | stabilization | 0.0007 | -0.1190 | 1.32 | 0.0000 | ✓ |
| 0 | curvature_early | 1.4173 | 1.3254 | 0.70 | 0.0000 | ✓ |
| 10 | speed | 13.8619 | 12.1395 | 2.93 | 0.0000 | ✓ |
| 10 | dir_consistency | 0.0540 | 0.5104 | -2.16 | 0.0000 | ✓ |
| 10 | stabilization | -0.0122 | -0.6048 | 0.87 | 0.0000 | ✓ |
| 10 | curvature_early | 1.3560 | 1.2928 | 1.13 | 0.0000 | ✓ |
| 13 | speed | 14.4491 | 12.4863 | 3.00 | 0.0000 | ✓ |
| 13 | dir_consistency | 0.0576 | 0.5094 | -2.14 | 0.0000 | ✓ |
| 13 | stabilization | -0.0069 | -0.6346 | 0.92 | 0.0000 | ✓ |
| 13 | curvature_early | 1.3351 | 1.2744 | 1.10 | 0.0000 | ✓ |
| 16 | speed | 17.3132 | 15.1971 | 2.53 | 0.0000 | ✓ |
| 16 | dir_consistency | 0.0594 | 0.5182 | -2.12 | 0.0000 | ✓ |
| 16 | stabilization | 0.0083 | -0.9328 | 0.96 | 0.0000 | ✓ |
| 16 | curvature_early | 1.3112 | 1.2495 | 1.04 | 0.0000 | ✓ |
| 24 | speed | 192.9256 | 130.8658 | 3.68 | 0.0000 | ✓ |
| 24 | dir_consistency | 0.0522 | 0.5310 | -2.18 | 0.0000 | ✓ |
| 24 | stabilization | 0.9975 | -24.0137 | 1.71 | 0.0000 | ✓ |
| 24 | curvature_early | 1.3353 | 1.1820 | 1.56 | 0.0000 | ✓ |

**G4 vs G2** (CoT Success vs Direct Success) — Isolates prompting effect, both succeeded

> [!NOTE]
> Small N in G2 limits power. Interpret with caution.

| Layer | Metric | G4 Mean | G2 Mean | Cohen's d | p-value | Sig? |
|---|---|---|---|---|---|---|
| 0 | speed | 0.5583 | 0.4994 | 3.54 | 0.0000 | ✓ |
| 0 | dir_consistency | 0.0484 | 0.0860 | -1.23 | 0.0000 | ✓ |
| 0 | stabilization | 0.0009 | -0.0039 | 0.78 | 0.0000 | ✓ |
| 0 | curvature_early | 1.4149 | 1.3926 | 0.64 | 0.0000 | ✓ |
| 10 | speed | 13.6859 | 13.0705 | 2.94 | 0.0000 | ✓ |
| 10 | dir_consistency | 0.0530 | 0.0941 | -1.28 | 0.0000 | ✓ |
| 10 | stabilization | -0.0177 | -0.0375 | 0.38 | 0.0190 |  |
| 10 | curvature_early | 1.3469 | 1.3245 | 1.12 | 0.0000 | ✓ |
| 13 | speed | 14.2805 | 13.6836 | 2.61 | 0.0000 | ✓ |
| 13 | dir_consistency | 0.0559 | 0.0952 | -1.20 | 0.0000 | ✓ |
| 13 | stabilization | -0.0117 | -0.0266 | 0.35 | 0.0310 |  |
| 13 | curvature_early | 1.3293 | 1.3099 | 0.98 | 0.0000 | ✓ |
| 16 | speed | 17.2675 | 16.5110 | 3.27 | 0.0000 | ✓ |
| 16 | dir_consistency | 0.0570 | 0.0873 | -0.87 | 0.0000 | ✓ |
| 16 | stabilization | 0.0085 | 0.0484 | -0.65 | 0.0000 | ✓ |
| 16 | curvature_early | 1.3054 | 1.2802 | 1.05 | 0.0000 | ✓ |
| 24 | speed | 190.8886 | 162.1330 | 4.51 | 0.0000 | ✓ |
| 24 | dir_consistency | 0.0524 | 0.0841 | -0.88 | 0.0000 | ✓ |
| 24 | stabilization | 0.6636 | 1.8427 | -0.84 | 0.0000 | ✓ |
| 24 | curvature_early | 1.3259 | 1.3138 | 0.37 | 0.0110 |  |

---

## 4. Robustness Checks

### 4.1 Window Size Sensitivity (16 vs 32 tokens)

Do effects hold with shorter analysis window?

| Layer | Metric | d (w=32) | d (w=16) | Direction Match? |
|---|---|---|---|---|
| 0 | speed | 2.68 | 2.33 | ✓ |
| 0 | dir_consistency | -2.60 | -2.43 | ✓ |
| 0 | stabilization | 1.59 | 1.56 | ✓ |
| 0 | curvature_early | 0.81 | 0.68 | ✓ |
| 10 | speed | 3.11 | 3.12 | ✓ |
| 10 | dir_consistency | -2.61 | -2.51 | ✓ |
| 10 | stabilization | 1.04 | 1.03 | ✓ |
| 10 | curvature_early | 1.13 | 0.71 | ✓ |
| 13 | speed | 3.26 | 3.26 | ✓ |
| 13 | dir_consistency | -2.59 | -2.49 | ✓ |
| 13 | stabilization | 1.10 | 1.04 | ✓ |
| 13 | curvature_early | 1.16 | 0.72 | ✓ |
| 16 | speed | 2.98 | 2.82 | ✓ |
| 16 | dir_consistency | -2.57 | -2.48 | ✓ |
| 16 | stabilization | 1.15 | 1.08 | ✓ |
| 16 | curvature_early | 1.09 | 0.64 | ✓ |
| 24 | speed | 4.25 | 4.14 | ✓ |
| 24 | dir_consistency | -2.63 | -2.65 | ✓ |
| 24 | stabilization | 2.03 | 1.76 | ✓ |
| 24 | curvature_early | 1.74 | 1.58 | ✓ |

### 4.2 Outlier Sensitivity

Do effects survive removal of top/bottom 5% of values?

| Layer | Metric | d (full) | d (trimmed) | Robust? |
|---|---|---|---|---|
| 0 | speed | 2.68 | 3.24 | ✓ |
| 0 | dir_consistency | -2.60 | -2.94 | ✓ |
| 0 | stabilization | 1.59 | 1.88 | ✓ |
| 0 | curvature_early | 0.81 | 1.07 | ✓ |
| 10 | speed | 3.11 | 3.85 | ✓ |
| 10 | dir_consistency | -2.61 | -3.18 | ✓ |
| 10 | stabilization | 1.04 | 1.22 | ✓ |
| 10 | curvature_early | 1.13 | 1.44 | ✓ |
| 13 | speed | 3.26 | 3.99 | ✓ |
| 13 | dir_consistency | -2.59 | -2.93 | ✓ |
| 13 | stabilization | 1.10 | 1.29 | ✓ |
| 13 | curvature_early | 1.16 | 1.47 | ✓ |
| 16 | speed | 2.98 | 3.67 | ✓ |
| 16 | dir_consistency | -2.57 | -2.93 | ✓ |
| 16 | stabilization | 1.15 | 1.34 | ✓ |
| 16 | curvature_early | 1.09 | 1.41 | ✓ |
| 24 | speed | 4.25 | 5.24 | ✓ |
| 24 | dir_consistency | -2.63 | -2.99 | ✓ |
| 24 | stabilization | 2.03 | 2.36 | ✓ |
| 24 | curvature_early | 1.74 | 1.95 | ✓ |

---

## 5. Summary Interpretation

### Effect Decomposition

- **Success effect within CoT (G4 vs G3)**: 4/20 significant
- **Success effect within Direct (G2 vs G1)**: 18/20 significant (low power)
- **Prompting effect (failures: G3 vs G1)**: 20/20 significant
- **Prompting effect (successes: G4 vs G2)**: 17/20 significant (low power)

> **Key Finding**: Prompting effect exists even when both conditions fail, suggesting prompting influences trajectory geometry independent of outcome.

### Robustness Verdict

- **Window sensitivity**: Effects generally consistent across 16 and 32 token windows
- **Outlier sensitivity**: Large effects survive 5% trimming

---

*Report generated by run_exp9_secondary_analysis.py*