# Experiment 10: Self-Report Consistency Findings

## 1. Correlations with Trajectory Geometry

| Self-Report | Metric | Pearson r (Imm) | p | Pearson r (Pert) | Retention | Result |
|---|---|---|---|---|---|---|
| Effort | Speed | -0.000 | 0.999 | 0.050 | 620.89 |  |
| Certainty | Stabilization | -0.009 | 0.933 | -0.000 | 0.04 |  |
| Exploration | Neg_DirCons | 0.187 | 0.062 | -0.107 | 0.57 |  |
| Smoothness | Neg_Curvature | 0.173 | 0.085 | -0.134 | 0.78 |  |

## 2. Predicting Success from Self-Report

Success Rate: 71/100

| Model | AUC | Balanced Acc | Macro F1 |
|---|---|---|---|
| Self-Report Ratings Only | 0.636 | 0.635 | 0.624 |
| Response Length Baseline | 0.663 | 0.633 | 0.636 |
| Random Baseline | 0.593 | 0.589 | 0.563 |

## Conclusion

FAIL/INCONCLUSIVE: No strong correlations found.