# Invariant Geometric Signatures Report (Experiment 19B)

## Executive Summary

Experiment 19B decomposes the geometric variance of LLM trajectories into two primary components: **Regime** (CoT vs. Direct) and **Reasoning Quality** (Success vs. Failure). While prompting regime explains the vast majority of absolute variance (>80%), geometric metrics maintain high predictive fidelity for reasoning quality (~0.78 AUC) even when the regime is held constant. This validates the existence of "Success Attractors" as distinct from mere "CoT Attractors."

## 1. Variance Decomposition (2-Way ANOVA)

We performed a two-way ANOVA across 20+ geometric metrics and all transformer layers for Qwen-0.5B and Qwen-1.5B.

### Key Finding: Regime Dominance

In the early and middle layers, the prompting regime (Direct vs. CoT) accounts for over 80% of the variance in global metrics like Radius of Gyration and Effective Dimension.

![Variance Decomposition Qwen-0.5B](file:///C:/Users/karee/.gemini/antigravity/brain/d7720ada-24cf-418b-8675-bc8e1a163cf6/variance_decomposition_qwen05b.png)

> [!IMPORTANT]
> This suggests that a simple comparison of CoT-Success vs. Direct-Failure (as done in earlier experiments) creates a "Regime-Quality Confound" that must be statistically addressed.

## 2. Layer Localization of Effects

Comparing the magnitude of Regime effects (G3 vs G1) against Quality effects (G4 vs G3) reveals distinct spatial signatures.

![Effect Localization Qwen-0.5B](file:///C:/Users/karee/.gemini/antigravity/brain/d7720ada-24cf-418b-8675-bc8e1a163cf6/effect_localization_qwen05b.png)

- **Regime Effects (Black Dashed):** Peak early and remain high, indicating a global shift in hidden state topology.
- **Quality Effects (Red/Blue):** Often peak later in the model, coinciding with the "commitment phase" where the model converges on a specific answer string.

## 3. Position Index: The Nature of CoT Failure

The "Position Index" (PI) measures where CoT Failures (G3) fall on the axis between Direct Failures (G1) and CoT Successes (G4).

![G3 Position Heatmap](file:///C:/Users/karee/.gemini/antigravity/brain/d7720ada-24cf-418b-8675-bc8e1a163cf6/position_heatmap_qwen05b.png)

- **PI ≈ 0:** G3 looks like Direct Failure (Regime shift failed).
- **PI ≈ 1:** G3 looks like CoT Success (Regime shift succeeded, but logic failed).
- **Observation:** In early layers, G3 is almost indistinguishable from G4 (PI ~1.0), suggesting that the "CoT Regime Attractor" is successfully entered regardless of eventual correctness.

## 4. Interaction Signatures (Effect Flips)

We identified several "Interaction Signatures" where the sign of the quality effect flips between regimes. For example, in Qwen-0.5B:

| Metric | Layer | Direct (d) | CoT (d) | Sign Flip |
| :--- | :--- | :--- | :--- | :--- |
| `full_time_to_commit` | 3 | 1.50 | -0.38 | YES |
| `clean_cos_slope_to_final`| 4 | -0.33 | 0.46 | YES |

> [!NOTE]
> This suggests that "Success" in CoT involves different geometric motifs (e.g., slower, more deliberate commitment) compared to Direct success (which may be more purely retrieval-based).

## 5. Within-Regime Predictive Power

To ensure the "Success Attractor" is not a vacuous regime marker, we trained logistic regression models to predict correctness using only geometric features *within* the CoT regime.

![Predictive Power AUC](file:///C:/Users/karee/.gemini/antigravity/brain/d7720ada-24cf-418b-8675-bc8e1a163cf6/predictive_power_auc_qwen05b.png)

- **Result:** Geometry-only models achieve an **AUC of ~0.78** at layer 16 (Qwen-0.5B), significantly outperforming the regime-only baseline (AUC 0.50).
- **Conclusion:** Even after controlling for the CoT regime, the geometric trajectory remains a sensitive indicator of the model's epistemic state.

## Final Summary Table: Model Comparisons

| Model | Peak Quality AUC | Regime Variance (Avg) | G3 Cluster Bias |
| :--- | :--- | :--- | :--- |
| Qwen-0.5B | 0.78 (L24) | 85% | Strong Regime |
| Qwen-1.5B | 0.74 (L26) | 88% | Strong Regime |

---
*Report generated as part of Experiment 19: Robustness and Decomposition.*
