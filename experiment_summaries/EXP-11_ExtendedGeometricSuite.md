# Experiment 11: Extended Geometric Suite

**Phase:** 3 — The Pivot to Dynamics & Intervention
**Date:** November 2025
**Model:** Qwen2.5-0.5B (using EXP-09 dataset)
**Status:** Completed — **SUCCESS**

## Connection to Prior Work

EXP-09 established Speed and Curvature as valid predictors of success, and EXP-10 proved that models cannot self-report their geometric state. We needed richer, more fundamental topological metrics to serve as external diagnostics.

## Research Question

**Are there deeper topological properties (dimensionality, path efficiency) that characterize the trajectory manifold beyond speed and curvature?**

## Hypotheses
1.  **H1 (Dimensionality of Truth):** Successful reasoning (G4) will utilize a higher-dimensional subspace of the residual stream than failed/reflexive answers (G1).
2.  **H2 (Path Tortuosity):** Correct reasoning is not just "curved" but involves a non-efficient path (winding) compared to the "ballistic" straight line of a hallucination.

## Method Summary
*   **Dataset:** 300 multi-step arithmetic problems (Exp 9 benchmark).
*   **Metrics Introduced:**
    1.  **Effective Dimension ($D_{eff}$):** PCA participation ratio of the hidden states across tokens. Measures the "richness" of the subspace used.
    2.  **Tortuosity ($\tau$):** The ratio of end-to-end distance to total arc length. A measure of path efficiency.
    3.  **Turning Angle:** Mean angular change between consecutive step deltas.
    4.  **Directional Autocorrelation:** The degree to which one "thought step" predicts the direction of the next.

## Key Results & Analysis
### 1. The "Dimensional Collapse" Discovery
We found a massive, statistically significant distinction in the **Effective Dimension** ($D_{eff}$).
*   **G4 (CoT Success):** $D_{eff} \approx 13.1$
*   **G1 (Direct Fail):** $D_{eff} \approx 3.4$
*   **Effect Size:** Cohen's $d > 4.5$.
*   **Meaning:** Successful reasoning is a high-dimensional process. Failure (specifically "Direct" failure) is geometrically "flat"—it collapses into a narrow subspace, likely indicating a "pattern matching" mode rather than execution.

### 2. The "Effort" of the Regime
Even *failed* CoT (G3) maintained a high dimensionality ($D_{eff} \approx 13.9$).
*   **Insight:** The *act* of Chain-of-Thought prompting forces the model into high-dimensional space regardless of outcome. It "engages the engine," even if the internal steering is wrong.

### 3. Path Winding (Tortuosity)
*   **G4 (Success):** Very low Tortuosity ($\tau \approx 0.04$). The path is extremely long relative to its displacement. It "winds" through semantic space.
*   **G1 (Direct Fail):** High Tortuosity ($\tau \approx 0.40$). The trajectory is relatively straight and direct towards the (wrong) answer.

## Methodological Issues / Limitations
*   **Metric Redundancy:** Some metrics (like Turning Angle and Directional Consistency) were highly correlated ($r > 0.9$), suggesting we only need one to capture "curvature."
*   **Window Sensitivity:** $D_{eff}$ is sensitive to the number of tokens analyzed. Comparing a 32-token CoT to a 5-token Direct answer requires careful normalization.

## Conclusions & Implications

**Verdict: SUCCESS.** CoT is not just "extra compute" — it is **"Dimensional Expansion."** CoT allows the model to "unfold" the problem into more dimensions where it can be resolved. $D_{eff}$ is a stronger predictor of the "Reasoning vs Retrieval" regime than Speed.

## Influence on Next Experiment

*   The discovery of Dimensional Collapse prompted **EXP-12** to investigate the *intrinsic* properties and *temporal evolution* of trajectory manifolds — moving from bulk averages to dynamic texture (fractal dimension, convergence profiles).
