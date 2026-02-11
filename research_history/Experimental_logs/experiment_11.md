# Experiment 11: Extended Geometric Suite

**Date / Context:** October 2026 / Phase 3: The Pivot to Dynamics & Intervention
**Model:** Qwen2.5-0.5B (using Experiment 9 dataset)
**Status:** Completed

## Motivation & Prior Assumptions
*   **Context:** Experiment 09 established the "Speed" and "Curvature" signatures as valid predictors of success.
*   **Assumption:** These early metrics, while effective, were proxies for more fundamental topological properties of the trajectory manifold. 
*   **Goal:** Develop a richer, "physics-inspired" suite of metrics to characterize the "texture" of thought, specifically focusing on dimensionality and path efficiency.

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

## Interpretation (Meanings & Implications)
*   **Verdict:** **SUCCESS**.
*   **Theoretical Shift:** We move from viewing CoT as "extra compute" to viewing it as **"Dimensional Expansion."** CoT allows the model to "unfold" the problem into more dimensions where it can be resolved.
*   **Predictive Power:** $D_{eff}$ is a stronger predictor of the "Reasoning vs Retrieval" regime than Speed.
