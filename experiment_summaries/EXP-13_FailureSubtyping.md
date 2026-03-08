# Experiment 13: Regime Mining and Failure Subtyping

**Phase:** 3 — The Pivot to Dynamics & Intervention
**Date:** Early February 2026
**Model:** Qwen2.5-0.5B (Analysis Layer: 13)
**Status:** Completed — **SUCCESS**

## Connection to Prior Work

EXP-09 through EXP-12 established that geometric signatures exist and predict success. But we were treating all successes as one group and all failures as another. EXP-12's convergence dynamics suggested that failures might have distinct *subtypes* with different geometric profiles.

## Research Question

**Does the model fail in multiple geometrically distinct ways?** Can unsupervised clustering reveal computational regime subtypes, and does trajectory geometry predict success better than metadata alone?

## Hypotheses
1.  **H1 (Failure Subtypes):** CoT failures (G3) will cluster into at least two modes: "Collapsed" (low dimension) and "Wandering" (high dimension but incoherent).
2.  **H2 (Retrieve-and-Commit):** Direct Success (G2) will have a distinct "retrieve-and-commit" signature (high speed, early commitment) compared to CoT success.
3.  **H3 (Predictive Value):** Trajectory geometry will predict problem success more accurately than metadata alone (like prompt type or length).

## Method Summary
*   **Clustering:** Applied K-Means (best $k=3$) to the 31-metric suite.
*   **Sliding Window Analysis:** Calculated "Dimension Drop" (Early Dim - Late Dim) to detect phase transitions.
*   **Predictive Modeling:** Trained a Logistic Regression model on trajectory metrics to predict Success/Failure (5-fold cross-validation).

## Key Results & Analysis
### 1. Failure Subtyping (The G3 Map)
CoT Failures (G3) were not monolithic. They clustered into:
*   **Subtype A: The "Broken Engine" (Collapsed Failure):** High tortuosity, low effective dimension. The model never "entered" the reasoning regime. 
*   **Subtype B: The "Lost Wanderer" (Incoherent Failure):** High expansion and high dimensionality, but negative convergence. The model "thought hard" but drifted away from the solution basin.

### 2. Direct Success (G2) vs CoT Success (G4)
We confirmed the **"Retrieve-and-Commit"** profile for Direct Success:
*   **G2** vs **G4**: Higher speed, higher tortuosity (straighter path), and significantly lower effective dimension.
*   **Insight:** Direct success is a retrieval event; CoT success is a computational event.

### 3. Phase Transition Detection
The "Dimension Drop" metric proved that **G4 (Success)** has a statistically significant drop in dimensionality in the second half of the trajectory.
*   **Interpretation:** This is the geometric signature of **"Consensus/Commitment."** The model expands to explore, then collapses once the answer is found.

### 4. Predictive Value
The geometric metrics predicted success with an **AUC of 0.898** for direct answers and **0.772** for CoT.
*   **Crucial Finding:** The geometry predicts success significantly better than prompt type alone ($AUC \approx 0.63$).

## Methodological Issues / Limitations
*   **Sample Size (G2):** The number of Direct Successes (G2) was relatively small ($N=52$), potentially limiting the robustness of the "Retrieve-and-Commit" profile comparison.
*   **Metric Ranking:** Some metrics (like `early_late_ratio`) were so dominant that they masked subtle effects from other topological features.

## Conclusions & Implications

**Verdict: SUCCESS.** We can now diagnose *mechanism* (Retrieval vs Reasoning) and *failure mode* (Collapse vs Confusion) purely from latent geometry, without reading the text. Reasoning is a two-phase process (Explore → Commit) and the "Commitment" signature is geometrically readable. Geometry predicts success (AUC 0.898) far better than metadata alone (AUC 0.63).

## Influence on Next Experiment

*   With ~10 metrics proving their worth, the question became: are we missing "high-order" signals?
*   **EXP-14** expanded the metric suite to 33 variables computed across all 28 layers — and discovered the shocking result that "good geometry" is regime-dependent (CoT success ≠ Direct success).
