# Experiment 13: Regime Mining and Failure Subtyping

**Status:** Completed
**Date:** Early Feb 2026
**Model:** Qwen2.5-0.5B
**Analysis Layer:** 13

## Motivation & Prior Assumptions
*   **Context:** Previous experiments established that geometric signatures exist.
*   **Problem:** We were treating all successes as one group and all failures as another. Does a model fail in only one way? Is "Direct Success" the same as "CoT Success"?
*   **Goal:** Use unsupervised learning (clustering) to "mine" the data for distinct computational regimes and failure subtypes.

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

## Interpretation (Meanings & Implications)
*   **Verdict:** **SUCCESS**.
*   **Geometry as Ground Truth:** We can now diagnose *mechanism* (Retrieval vs Reasoning) and *failure mode* (Collapse vs Confusion) purely from latent geometry, without reading the text.
*   **The "Commitment" Signature:** Validated that reasoning is a two-phase process (Explore $\to$ Commit) and that we can "watch" this transition happen.
