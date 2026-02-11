# Experiment 12: Advanced Geometric Diagnostics

**Status:** Completed
**Date:** Late Jan 2026
**Model:** Qwen2.5-0.5B
**N Problems:** 300

## Motivation & Prior Assumptions
*   **Context:** Experiment 11 identified "Dimensional Collapse" in failures.
*   **Goal:** Move beyond bulk statistical averages (mean speed/dim) to explore the **intrinsic properties** and **temporal evolution** of the trajectory manifolds. We wanted to find the "geometric texture" of reasoning.

## Hypotheses
1.  **H1 (Intrinsic Complexity):** Logic-driven reasoning will exhibit higher **Fractal Dimension** than retrieval-driven patterns.
2.  **H2 (Convergence Dynamics):** Success involves a specific phase transition from expansion (exploration) to contraction (commitment), whereas failure will show inconsistent convergence.
3.  **H3 (Layer Emergence):** These signals will be strongest in the "middle layers" where high-level semantic transformations occurred in previous experiments.

## Method Summary
*   **Advanced Metrics Suite:**
    1.  **Fractal Dimension ($D_f$):** Measures the space-filling complexity of the trajectory.
    2.  **Intrinsic Dimension:** Estimate of the minimum number of variables needed to describe the manifold.
    3.  **Convergence Slopes (Cos/Dist):** Rate at which tokens approach the final hidden state.
    4.  **Early-Late Ratio:** Comparison of trajectory "energy" in the first half vs second half.
    5.  **Recurrence Quantification (RQA):** Measuring repeating patterns (determinism, laminarity) in the path.

## Key Results & Analysis
### 1. Fractal Complexity ($D_f$)
Successful trajectories (G4) showed significantly higher fractal complexity than failures (G1).
*   **G4 $D_f \approx 2.0$** vs **G1 $D_f \approx 1.7$**.
*   **Interpretation:** Reasoning is not just high-dimensional; it is **fractally dense**. It iterates and re-evaluates in a way that fills the local representational volume.

### 2. Convergence Dynamics: The Commitment Profile
We observed a distinct "Convergence Profile" in successful CoT:
*   **Success:** Initial **high-dimensional expansion** (Exploration) followed by a **sharp, steep contraction** (Commitment) towards the end token.
*   **Failure:** Often showed "Flat Convergence" (no expansion) or "Divergent Wandering" (no final commitment).

### 3. Layer-wise Evolution
The "Geometric Signature" (the Delta between Success and failure) was not uniform.
*   **Peak Signal:** Middle Layers (Layers 10-16).
*   **Discovery:** The geometry of success is "born" in the middle of the network. Early layers are too focused on input parsing; late layers are too focused on token output. Reasoning happens in the middle.

## Methodological Issues / Limitations
*   **RQA Noise:** Recurrence Quantification metrics (Determinism/Laminarity) were too noisy in small models (0.5B). The trajectories rarely "revisit" the exact same state, making recurrence counts inconsistent.
*   **PCA Variance:** $D_{eff}$ can be dominated by a few high-variance components, potentially masking subtle reasoning signals.

## Interpretation (Meanings & Implications)
*   **Verdict:** **SUCCESS**.
*   **The "Texture" of Reasoning:** Reasoning is characterized by **High Intrinsic Complexity** and **Dynamic Phase Transitions**.
*   **Diagnostic Anchor:** We can now identify where the model "made up its mind" by looking at the inflection point of the convergence slope.
