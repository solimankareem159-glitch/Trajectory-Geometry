# Experiment 14: Comprehensive Metric Expansion

**Status:** Completed
**Date:** Feb 2026
**Model:** Qwen2.5-0.5B
**N Layers:** 28 (Full depth)

## Motivation & Prior Assumptions
*   **Context:** Experiments 11-13 proved that simple geometry tracks reasoning.
*   **Problem:** We were using a limited set of ~10 metrics. Are we missing "high-order" signals?
*   **Goal:** Expand the metric suite to 33 variables and compute them across every layer of the network to find the "Universal Signature" of success.

## Hypotheses
1.  **H1 (Regime Convergence):** There is a single geometric state that defines "Success" regardless of whether the model used CoT or Direct answering. (FAILED - See below).
2.  **H2 (Commitment Inflexion):** There is a readable "moment of commitment" in the trajectory where the model transition from exploration to generation.
3.  **H3 (Cross-Layer Coordination):** Reasoning requires higher synchronization between layers than retrieval.

## Method Summary
*   **Full Depth Extraction:** Hidden states extracted from all 28 layers.
*   **Metric Expansion:** Suite increased to 33 metrics, including:
    *   `cos_to_running_mean` (Coherence)
    *   `time_to_commit` (Phase Timing)
    *   `msd_exponent` (Diffusion character)
    *   `interlayer_alignment` (Cross-layer sync)
    *   `spectral_entropy` (Path complexity)

## Key Results & Analysis
### 1. Paradigm Shift: Regime-Dependent Success
The most shocking finding was that **"Good Geometry" is regime-relative.**
*   **The Discovery:** 10 of 14 key metrics showed **opposite** effects for success in CoT vs Direct regimes.
*   **CoT Success** = Lower Speed, Smaller Radius, Higher Coherence. (Focusing/Zooming).
*   **Direct Success** = Higher Speed, Larger Radius, Lower Coherence. (Ballistic Retrieval).
*   **Analysis:** You cannot build a universal success detector. A "good" CoT trajectory would look like a failure if it were a Direct answer.

### 2. The Commitment Timing Signature
We introduced `time_to_commit` to measure when the Radius of Gyration drops most sharply.
*   **Direct Success:** Commits nearly instantly (~5 tokens).
*   **CoT Success:** Commits in the middle (~11 tokens).
*   **Failures (G1/G3):** Commit significantly later or not at all.
*   **Result:** Commitment timing is a readable, predictive signal of both strategy and outcome.

### 3. Layer Depth Profiles
*   **Regime signals** (CoT vs Direct) are strongest in **early layers (0-7)**.
*   **Success signals** (Correct vs Wrong) are strongest in **middle layers (10-14)**.
*   **Commitment signals** (Phase ends) are strongest in **late layers (20-24)**.

## Methodological Issues / Limitations
*   **Computational Expense:** Computing 33 metrics for 28 layers across 602 trajectories ($33 \times 28 \times 602 \approx 550,000$ data points) pushed the limits of the DirectML acceleration.
*   **Normalization:** Comparing `msd_exponent` between very short trajectories (Direct) and long ones (CoT) remains statistically challenging.

## Interpretation (Meanings & Implications)
*   **Verdict:** **BREAKTHROUGH**.
*   **Functional Specialization:** The model uses different "depth regions" for different parts of the reasoning task. 
*   **Divergent Strategies:** We've proven that **Reasoning** and **Retrieval** are not just different weights; they are different **dynamical regimes**.
