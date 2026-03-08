# Experiment 14: Comprehensive Metric Expansion (Universal Signature)

**Phase:** 3/4 — Scaling & Cross-Model Validation
**Date:** February 2026
**Model:** Qwen2.5-0.5B (28 layers, full depth)
**Status:** Completed — **BREAKTHROUGH**

## Connection to Prior Work

EXP-11 through EXP-13 proved that ~10 geometric metrics reliably track reasoning success, failure subtypes, and the Explore → Commit phase transition. But the metric suite had grown organically. EXP-14 asked: with a comprehensive 33-metric suite across all layers, is there a "Universal Signature" of success?

## Research Question

**Is there a single geometric state that defines "Success" regardless of reasoning mode?** And: how do geometric signatures distribute across the depth of the network?

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

## Conclusions & Implications

**Verdict: BREAKTHROUGH.** The model uses different depth regions for different parts of the reasoning task. Reasoning and Retrieval are not just different weights — they are different **dynamical regimes**. You cannot build a universal success detector; success must be evaluated relative to the computational regime.

## Influence on Next Experiment

*   The regime-dependent nature of success raised the critical counter-argument: is this just the "length confound" in disguise?
*   **EXP-15** was designed as a direct stress test — proving that geometry provides diagnostic signal *beyond* trajectory length by stratifying problems by difficulty.
