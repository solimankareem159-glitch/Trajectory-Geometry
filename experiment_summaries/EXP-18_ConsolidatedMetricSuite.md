# Experiment 18: Consolidated Metric Suite

**Phase:** 4 — Consolidation & Robustness
**Date:** February 2026 (2026-02-13)
**Model:** Qwen2.5-0.5B (using data from prior experiments)
**Status:** Completed

## Connection to Prior Work

By EXP-17, the metric suite had grown organically from 3 metrics (EXP-08) to 10 (EXP-09) to 33 (EXP-14) across multiple experiments. Each experiment added metrics ad hoc to test specific hypotheses. EXP-18 was the engineering response: a systematic consolidation of all proven metrics into a single, formalized, production-ready analytical framework.

## Research Question

**Can we formalize and unify the full metric suite into a standardized codebase capable of holistic trajectory profiling?** The goal was not to discover new phenomena but to transform exploratory tools into a rigorous, reproducible analytical pipeline.

## Experimental Design

*   **Data Aggregation:** Combined the dataset from Experiment 9 (300 problems, behavioral labels) with the hidden states generated during Experiment 14 (all 28 layers).
*   **Metric Formalization:** Implemented 54 distinct metrics grouped into 12 conceptual families:
    1.  **Kinematic:** Speed, Acceleration, Jerk
    2.  **Volumetric:** Radius of Gyration, Effective Dimension, Convex Hull Volume
    3.  **Convergence:** Cosine/Distance convergence slopes, Time-to-Commit, Commitment Sharpness
    4.  **Diffusion:** MSD Exponent, Anomalous Diffusion Classification
    5.  **Spectral:** Spectral Entropy, Dominant Frequency
    6.  **RQA:** Recurrence Rate, Determinism, Laminarity
    7.  **Cross-Layer:** Interlayer Alignment, Layer-wise Speed Correlation
    8.  **Landmark:** Distance to Success/Failure Centroids
    9.  **Attractor:** Local Expansion Rate, Basin Depth
    10. **Embedding Stability:** Token-level cosine drift
    11. **Information:** Entropy Rate, Mutual Information
    12. **Inference:** Phase Count, Phase Transition Detection
*   **Pipeline:** The `TrajectoryMetrics` class was structured for comprehensive geometric profiling across all identified dimensions of thought trajectories.

## Key Findings

*   **Definitive Suite Established:** A robust codebase (`metric_suite.py`) capable of evaluating trajectory geometry holistically, replacing the ad hoc per-experiment scripts.
*   **Redundancy Identification:** Several metrics were found to be highly correlated (e.g., Speed and Acceleration; Turning Angle and Directional Consistency), enabling informed dimensionality reduction for future predictive modeling.
*   **Reproducibility:** Any future experiment can now compute the full metric suite with a single function call, ensuring cross-experiment comparability.

## Limitations

*   **No New Empirical Findings:** This was a consolidation exercise, not a discovery experiment.
*   **Computational Cost:** Computing 54 metrics across 28 layers is expensive; future work needed a strategy for selective computation.

## Conclusions & Implications

**Verdict: SUCCESS (Engineering).** EXP-18 transformed the project from exploratory data analysis into a standardized analytical framework. This was a necessary infrastructure investment that enabled the rigorous cross-model comparisons of EXP-18B and EXP-19.

## Influence on Next Experiment

The formalized suite was immediately deployed in **EXP-18B: Scaling Geometry**, which stress-tested the full 57-metric pipeline (54 base + 3 derived) across three architectures simultaneously — revealing critical infrastructure bugs that had to be resolved before the final validation study.
