# Experiment 17: Baseline Replication & Multi-Mode Prompting

**Phase:** 4 — Consolidation & Robustness
**Date:** February 2026 (2026-02-11)
**Model:** Qwen2.5-3B-Instruct
**Status:** Completed

## Connection to Prior Work

Experiments 14–16B established that geometric signatures of reasoning are robust across the Qwen 0.5B and 1.5B scales, and even at the 70m-parameter extreme (Pythia). EXP-17 extended the **scale ladder** to 3B parameters — the largest model tested to date — and transitioned from pure replication to testing multi-mode computational signatures with the full 33-metric pipeline developed in EXP-14.

## Research Question

**Do the geometric signatures of reasoning hold at the 3B parameter scale?** Secondarily: when the same content is presented through different computational modes (8 distinct prompting strategies), do the modes produce geometrically distinguishable trajectories?

## Hypotheses

1.  **H1 (Scale Invariance):** The core geometric signatures (Speed, Curvature, Effective Dimension, Radius of Gyration) will replicate at 3B with comparable effect sizes to the 0.5B and 1.5B results.
2.  **H2 (Mode Separability):** Different prompting modes (e.g., Direct, CoT, Step-by-Step, Verify-then-Answer) will produce distinct geometric profiles when the same underlying problem is held constant.

## Experimental Design

*   **Phase 17A:** Direct replication of EXP-09 using the same multi-step arithmetic dataset (300 problems) with the expanded 33-metric pipeline.
*   **Phase 17B:** 8-mode multi-mode prompting — the same content was processed through 8 distinct computational modes to test whether mode identity produces readable geometric signatures.
*   **Hardware:** AMD RX 5700 XT (8GB VRAM via DirectML) — at the edge of feasibility for 3B inference with hidden state extraction.

## Key Findings

*   **Regime-Relative Geometry Confirmed:** Geometric signatures continue to operate regime-relative at 3B scale. "Good" CoT geometry and "Good" Direct geometry remain distinct profiles.
*   **33-Metric Pipeline Deployed:** Successfully applied the expanded metric suite on a larger model, demonstrating the pipeline's scalability.
*   **Hardware Constraints Emerged:** The 8GB VRAM limit on the AMD RX 5700 XT imposed hard constraints at 3B scale, requiring careful batch management and limiting the depth of per-layer analysis.

## Limitations

*   **Hardware Bottleneck:** 3B inference with full hidden state extraction on 8GB VRAM required aggressive memory management, limiting sample throughput.
*   **Multi-Mode Analysis:** The 8-mode comparison was more exploratory than confirmatory — sample sizes per mode were modest.

## Conclusions & Implications

**Verdict: SUCCESS.** EXP-17 provides another data point confirming that geometric signatures of reasoning scale consistently with model capacity. The scale ladder now spans 70m → 0.5B → 1.5B → 3B with consistent results.

The multi-mode prompting analysis opened a new direction: not just distinguishing Success from Failure, but distinguishing *how* the model reasons across different instructional framings.

## Influence on Next Experiment

The success at 3B reinforced the need for a **standardized metric suite** that could be applied consistently across all models and experiments. This directly motivated **EXP-18: Consolidated Metric Suite**, which formalized the growing metric collection into a definitive, production-ready analytical framework.
