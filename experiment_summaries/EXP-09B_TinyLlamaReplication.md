# Experiment 09B: Cross-Model Replication (TinyLlama)

**Phase:** 3 — The Pivot to Dynamics & Intervention
**Date:** November 2025
**Model:** TinyLlama-1.1B-Chat
**Status:** Completed — **FAILURE (Technical)**

## Connection to Prior Work

EXP-09 successfully identified geometric signatures of correctness in Qwen2.5-0.5B. The natural next step was to test whether these signatures are architecture-agnostic by replicating on a different model family.

## Research Question

**Are the geometric signatures of reasoning (Speed, Curvature, Stabilization) model-agnostic?** Can they be observed in TinyLlama-1.1B using the same dataset?

## Hypotheses

1.  **H1:** The Speed/Curvature/Stabilization signatures from EXP-09 will replicate in TinyLlama-1.1B.
2.  **Null:** The signatures are Qwen-specific artifacts.

## Experimental Design

*   **Model:** TinyLlama-1.1B-Chat (vs Qwen2.5-0.5B in EXP-09)
*   **Dataset:** 300 multi-step arithmetic problems (same as EXP-09)
*   **Metrics:** Speed, Curvature, Stabilization Rate
*   **Procedure:** Run CoT and Direct prompts, classify into G1/G4 groups, compare metrics.

## Key Findings

### 1. Capability Failure
The replication failed immediately due to model capability issues. TinyLlama-1.1B could not solve the problems.

*   **Initial Report (N=176):** G4 (CoT Success) = 1 (<1%)
*   **Full Dataset (N=300):** G4 (CoT Success) = 2 (<1%)
*   **Data Audit:** Manual verification confirmed that **zero** additional correct answers were hidden in the text. The failure rate is genuine.

### 2. Statistical Vacuum
With only 1 success sample, no statistical comparison was possible.

## Limitations

*   The experiment could not test the hypothesis because TinyLlama lacks the baseline reasoning capability needed to produce the success/failure variation.
*   This is a capability floor problem, not a geometric signature problem.

## Conclusions & Implications

**Verdict: FAILURE (Technical).** Geometric signatures require a *competent* trajectory to measure. If the model cannot generate a coherent chain of thought, there is no "correct geometry" to measure. This established the concept of a **"capability floor"** for geometric analysis.

## Influence on Next Experiment

*   We returned to Qwen2.5 and focused on depth of analysis rather than cross-model generalization.
*   **EXP-10** tested whether the model has introspective access to its own geometric state (it doesn't).
*   Cross-model validation was revisited later in EXP-16/16B with more capable architectures.
