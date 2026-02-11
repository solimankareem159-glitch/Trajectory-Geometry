# Experiment 09B: Cross-Model Replication

**Date / Context:** September 2026 / Phase 3: The Pivot to Dynamics & Intervention

## Motivation & Prior Assumptions
*   **Context:** Experiment 09 successfully identified geometric signatures of correctness (Speed, Curvature, Stabilization) in Qwen2.5-0.5B.
*   **Hypothesis:** These signatures should be model-agnostic and present in other architectures.
*   **Action:** Try to replicate the findings on TinyLlama-1.1B-Chat using the same prompts and dataset.

## Experimental Setup
*   **Model:** TinyLlama-1.1B-Chat (vs Qwen2.5-0.5B in Exp 9)
*   **Dataset:** 300 multi-step arithmetic problems (Same as Exp 9)
*   **Metric:** Speed, Curvature, Stabilization Rate

## Procedure
1.  Run the standard CoT and Direct prompts on TinyLlama.
2.  Classify trajectories into G1 (Direct Fail), G4 (CoT Success), etc.
3.  Compare G4 vs G1 metrics.

## Results & Observations

### 1. Capability Failure
The replication failed immediately due to model capability issues. TinyLlama-1.1B could not solve the problems.

*   **Initial Report (N=176):** G4 (CoT Success) = 1 (<1%)
*   **Full Dataset (N=300):** G4 (CoT Success) = 2 (<1%)
*   **Data Audit:** I manually verified the parsing logic (September 2026). While the original parser only caught the last integer, a more robust check confirmed that **zero** additional correct answers were hidden in the text. The failure rate is genuine.

### 2. Statistical Vacuum
With only 1 success sample, no statistical comparison was possible.

## Interpretation
*   **Result:** FAILURE (Technical).
*   **Key Insight:** Geometric signatures differ from "static interactions" in that they require a *competent* trajectory to measure. If the model cannot generate a coherent chain of thought, there is no "correct geometry" to measure.
*   **Pivot:** We returned to Qwen2.5 and focused on *internal* interventions rather than cross-model generalization for now.

## Next Steps
*   Proceed to Experiment 10: Causal Intervention (steering the geometry).
