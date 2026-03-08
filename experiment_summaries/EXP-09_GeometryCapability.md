# Experiment 09: Geometry-Capability Correlation

**Phase:** 3 — The Pivot to Dynamics & Intervention
**Date:** November 2025
**Model:** Qwen2.5-0.5B
**Status:** Completed — **BREAKTHROUGH**

## Connection to Prior Work

Experiment 08 gave us the *measurement tools* (Speed, Curvature, Radius of Gyration) and proved that distinct dynamic regimes exist in latent space. Experiment 04 gave us the *performance variation* — clear success/failure groups from multi-pass gating. EXP-09 was the critical experiment that married these two threads: applying the Exp 8 metrics to the Exp 4 performance groups to ask the defining question of the project.

## Research Question

**Does a "correct" thought look geometrically different from a "hallucination" in transformer hidden states?** Specifically: can the dynamic metrics (Speed, Curvature, Stabilization) from Exp 8 reliably distinguish between successful reasoning (G4: CoT Success) and reflexive failure (G1: Direct Failure)?

## Hypotheses

1.  **H1 (The Speed of Thought):** Successful reasoning (G4) will traverse more distance (higher Speed) than reflexive failures (G1).
2.  **H2 (The Shape of Thought):** Reasoning requires "turning" (changing subspace). G4 should have higher curvature (lower directional consistency) than G1.
3.  **H3 (Stabilization):** A correct answer should "land" in a stable region. G4 should have higher late-layer stability than G1.

## Experimental Design

*   **Dataset:** 300 multi-step arithmetic/logic problems.
*   **Groups:**
    *   **G1 (Direct Failure):** Model answers immediately and incorrectly.
    *   **G4 (CoT Success):** Model uses Chain-of-Thought and answers correctly.
*   **Metrics:** Speed ($||h_t - h_{t-1}||$), Directional Consistency ($||\sum \Delta h|| / \sum ||\Delta h||$), Stabilization (Cosine Similarity of $h_t$ vs $h_{t-1}$).

## Key Findings

The differences were not subtle — they were astronomical:

*   **Speed:** G4 is **3-4x faster** than G1 (Cohen's $d > 3.0$). The "thought" literally moves through representational space with more energy.
*   **Curvature:** G1 travels in a straight line (Directional Consistency ~0.5), while G4 winds and turns (Directional Consistency ~0.05).
    *   **Hallucination is a straight line.** The model collapses into a simple, wrong prediction and stays there. **Reasoning is a winding path.** The model navigates through different semantic subspaces.
*   **Stabilization:** G4 stabilizes in the final layers (converging on an answer). G1 destabilizes (wandering into noise).

### The "Ballistic vs Diffusive" Framework

This experiment crystallized the central metaphor of the research:
*   **System 1 (G1):** Ballistic — straight-line, low energy. The model fires a shot and misses.
*   **System 2 (G4):** Diffusive — high-curvature, high energy. The model "explores" the space.

## Limitations

*   **Length Confound:** CoT is longer than Direct answers. Secondary analysis showed the effect persists even when controlling for window size, but the confound remained an open concern (addressed definitively in Exp 15).
*   **Causality:** Does the geometry *cause* the success, or is it just a byproduct of outputting more tokens? (Investigated in Exp 10).

## Conclusions & Implications

**Verdict: BREAKTHROUGH.** We found the physical signature of "System 2" thinking. We can now detect if a model is "thinking" *without reading the output* — by measuring the tortuosity of its latent trajectory.

This was the foundational result that validated the entire Trajectory Geometry research program.

## Influence on Next Experiment

Two immediate follow-ups:
*   **EXP-09B (Cross-Model Replication):** Test if the signatures are architecture-agnostic by replicating on TinyLlama-1.1B.
*   **EXP-10 (Self-Report Consistency):** Test whether the model has introspective access to its own geometric state — can it "feel" when it's reasoning vs hallucinating?
