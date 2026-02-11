# Experiment 09: Geometry-Capability Correlation

**Date / Context:** September 2026 / Phase 3: The Pivot to Dynamics & Intervention

## Motivation & Prior Assumptions
*   **Context:** Experiment 08 defined the metrics (Speed, Curvature). Experiment 04 defined the groups (Success vs Failure).
*   **Assumption:** If "thinking" is a physical process in high-dimensional space, then a "correct" thought should move differently than a "confused" one.
*   **Goal:** To determine if the dynamic metrics from Exp 8 can distinguish between the success/failure groups from Exp 4.
    *   **G1:** Direct Failure (The "Reflexive Error").
    *   **G4:** CoT Success (The "Reasoned Answer").

## Hypotheses
1.  **H1 (The Speed of Thought):** Successful reasoning (G4) will traverse more distance (Higher Speed) than reflexive failures (G1).
2.  **H2 (The Shape of Thought):** Reasoning requires "turning" (changing subspace). G4 should have higher curvature (lower directional consistency) than G1.
3.  **H3 (Stabilization):** A correct answer should "land" in a stable region. G4 should have higher late-layer stability than G1.

## Method Summary
*   **Model:** Qwen2.5-0.5B.
*   **Dataset:** 300 multi-step arithmetic/logic problems.
*   **Groups:**
    *   **G1 (Direct Failure):** Model answers immediately and incorrectly.
    *   **G4 (CoT Success):** Model uses Chain-of-Thought and answers correctly.
*   **Metrics:** Speed ($||h_t - h_{t-1}||$), Directional Consistency ($||\sum \Delta h|| / \sum ||\Delta h||$), Stabilization (Cosine Sim of $t$ vs $t-1$).

## Key Results
*   **Massive Separation:** The differences were not subtle. They were astronomical.
    *   **Speed:** G4 is **3-4x faster** than G1 (Cohen's $d > 3.0$). The "thought" literally moves with more energy.
    *   **Curvature:** G1 travels in a straight line (Dir Consistency ~0.5), while G4 winds and turns (Dir Consistency ~0.05).
        *   *Interpretation:* **Hallucination is a straight line.** The model collapses into a simple, wrong prediction and stays there. **Reasoning is a winding path.** The model navigates through different semantic subspaces (e.g., "Plan" $\to$ "Calc" $\to$ "Critique").
    *   **Stabilization:** G4 stabilizes in the final layers (converging on an answer). G1 destabilizes (wandering into noise).

## Methodological Issues / Limitations
*   **Confounding:** Is this just "Length"? CoT is longer than Direct answer. (Secondary analysis showed effect persists even when controlling for window size).
*   **Causality:** Does the geometry *cause* the success, or is it just a byproduct of "outputting more tokens"? (Answered in Exp 10).

## Interpretation (The "ballistic" vs "diffusive" Insight)
*   **Verdict:** **Breakthrough**.
*   **Meaning:** We found the physical signature of "System 2" thinking.
    *   **System 1 (G1):** Ballistic, straight-line, low energy. The model fires a shot and misses.
    *   **System 2 (G4):** Diffusive, high-curvature, high energy. The model "explores" the space.
*   **Paradigm Shift:** We can now detect if a model is "thinking" *without reading the output*. We just need to measure the **tortuosity** of its path.

## How This Informed the Next Experiment
*   **The Diagnostic:** We can now *diagnose* failure (Straight Line).
*   **The Cure:** Can we *cause* success by forcing the trajectory to curve?
*   **Next Step:** **Experiment 10** (Intervention) attempts to artificially inject "energy" (noise/steering) to break the straight-line collapse of G1.
