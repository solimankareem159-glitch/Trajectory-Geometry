# Experiment 08: Trajectory Geometry

**Date / Context:** August 2026 / Phase 3: The Pivot to Dynamics & Intervention

## Motivation & Prior Assumptions
*   **Context:** Experiment 07 (Statics) hit a ceiling. Experiment 04 (Gating) showed that *forcing* a dynamic process improves performance.
*   **Assumption:** Competence isn't a *place* (static vector); it's a *path* (trajectory). The "shape" of the thought process (speed of evolution, curvature of turns) should distinguish "reasoning" from "rambling."
*   **Goal:** To define and measure differential geometry metrics for transformer trajectories and validate if they form coherent "Dynamic Regimes."

## Hypotheses
1.  **H1 (Measurability):** We can define meaningful metrics for "Speed" ($||h_t - h_{t-1}||$), "Curvature" (angle between steps), and "Tortuosity" (path efficiency) in latent space.
2.  **H2 (Regime Discovery):** Latent states will cluster not just by *location* (as in Exp 7) but by *dynamic behavior* (e.g., a "high-speed, low-curvature" regime vs. a "low-speed, high-curvature" regime).
3.  **H3 (Causal Transitions):** We can trigger a shift in geometry (e.g., "Start Thinking") by injecting a specific cue. (Tested in Exp 8').

## Method Summary
*   **Model:** Qwen2.5-0.5B.
*   **Metrics defined:**
    *   **Speed:** Euclidean distance between consecutive token states.
    *   **Curvature:** Cosine angle between the entrance vector and exit vector of a state.
    *   **Radius of Gyration ($R_g$):** Volume of the trajectory cloud.
*   **Procedure:**
    1.  Extracted full trajectories for diverse prompts.
    2.  Applied K-Means clustering to the *dynamic metrics* (not the states themselves) to find "Regimes."
    3.  **Exp 8' (Transition Test):** Measured change in metrics before/after specific "cue words" (e.g., "Wait", "Therefore").

## Key Results
*   **Regime Discovery (Success):** We successfully identified distinct dynamic regimes in the mid-layers (Layer 13).
    *   *Result:* K=9 clusters found with good silhouette scores (0.179) and high predictability (69%).
    *   *Meaning:* The model isn't just "processing"; it switches between distinct "modes of motion."
*   **Transition Failure (Exp 8'):** While distinct regimes *exist*, we could not reliably *trigger* them with simple cue words.
    *   *Result:* "Planning" cues did not consistently increase stability or decrease speed ($p > 0.05$).
    *   *Interpretation:* Dynamics are emergent properties of the flow, not simple reactions to single tokens.

## Methodological Issues / Limitations
*   **Descriptive, not Predictive:** We can *see* the regimes, but (in this experiment) we didn't know which ones were "good" or "bad." Is high curvature a sign of confusion or careful consideration?
*   **Triggering Difficulty:** The failure of Exp 8' showed that "steering" the geometry is harder than just prompting it.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Foundational Success**.
*   **Meaning:** We finally have the right *ruler*. Measuring coordinates (Exp 1-7) was wrong. Measuring **Speed** and **Curvature** gives us a reliable way to characterize the "texture" of thought.
*   **Shift:** We stopped asking "Where is the geometric signature?" and started asking "Does the **geometry predict correctness**?"

## How This Informed the Next Experiment
*   **The Crucial Link:** Exp 8 gave us the measures. Exp 4 gave us the performance variation (Success vs Failure).
*   **Next Step:** **Experiment 09** combined these. We applied the Exp 8 metrics to the Exp 4 groups (Success/Failure) to answer the billion-dollar question: **Does a "correct" thought look different from a "hallucination"?** (spoiler: yes).
