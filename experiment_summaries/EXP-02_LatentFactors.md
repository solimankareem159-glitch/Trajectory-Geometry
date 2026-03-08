# Experiment 02: Latent Factors (Token-Level)

**Phase:** 1 — The Intuition of "Shapes" (Part 2)
**Date:** November 2025
**Model:** Qwen2.5-0.5B
**Status:** Completed — **INVALID / TRIVIAL**

## Connection to Prior Work

EXP-01 failed to find geometric signatures in *output* embeddings — the signal was swamped by topic and lexical noise. We suspected the cognitive move happens in the high-dimensional hidden states *before* the output, and pivoted to open-weights models to access them directly.

## Research Question

**Can we decompose internal hidden states using NMF to find the "atomic units" of thought?** (e.g., is "Critique" a mix of "Analyze" + "Negate"?)

## Hypotheses
1.  **H1 (Internal Signatures):** Different operators (Summarize vs. Critique) will have distinct "warp traces" (magnitude of change) in hidden states.
2.  **H2 (Compositionality):** A composite task ("Summarize + Critique") will be a linear combination of the warp signatures of its components.
3.  **Null:** Warp traces will be dominated by universal transformer dynamics (e.g., processing spikes) regardless of the operator.

## Method Summary
*   **Model:** Qwen2.5-0.5B (Open weights, allowing access to hidden states $h$).
*   **Metric:** "Warp" $W_t = ||h_t - h_{t-1}||_2$ (Euclidean distance between consecutive token states).
*   **Procedure:**
    1.  Ran single and composite operator prompts.
    2.  Extracted hidden states across all 24 layers.
    3.  Applied NMF to decompose the warp traces into $k$ basis factors.
    4.  Attempted to reconstruct composite traces from single-operator factors.

## Key Results
*   **Universal Signature (Negative Result):** All operators produced the **exact same warp signature**: a massive spike at $t=0$ (processing the prompt) followed by a flat line.
    *   *There was no distinct "oscillating" or "ramping" profile for specific operators.*
*   **Layer Stability:** Middle layers (5-12) showed coherent, stable geometry. The final layer (24) was chaotic and dominated by vocabulary projection noise.
*   **Trivial Compositionality:** Linear reconstruction worked perfectly, but only because it was reconstructing a "spike + flat" signal from "spike + flat" basis vectors. It proved structural similarity, not semantic composition.

## Methodological Issues / Limitations
*   **Metric Flaw:** "Warp" (magnitude of change) discarded all *directional* information. It measured *that* the model changed state, not *where* it went.
*   **Universal Artifacts:** The $t=0$ spike is a fundamental property of how transformers ingest the first token (embedding $\to$ first hidden state). It drowned out any subtle operator signals.
*   **Sample Size:** Analysis relied on single examples per operator, making it susceptible to prompt-specific noise.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Invalid / Trivial**. The "distinct geometric profiles" were a mirage caused by over-interpreting specific prompt responses.
*   **Meaning:** Measuring *magnitude* ($||v||$) is insufficient. A "thought" is a vector with a direction. By taking the norm, we threw away the actual information.
*   **Shift:** We must stop looking at scalar metrics (distance/speed) in isolation and start looking at **direction** and **regime** (what "mode" the model is in).

## Open Questions
*   If we look at the *direction* rather than the *magnitude*, can we see the difference?
*   Do these signatures change when the model is "Thinking" vs "Speaking"?

## Conclusions & Implications

**Verdict: INVALID / TRIVIAL.** Measuring *magnitude* ($||v||$) is insufficient. A "thought" is a vector with a direction. By taking the norm, we threw away the actual information. We must look at **direction** and **regime**.

However, the finding that middle layers (5-12) show coherent, stable geometry while the final layer is chaotic became an important heuristic for all future experiments.

## Influence on Next Experiment

*   **Pivot to Regimes:** The realization that "Thinking" might act differently than "Speaking" led to **EXP-03: Regime Invariants**, where we explicitly separated the "Listen", "Think", and "Speak" phases to test for geometric invariance across them.
