# Experiment 07: Static Operator Geometry (The Ceiling)

**Phase:** 2 — The "Invariant" Trap (Final Attempt)
**Date:** November 2025
**Model:** Qwen2.5-0.5B
**Status:** Completed — **VALID BUT INSUFFICIENT**

## Connection to Prior Work

EXP-06's pilot study showed only weak signals. We needed one definitive, large-scale experiment to prove or disprove static operator theory before committing to the paradigm shift toward dynamics.

## Research Question

**At maximum statistical power (N=2,000), can static methods reliably separate cognitive operators in hidden state space?**

## Hypotheses
1.  **H1 (Mid-Layer Abstraction):** Operator coherence will peak in the middle layers (where "concepts" supposedly live), not in the input/output.
2.  **H2 (Centroid Stability):** The "average" operator vector (centroid) will be stable across diverse content (Cosine Similarity > 0.9).
3.  **Null:** Operators are just ad-hoc instruction following; no stable abstract representation exists.

## Method Summary
*   **Model:** Qwen2.5-0.5B.
*   **Dataset:** Large-scale N=2,000 (10 Operators $\times$ 20 Content Domains $\times$ 10 Paraphrases).
*   **Procedure:**
    1.  **Extraction:** Streaming feature extraction (Directional Deltas) using "Speed Patch" method.
    2.  **Stability Metric:** Mean pairwise cosine similarity of an operator's centroid across different content domains.
    3.  **Decodability:** Nearest Centroid Classifier (NCC) to predict operator from held-out samples.

## Key Results
*   **Peak Stability:** Operator stability **peaked at Layer 16** (Score: 0.924), significantly higher than input/output. This confirmed that the model *does* abstract the operator instruction into a stable internal state.
*   **Weak Separability:** While stable, the operators were **hard to distinguish** from each other. NCC F1 was **~0.30** (Chance = 0.10).
    *   *Interpretation: The "thought" is stable, but the "thoughts" for different operators overlap heavily.*
*   **Topology:** Related operators (e.g., *Critique* and *Question*) shared closer geometric subspaces than unrelated ones, suggesting a semantic map.

## Methodological Issues / Limitations
*   **The Ceiling Effect:** F1 of 0.30 is "statistically significant" but "practically useless" for reliable control. We hit the ceiling of what static analysis could reveal.
*   **Static Blindness:** By averaging into centroids, we smoothed out the *motion* of the thought. We measured where the car *parked*, not how it *drove*.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Valid but Insufficient**.
*   **Meaning:** Theories of "Concept Vectors" are partially true but incomplete. You can find a "Summarize" direction, but it's fuzzy. Relying on it for precise control is impossible.
*   **Shift:** This experiment was the "graduation" from Phase 2. We proved static structure exists (Stability > 0.9), but we also proved it's too weak to explain competence (F1 < 0.35).
*   **The Paradigm Shift:** If the *location* distinguishes operators poorly, maybe the *path* distinguishes them better. This explicitly set the stage for **Trajectory Geometry**.

## Conclusions & Implications

**Verdict: VALID BUT INSUFFICIENT.** "Concept Vectors" are partially true but incomplete. You can find a "Summarize" direction (Stability > 0.9), but it's too fuzzy for reliable control (F1 < 0.35). This was the "graduation" from Phase 2. If *location* distinguishes operators poorly, maybe the *path* distinguishes them better.

## Influence on Next Experiment

*   **The Paradigm Shift:** From statics to dynamics. Having characterized the static "nodes" (EXP-07), we moved to characterizing the "edges" (transitions).
*   **EXP-08: Trajectory Geometry** abandoned the search for "Where is Summarize?" and asked "How does Summarize *move*?" using differential geometry metrics (Speed, Curvature).
