# Experiment 06: Pilot Metric Validation (The Wilderness)

**Phase:** 2 — The "Invariant" Trap (The Wilderness Phase)
**Date:** November 2025
**Model:** Qwen2.5-0.5B
**Status:** Completed — **INCONCLUSIVE / WEAK**

## Connection to Prior Work

EXP-03 failed due to massive topic confounding — probes learned to distinguish topics, not operators. We hypothesized that stricter cross-validation (LOPO, LOCO) could eliminate the confound and reveal the true operator signal.

## Research Question

**Does *any* static operator signal exist when proper cross-validation controls are applied?**

## Hypotheses
1.  **H1 (Paraphrase Invariance):** Probes will generalize across paraphrases (LOPO F1 > Chance).
2.  **H2 (Content Invariance):** Probes will generalize across topics (LOCO F1 > Chance).
3.  **Null:** Performance will drop to random variance when strict holdouts are applied.

## Method Summary
*   **Model:** Qwen2.5-0.5B.
*   **Design:** Pilot study with a small subset of operators (likely 3 classes based on chance baseline).
*   **Procedure:**
    1.  **LOPO:** Train on $N-1$ paraphrases, test on held-out paraphrase.
    2.  **LOCO:** Train on $N-1$ content types, test on held-out content.
    3.  **Permutation Control:** Shuffle labels to establish a true empirical baseline.

## Key Results
*   **Weak Signals:**
    *   **LOPO F1:** ~0.30 - 0.48 (vs Chance ~0.33).
    *   **LOCO F1:** ~0.45 - 0.50 (vs Chance ~0.33).
*   **Interpretation:** While slightly above random (and above the permutation baseline of ~0.24), the signal was **weak**. Ideally, "distinct cognitive operators" should obtain F1 > 0.8.
*   **No "Grand Unification":** There was no "smoking gun" layer or metric that reliably separated operators across all contexts.

## Methodological Issues / Limitations
*   **Pilot Scale:** The study was small (3 folds), likely masking larger variance issues.
*   **Diminishing Returns:** We spent significant effort engineering strict validation (LOPO/LOCO) only to find that the underlying signal (static embeddings) was barely detectable.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Inconclusive / Weak**.
*   **Meaning:** "The Wilderness." We were digging in the wrong place. We kept refining the *measurement* of static states (Probing, LOPO, CCA) but the signal just wasn't there.
*   **Shift:** This contributed to the "slow realization" that we needed to stop looking for static vectors. If the signal is this weak even with rigorous controls, it's likely not a static property at all.

## Open Questions
*   Why is the signal so weak? Is it because the operator "moves" the state rather than "locating" it?

## Conclusions & Implications

**Verdict: INCONCLUSIVE / WEAK.** "The Wilderness." We kept refining the *measurement* of static states but the signal simply wasn't there. If the signal is this weak even with rigorous controls, it's likely not a static property at all.

## Influence on Next Experiment

*   **Desperation → Innovation:** The failure to find strong results in EXP-05 and EXP-06 forced us to rethink the physics of the problem.
*   **One Last Try:** We decided to try one final rigorous static analysis (EXP-07) at large scale (N=2,000) to definitively kill or confirm the "static vector" hypothesis.
