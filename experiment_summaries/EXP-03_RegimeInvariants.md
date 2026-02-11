# Experiment 03: Regime Invariants

**Date / Context:** March 2026 / Phase 2: The "Invariant" Trap

## Motivation & Prior Assumptions
*   **Context:** After Experiments 01 and 02 failed to find static signatures in output or hidden states, we hypothesized that the "mode" of the model matters.
*   **Assumption:** There must be a universal abstract representation of an operator (e.g., "Summarization") that persists regardless of whether the model is "Listening" (processing instruction), "Thinking" (internal CoT), or "Speaking" (generating output).
*   **Goal:** To find a linear mapping or shared subspace that links these three regimes.

## Hypotheses
1.  **H1 (Invariance):** A classifier trained on "Listening" states should generalize to "Thinking" states (Zero-shot transfer).
2.  **H2 (Coupling):** There exists a linear map $f$ such that $Thinking = f(Listening)$ with high $R^2$.
3.  **Null:** The regimes are geometrically disjoint; "thought" is a non-linear transformation of "instruction."

## Method Summary
*   **Model:** Qwen2.5-0.5B.
*   **Design:** Balanced dataset of 10 operators $\times$ 2 Topics (AI vs Climate) $\times$ 10 Paraphrases.
*   **Procedure:**
    1.  Extracted hidden states separated into three regimes: Listen (User Prompt), Think (CoT), Speak (Final Answer).
    2.  Trained linear probes (Logistic Regression) to predict operators within and across regimes.
    3.  Attempted Canonical Correlation Analysis (CCA) to find shared dimensions.

## Key Results
*   **Probing Artifacts (False Positive):** Probes achieved **100% accuracy** in Think/Speak regimes, but this was proven to be **Topic Confounding**. The probes learned to distinguish "AI" from "Climate," which perfectly separated the balanced operator set.
*   **Transfer Failure:** Cross-regime mapping failed completely. Coupling maps explained **<7% of variance** ($R^2 < 0.07$).
*   **No Shared Space:** CCA alignment accuracy was **27.5%** (marginally above random 10%), indicating no significant shared manifold.

## Methodological Issues / Limitations
*   **Confounding:** The experimental design (balanced 50/50 split of topics per operator) meant that any topic detector was also a perfect operator detector for that subset.
*   **Linearity Assumption:** We assumed the transition from "Listening" to "Thinking" was a rotation or scaling. In reality, it appears to be a highly non-linear unfolding.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Failed / Invalidated**.
*   **Meaning:** "Thought" is **not** just a rotated version of the input. You cannot simply "map" the instruction vector to the reasoning vector. The geometry undergoes a fundamental topological change when the model starts generating.
*   **Shift:** This killed the "Static Vector" theory entirely. We realized we couldn't just look for *where* the state is. We had to look for *how it moves*.

## Open Questions
*   If the states aren't linearly related, how *are* they related? Do they share dynamic properties (speed, curvature) even if they don't share coordinates?

## How This Informed the Next Experiment
*   **Pivot to Dynamics:** Since static coordinates failed (Exp 1-3), we abandoned the search for "locations" and started looking for "motion." This led to **Experiment 08: Trajectory Geometry**, where we defined differential metrics like Speed and Curvature.
    *   *(Experiments 04-07 were wandering "wilderness" experiments trying to fix the confounding, which mostly confirmed the dead end.)*
