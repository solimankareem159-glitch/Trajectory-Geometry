# Experiment 04: Operator-Gated Multi-Pass Thinking (OG-MPT)

**Phase:** 2/3 — The Pivot to Dynamics & Intervention
**Date:** November 2025
**Model:** Qwen2.5-0.5B
**Status:** Completed — **SUCCESS**

## Connection to Prior Work

EXP 01-03 failed to find static "thought vectors" that could be passively monitored. We realized that "thought" is an active process, not a passive state. Following these failures, we overhauled the experimental design — using **ChatGPT metaprompting** to generate adversarially-filtered datasets and critique evaluation logic.

## Research Question

**If we can't *find* the operator in the vector space, can we *force* it?** If we explicitly tell the model *how* to think based on the task type, do we see performance gains that passive scaling cannot achieve?

## Hypotheses
1.  **H1 (Gating Efficacy):** A small model (0.5B) using multi-pass gating will outperform its single-pass baseline.
2.  **H2 (Task Specificity):** Different tasks benefit from different "thought structures" (e.g., Constraints need a "Check" pass; Reasoning needs a "Decompose" pass).
3.  **Null:** More tokens $\neq$ better performance; the model will just hallucinate more elaborately.

## Method Summary
*   **Model:** Qwen2.5-0.5B (Baseline vs. Orchestrator).
*   **Dataset:** 60 rigorous prompts (Math, Constraints, Safety) generated via ChatGPT metaprompting.
*   **Procedure:**
    *   **Detection:** A heuristic probe classified the input (e.g., "Math" $\to$ Reasoning).
    *   **Gating:** The Orchestrator forced a multi-turn conversation:
        *   *Reasoning:* Plan $\to$ Calculate $\to$ Verify $\to$ Answer.
        *   *Constraint:* List Constraints $\to$ Draft $\to$ Check Count $\to$ Answer.
        *   *Safety:* Identify Harm $\to$ Check Policy $\to$ Refuse/Answer.
    *   **Evaluation:** Strict regex and programmatic checking.

## Key Results
*   **Overall Acc:** Orchestrator (**76.6%**) crushed the Baseline (**53.3%**). **(+23.3%)**
*   **Constraint Mastery:** The biggest gain was in formulation constraints (e.g., "no sentences with the letter 'e'").
    *   Baseline: **40%**
    *   Orchestrator: **85%** (+45%)
*   **Reasoning Gains:** Math/Logic improved by **+25%**.

## Methodological Issues / Limitations
*   **Sample Size:** While rigorous, N=60 is still small.
*   **Heuristics:** The "Orchestrator" used simple regex/keywords for detection, not a learned probe. It was a "System 2" simulation, not a true internal circuit.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Success**.
*   **Meaning:** "Cognitive Capability" is not just about raw weights; it's about **control flow**. A small model can punch above its weight class if you structure its "thinking" process explicitly.
*   **Shift:** This confirmed that **Dynamics > Statics**. We don't need to *find* the "Constraint Vector" to use it; we just need to *trigger* the "Constraint Behavior."

## Open Questions
*   Can we automate this? Can the model learn to facilitate its *own* multi-pass thinking without a hard-coded Orchestrator?
*   Does this "forced thought" produce the geometric signatures we were looking for in Exp 1-3? (i.e., does the "Check" pass have a distinct shape?)

## Conclusions & Implications

**Verdict: SUCCESS.** "Cognitive Capability" is not just about raw weights — it's about **control flow**. A small model can punch above its weight class if you structure its "thinking" process explicitly. This confirmed that **Dynamics > Statics**.

## Influence on Next Experiment

*   **Closing the Loop:** We proved *behavioral* intervention works (EXP-04), so we returned to *measurement*. If we force the model to "Think," does it generate the "Trajectory" we hypothesized in EXP-01?
*   **EXP-05** tested whether gating also helps with safety (it didn't — capacity mismatch).
*   **EXP-08** finally measured the *geometric properties* (Speed, Curvature) of these successful multi-pass trajectories.
