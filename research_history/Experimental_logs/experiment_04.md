# Experiment 04: Operator-Gated Multi-Pass Thinking (OG-MPT)

**Date / Context:** April 2026 / Phase 3: The Pivot to Dynamics & Intervention

## Motivation & Prior Assumptions
*   **Context:** Experiments 01-03 failed to find static "thought vectors" that could be passively monitored. We realized that "thought" is an active process, not a passive state.
*   **Methodological Shift:** Following the failures of Exp 1-3, we overhauled our experimental design. We began using **ChatGPT metaprompting** to generate rigorous, adversarially-filtered datasets and to critique our evaluation logic, ensuring we weren't just measuring noise.
*   **Assumption:** If we can't *find* the operator in the vector space, maybe we can *force* it. If we explicitly tell the model *how* to think (e.g., "Plan first, then Check, then Speak") based on the task type, we should see performance gains that passive scaling cannot achieve.
*   **Goal:** To implement an "Orchestrator" that detects the task type (Reasoning, Constraint, Safety) and dynamically gates the model's internal monologue into specific steps.

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

## How This Informed the Next Experiment
*   **Closing the Loop:** Now that we proved *behavioral* intervention works (Exp 4), we wanted to return to *measurement*. If we force the model to "Think," does it generate the "Trajectory" we hypothesized in Exp 1?
*   **Next Step:** This led to **Experiment 08: Trajectory Geometry**, where we finally successfully measured the *geometric properties* (Speed, Curvature) of these successful multi-pass trajectories.
