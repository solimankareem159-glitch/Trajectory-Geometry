# Experiment 05: Safety Resilience (OG-MPT Expansion)

**Phase:** 2/3 — The Pivot to Dynamics & Intervention
**Date:** November 2025
**Model:** Qwen2.5-0.5B
**Status:** Completed — **INVALID / FAILED**

## Connection to Prior Work

EXP-04 proved that Operator Gating significantly improved reasoning and constraint performance. The natural question: does orchestration also help with *safety*? If a model "Plans" and "Checks" its response, it should be harder to jailbreak.

## Research Question

**Does the OG-MPT architecture (Plan → Check → Speak) reduce Attack Success Rate against standard jailbreaks?**

## Hypotheses
1.  **H1 (Safety Buffer):** The "Check" pass will catch harmful content generated in the "Plan" pass before it is output to the user.
2.  **H2 (Robustness):** Multi-pass models will be more resistant to "frame attacks" (e.g., specific fictional scenarios) because the internal monologue will identify the true intent.
3.  **Null:** A 0.5B model is too weak to separate "simulated harm" from "actual assistance," and will either refuse everything or comply with everything regardless of gating.

## Method Summary
*   **Model:** Qwen2.5-0.5B (Baseline vs. Orchestrator).
*   **Dataset:** 54 prompts (benign control + adversarial attacks like "Fictional Sandbox", "Authority Override").
*   **Procedure:**
    *   **Detection:** Keyword-based heuristic to trigger "Safety" mode.
    *   **Gating:** System 2 flow: Identify Harm $\to$ Check Policy $\to$ Refuse/Answer.
    *   **Evaluation:** Keyword ASR (Attack Success Rate).

## Key Results
*   **Methodological Failure:** The experiment collapsed.
    *   **Model Collapse:** The 0.5B model could not handle the complex ChatML-based multi-pass prompts. It degenerated into infinite repetition loops (e.g., repeating "Gründe" 75 times) or echoed the system prompt.
    *   **Valid Baseline:** The unaligned base model had **56.8% ASR** (high vulnerability), complying with almost all "Authority" and "Sandbox" attacks.
    *   **Invalid Comparison:** Due to the collapse, we could only evaluate 15 OG-MPT samples vs 54 Baseline samples. The metrics (50% vs 56% ASR) were statistically meaningless.

## Methodological Issues / Limitations
*   **Capacity Mismatch:** We tried to force "GPT-4 level thinking architectures" onto a "0.5B parameter model." It simply lacked the instruction-following bandwidth to maintain state across 3 internal passes.
*   **Prompt Formatting:** The use of raw ChatML tags (`<|im_start|>`) confused the base model, which may not have been fine-tuned for that specific format.
*   **Scoring Bug:** A code error misclassified benign controls, rendering the "helpfulness" metric uniform zero.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Invalid / Failed**.
*   **Meaning:** Architecture isn't magic. You cannot "prompt engineer" intelligence that isn't there. Complex dynamic flows (System 2) require a minimum threshold of model capability (likely 7B+ parameters) to be stable.
*   **Shift:** We stopped trying to squeeze complex cognition out of toy models. Future architectural experiments must use models capable of stable instruction following (at least 7B or distilled 1.5B Instruct).

## Open Questions
*   Would this have worked on a larger model (e.g., 7B)? Or is the "Check" pass fundamentally flawed because the *same* model that generates the harm also validates it? (The "Self-Correction Fallacy").

## Conclusions & Implications

**Verdict: INVALID / FAILED.** Architecture isn't magic. Complex dynamic flows (System 2) require a minimum threshold of model capability (likely 7B+) to be stable. You cannot "prompt engineer" intelligence that isn't there.

## Influence on Next Experiment

*   **Boundary Clarification:** Combined with EXP-04's success, this clarified that *gating works* but *requires capacity*.
*   **The "Wilderness" Ends:** We stopped hacking prompts and returned to measuring the fundamental physics of trajectories, leading to **EXP-08** where we finally measured the geometry of "thought" tokens.
