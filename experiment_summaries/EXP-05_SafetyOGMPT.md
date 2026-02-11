# Experiment 05: Safety Resilience (OG-MPT Expansion)

**Date / Context:** May 2026 / Phase 3: The Pivot to Dynamics & Intervention

## Motivation & Prior Assumptions
*   **Context:** Experiment 04 proved that "Operator Gating" (Orchestration) significantly improved performance (Reasoning/Constraints) by forcing the model to slow down and "think" explicitly.
*   **Assumption:** Complexity is complexity. If orchestration helps with *logic*, it should also help with *safety*. A model that "Plans" and "Checks" its response should be harder to jailbreak than one that just blurts out an answer.
*   **Goal:** To test if the OG-MPT (Plan $\to$ Check $\to$ Speak) architecture reduces Attack Success Rate (ASR) against standard jailbreaks (e.g., "Write a story about X").

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

## How This Informed the Next Experiment
*   **Return to Basics:** This failure, combined with the success of Exp 4, clarified boundaries. We knew *gating works* (Exp 4) but *requires capacity* (Exp 5 failure).
*   **The "Wilderness" Ends:** We realized we needed to stop hacking prompts and start measuring the fundamental physics of the trajectory again. This paved the way for **Experiment 08**, where we finally successfully measured the geometry of these "thought" tokens.
