# Experiment 01: Geometric Signatures (Turn-Level)

**Date / Context:** February 2026 / Phase 1: The Intuition of "Shapes"

## Motivation & Prior Assumptions
*   **Context:** We suspected that different cognitive operations (e.g., *Summarize*, *Critique*) enact distinct transformations on information.
*   **Assumption:** These transformations should leave a measurable "geometric signature" in the embedding space of the model's output.
*   **Goal:** To detect and classify cognitive operators solely by analyzing the "warp" (distance vector) between consecutive conversational turns.

## Hypotheses
1.  **H1 (Clustering):** Vectors representing the same operator (e.g., "Summarize") will cluster together, distinct from other operators.
2.  **H2 (Robustness):** The signature will remain stable across different paraphrases and topics (Small within-operator variance).
3.  **Null:** Warp vectors will be dominated by semantic content (Topic/Lexical features) and will not cluster by operator.

## Method Summary
*   **Model:** Gemini API (`text-embedding-004`).
*   **Dataset:** 500 independent single-turn prompts (10 operators × 10 paraphrases × 5 topics).
*   **Procedure:**
    1.  Artificially concatenated single turns to simulate "conversations" (Topic: AI Safety).
    2.  Computed "Warp" vectors $w_t = E(response_t) - E(turn_t)$.
    3.  Applied Clustering (HDBSCAN, K-Means) and Classification (GRU predictor).
*   **Metrics:** Silhouette Score, Adjusted Mutual Information (AMI), Within/Between-Cluster Variance Ratio.

## Key Results
*   **Clustering Failed:** K-Means AMI was **0.13** (barely above random). HDBSCAN found **0 clusters**.
*   **Robustness Failed:** Within-operator variance (0.296) was **higher** than between-operator variance (0.112).
    *   *Meaning: Two paraphrases of "Summarize" were more different than "Summarize" vs "Critique".*
*   **Prediction Ambiguity:** Sequence predictors (GRU) achieved **57-66% accuracy** (vs 10% random), but likely learned transition artifacts rather than geometry.

## Methodological Issues / Limitations
*   **Artificiality:** The "conversations" were concatenated independent turns, meaning no actual internal state carried over.
*   **Measurement Error:** `text-embedding-004` captures semantic content. The "warp" measured lexical differences in the *prompt*, not the cognitive act of the *model*.
*   **Redundancy:** State definitions (window sizes) were effectively identical due to lack of history.

## Interpretation (What This Actually Meant)
*   **Verdict:** **Invalid**. The initial intuition was not supported by the data.
*   **Meaning:** "Thought" does not manifest as a simple, static additive vector in the output embedding space. The signal is swamped by topic and lexical noise.
*   **Shift:** We cannot rely on API embeddings or output text. We must look "inside" the model (hidden states) to find the operator.

## Open Questions
*   If the signatures aren't in the *output* embeddings, are they in the *internal* activations?
*   Are operators monolithic shapes, or combinations of smaller primitive factors?

## How This Informed the Next Experiment
*   **Pivot to Internals:** We moved from API embeddings to open-weights models to access hidden states.
*   **Pivot to Decomposition:** This failure led to **Experiment 02: Latent Factors**, where we used NMF to decompose internal states into primitive components, suspecting the signal was compositional rather than monolithic.
