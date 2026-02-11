# MVP Experiment Specification: Geometric Signatures in Embedding Space

This document outlines the protocol for the "Geometric Signatures" experiment, designed to test if conversational moves have distinct, predictable geometric representations.

## 1. Hypotheses
We propose three falsifiable hypotheses:

*   **H1 (Operator Existence):** Conversational "warp vectors" (difference between current and previous state embeddings) cluster into stable families corresponding to the underlying operator move (e.g., "Summarize", "Refactor"), invariant to phrasing.
*   **H2 (Operator Predictability):** A predictive model can determine the next operator cluster significantly better than random or majority-class baselines.
*   **H3 (Cross-Topic Invariance):** These operator clusters remain stable across different conversational topics (e.g., the "Summarize" vector looks similar in a "Physics" conversation and a "Cooking" conversation).

## 2. Falsification Criteria
The theory is considered falsified if:
*   **Unstable Clusters:** Clustering algorithms (HDBSCAN/k-means) fail to find consistent structures across different random seeds or initialization parameters.
*   **Topic Dominance:** Clusters align primarily with *Topic* (e.g., all "Physics" turns cluster together) rather than *Operator* (content dominates structure).
*   **Prediction Failure:** A sequence model fails to predict the next operator cluster with accuracy exceeding simple baselines (Majority Class, Markov-1).

## 3. Operator List (10)
We will use 10 distinct, high-contrast operators:
1.  **Clarify:** Ask questions to disambiguate or better understand.
2.  **Summarize:** Condense the previous context.
3.  **Reframe:** Shift the perspective or interpretation.
4.  **Zoom Out:** Broaden the context; look at the big picture.
5.  **Zoom In:** Focus on details or specific steps.
6.  **Steelman:** Construct the strongest possible version of an opposing argument.
7.  **Devil’s Advocate:** Propose counter-arguments or critique the current view.
8.  **Formalize:** Convert natural language into equations, definitions, or code.
9.  **Decompose:** Break a complex problem into sub-problems.
10. **Plan:** Outline a sequence of actionable steps.

## 4. Dataset Design
To ensure robustness against phrasing and topic effects, we will construct a synthetic dataset:
*   **Dimensions:**
    *   **10 Operators** (as listed above)
    *   **10 Paraphrases** per operator (e.g., "Reframe" -> "View this differently", "Spin this positively")
    *   **5 Topics** (AI Safety, Relationships, Career, Politics, Health)
*   **Total Size:** 10 Operators × 10 Paraphrases × 5 Topics = **500 Operator Turns**

## 5. Embedding State Definitions
We will compute state embeddings ($s_t$) using three definitions to test robustness:
1.  **Assistant-Only:** $s_t = \text{embed}(a_t)$ (The assistant's response text).
2.  **Pair:** $s_t = \text{embed}(u_t + "\backslash n\backslash n" + a_t)$ (The user prompt + assistant response).
3.  **Rolling Window:** $s_t = \text{embed}(text_{t-1} + text_t)$ (Last $k=2$ turns concatenated).

The "Warp Vector" is defined as $\Delta_t = s_t - s_{t-1}$.

## 6. Metrics
### Clustering Quality
*   **Stability:** Consistency of cluster assignments across seeds.
*   **Silhouette Score:** Measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
*   **Purity (Operator vs. Topic):** We expect high purity for Operator labels and low purity for Topic labels within the Warp-space clusters.

### Prediction Quality
*   **Accuracy:** Percentage of correct next-operator predictions.
*   **Macro-F1:** Harmonic mean of precision and recall, averaged across classes (handles class imbalance better).
*   **Calibration:** (If probabilistic) Agreement between predicted probabilities and observed frequencies.
