# Experiment 6: Mid-Layer Conversational Operator Decodability

**Status:** DRAFT  
**Principal Investigator:** Antigravity (Agentic AI)  
**Date:** 2026-01-27  

## 1. Abstract

This protocol defines the methodology for Experiment 6, designed to test the hypothesis that conversational operators (e.g., summarize, critique, reframe) are linearly decodable from mid-layer transformer representations, independent of topic content and instruction phrasing. The study employs a rigorous crossed experimental design with strict controls for lexical leakage and content confounds.

## 2. Research Question

Can the intended conversational operator be predicted from mid-layer hidden-state dynamics (residuals, attention outputs, MLP outputs) when topic, base content, and instruction phrasing are strictly controlled and generalized across?

## 3. Hypothesis

**H1 (Directional):** There exists a contiguous band of transformer layers where the abstract identity of a conversational operator is linearly or shallowly decodable from internal state dynamics. This decoding performance must:

1. Significantly exceed chance and content-only baselines.
2. Generalize across unseen topics (content hold-out).
3. Generalize across unseen instruction phrasings (paraphrase hold-out).
4. Survive instruction masking controls (indicating latent abstraction rather than lexical pattern matching).

**Falsification Condition:** The hypothesis is rejected if decoding accuracy collapses to the baseline level under the joint content-and-paraphrase held-out split, or if the "Instruction Masking" control yields performance equivalent to the full prompt condition (implying reliance on surface-level lexical cues).

## 4. Dataset Design

The experiment uses a fully crossed factorial design ($Operators \times Contents \times Paraphrases$).

### 4.1. Factors

* **Operators ($O=10$):**
    1. Summarize
    2. Critique
    3. Reframe
    4. Decompose
    5. Compare
    6. Extend
    7. Translate
    8. Exemplify
    9. Formalize
    10. Question
* **Base Contents ($C \ge 20$):** Short passages (~100-200 tokens) covering diverse domains (e.g., technical documentation, fiction, news, dialogue, legal text).
* **Instruction Paraphrases ($P \ge 10$):** Distinct phrasings for each operator (e.g., "Summarize this", "Give me the gist", "TL;DR").

### 4.2. Sample Size

* Total Prompts $\approx 10 \times 20 \times 10 = 2,000$.
* Generation Settings: Fixed temperature (0.0 for deterministic analysis, or low e.g., 0.2), fixed token cap (sufficient to capture onset dynamics, e.g., 128 new tokens).

## 5. Representation Extraction

Data is collected for each Prompt-Response pair ($x_i, y_i$).

### 5.1. Target Window

Analysis focuses on the **early response window** (e.g., first 32 generated tokens). This isolates the *activation* of the operator mode before the specific content dominates.

### 5.2. Feature extraction ($h_L^{(t)}$)

For each layer $L$ and token position $t$:

1. **Directional Deltas:**
    $$ \Delta h_L(t) = \text{Norm}(h_L(t) - h_L(t-1)) $$
    *Rationale:* Captures the *process* or *trajectory* of computation, reducing static content bias.

2. **Residual Stream Components:**
    * Separate extraction of Attention Output ($Attn_L$) and MLP Output ($MLP_L$) where model architecture permits.

3. **Baseline-Subtracted Features:**
    $$ h_{diff} = h_{operator} - h_{neutral} $$
    * $h_{neutral}$: Hidden state from a "Continue this text" (neutral) condition on the *same* content.
    * *Rationale:* Removes the shared "content manifold" to isolate the operator vector.

## 6. Decoding & Splits

### 6.1. Models

* **Primary:** Linear Probe (Logistic Regression with L2 regularization).
* **Secondary:** Shallow MLP (1 hidden layer) to test for non-linear but readily accessible information.

### 6.2. Cross-Validation Splits (Mandatory)

1. **Cross-Topic (Standard):** Train on $C_{train}$, Test on $C_{test}$. Tests independence from specific subject matter.
2. **Cross-Paraphrase (Robustness):** Train on $P_A$, Test on $P_B$. Tests independence from specific trigger words.
3. **Joint Hold-out (Strict):** Train on ($C_{train} \times P_{train}$), Test on ($C_{test} \times P_{test}$). THIS IS THE CRITICAL TEST FOR GENERALIZATION.

## 7. Controls (The "Allergy Test")

Every positive result must survive these controls:

1. **Instruction Masking:** Replace the specific operator formatting with a uniform placeholder (e.g., [OP]) in the embedding space (if possible) or use a neutral prompt but label it as the operator. *Expectation:* Decoding should fail (chance level).
2. **Content-Only Decoding:** Attempt to decode Operator ID from the *content* tokens alone (before instruction). *Expectation:* Chance level.
3. **Label Permutation:** Shuffle labels $y$ while keeping $X$ fixed. *Expectation:* Chance level (validates the probe is not overfitting noise).
4. **Lexical Baseline:** TF-IDF/Embedding of the instruction text only. *Expectation:* The internal state probe should significantly outperform this surface-level baseline in the Joint Hold-out setting.

## 8. Interpretation Logic

* **Success:** High accuracy in Joint Hold-out + low accuracy in Controls = Evidence for a distinct, abstract "Operator" geometry.
* **Partial Success:** High accuracy in Cross-Topic but drops in Cross-Paraphrase = The model is memorizing instruction templates, not the abstract operator.
* **Failure:** Accuracy tracks Content-Only control or fails Permutation test = Spurious correlation; no evidence of operator emergence.

## 9. Metrics

* **Accuracy** vs. Chance (10%).
* **Macro-F1 Score**.
* **Bootstrap Confidence Intervals** (95%).
* **Confusion Matrices** (to identify similar operators, e.g., distinct "Summarize" vs "Decompose" clusters).
