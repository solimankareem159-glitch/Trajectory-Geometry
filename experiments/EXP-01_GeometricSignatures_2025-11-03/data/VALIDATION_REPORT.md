# Geometric Signatures Experiment - Validation Report

## Executive Summary

**Status: CONCLUSIONS NOT FULLY SUPPORTED BY DATA**

The claims made in `walkthrough.md` about "distinguishable warp signatures" and "robust prediction" are overstated relative to the actual experimental results documented in the reports.

---

## 1. Claims vs. Evidence

### Claim 1: "Operators produce distinguishable 'warp signatures'"

**Evidence from CLUSTER_REPORT.md:**

| Metric | Value | Interpretation |
|:---|:---|:---|
| HDBSCAN Clusters | 0 | No natural cluster structure detected |
| K-Means Silhouette | 0.016-0.019 | Near-zero = no meaningful clustering |
| AMI (Operator) | 0.13-0.14 | Very weak alignment (max=1.0) |
| AMI (Topic) | ~0.00 | Negatives indicate random |

**Verdict: ❌ NOT SUPPORTED**

If operators produced truly distinguishable signatures, we would expect:

- Silhouette > 0.2 (ideally > 0.5)
- AMI > 0.3
- HDBSCAN to identify at least some clusters

The AMI of 0.13 means cluster assignments share only 13% mutual information with operator labels—barely above random.

---

### Claim 2: "Signatures were robust across paraphrases"

**Evidence from ROBUSTNESS_REPORT.md:**

| State Def | Within-Op Variance | Between-Op Variance | Ratio |
|:---|:---|:---|:---|
| a | 0.296 | 0.112 | **0.38** |
| b | 0.253 | 0.136 | **0.54** |
| c | 0.253 | 0.136 | **0.54** |

| State Def | Mean Dist (Within) | Mean Dist (Between) |
|:---|:---|:---|
| a | 0.756 | 0.755 |
| b | 0.698 | 0.701 |
| c | 0.698 | 0.701 |

**Verdict: ❌ EXPLICITLY FAILED**

The report itself marks all tests with ❌ **FAIL**. Within-operator variance exceeds between-operator variance, meaning:

- **Two different operators** are just as similar in warp space as
- **Two paraphrases of the same operator**

This is the opposite of "robust signatures."

---

### Claim 3: "Predictive of next operator in sequence"

**Evidence from PREDICTOR_REPORT.md:**

| Model | Accuracy | Macro-F1 |
|:---|:---|:---|
| Markov-1 Baseline | 37.8-54.1% | 0.15-0.20 |
| GRU (N=5) | 57.4-66.0% | 0.25-0.56 |

**Verdict: ⚠️ PARTIALLY SUPPORTED BUT MISLEADING**

Issues:

1. **Random baseline for 10 classes = 10%**. Both models beat random, but this could be due to:
   - Markov transition probabilities (operator A → operator B patterns)
   - Topic correlations
   - NOT necessarily geometric features

2. **Macro-F1 is low** (0.25-0.56), indicating poor performance on minority classes.

3. **No ablation**: The study doesn't verify whether the GRU uses geometric features vs. just memorizing operator transition statistics.

---

## 2. Methodological Issues

### 2.1 Artificial "Conversations"

The dataset consists of **500 independent single-turn prompts** (10 operators × 10 paraphrases × 5 topics), artificially concatenated within topics to simulate conversations.

```
Topic: AI Safety
  Turn 0: Clarify paraphrase 0 → response
  Turn 1: Summarize paraphrase 0 → response
  Turn 2: Reframe paraphrase 0 → response
  ...
```

This is NOT a conversation—each turn is independent. The "warp" (difference between consecutive embeddings) is measuring:

- Transition from one topic-operator-paraphrase to another
- NOT internal model dynamics during dialogue

### 2.2 State Definition Redundancy

```python
text_c = text_b  # Lines 88-89 of embed_states.py
```

State definitions 'b' and 'c' are identical because "rolling k=2 window" equals "current turn" when there's no conversation history. This explains why their results are identical.

### 2.3 What "Warp" Actually Measures

The embedding model (`text-embedding-004`) encodes semantic content. The "warp" between turns measures:

```
warp_t = embed("Summarize... AI Safety response") - embed("Clarify... AI Safety response")
```

This is dominated by:

1. **Lexical differences** in the instruction text ("Summarize" vs "Clarify")
2. **Response content differences**

NOT any deep "cognitive signature" of the operator.

---

## 3. What The Data Actually Shows

**The data shows that:**

1. Operators are NOT geometrically distinguishable in Gemini's embedding space
2. Paraphrases of the same operator are NOT more similar than different operators
3. Sequence prediction works marginally better than Markov, possibly due to transition patterns

**The data does NOT show:**

- That conversational operators have "distinct geometric trajectories"
- That these signatures are "robust" to content variation
- That geometry (vs. transition statistics) drives prediction

---

## 4. Conclusions

| Original Claim | Validity | Corrected Interpretation |
|:---|:---|:---|
| "Distinct geometric profiles" | ❌ Invalid | Clusters nearly random; AMI = 0.13 |
| "Robust to paraphrases" | ❌ Invalid | Within-variance > between-variance |
| "Predictable sequences" | ⚠️ Partial | Beats random, but mechanism unclear |

### Recommendations

1. **Retract or qualify walkthrough claims**: The statements about "distinguishable signatures" should be marked as unvalidated hypotheses.

2. **Run proper ablation**: Test if prediction works on:
   - Just operator labels (no geometry)
   - Just transition probabilities
   - Geometry features alone

3. **Use actual conversations**: Test with real multi-turn dialogues, not single-turn prompts concatenated.

4. **Consider different metrics**: If seeking operator signatures, try:
   - Supervised linear probes on embeddings
   - Contrastive learning to learn operator-specific representations
   - Token-level hidden states (not just text embeddings)

---

## 5. Update to Walkthrough

The following claims should be revised:

**Original:**
> "Operators show distinguishable warp traces."

**Corrected:**
> "Initial clustering analysis showed weak separation (AMI=0.13); robustness tests failed, indicating operator signals are not clearly distinguishable in the embedding space."

**Original:**
> "These signatures were robust enough to predict the next operator."

**Corrected:**
> "Sequence models outperformed Markov baselines (57-66% vs 38-54%), but it's unclear whether this is due to geometric features or operator transition statistics."
