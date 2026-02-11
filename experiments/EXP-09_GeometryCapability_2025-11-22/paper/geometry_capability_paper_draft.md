# Trajectory Geometry Correlates with Reasoning Success Under Chain-of-Thought Prompting

---

## Abstract

Chain-of-thought (CoT) prompting improves language model performance on multi-step reasoning tasks, but the internal changes associated with this improvement are not well characterized. We examine whether trajectory-level properties of hidden state evolution differ between successful and failed reasoning attempts. Using 300 multi-step arithmetic problems with a 0.5B parameter transformer, we measure representational speed, directional consistency, and stabilization rate across layers under both direct-answer and CoT prompting conditions. We find that successful CoT responses are associated with higher representational speed (Cohen's d = 2.66–4.25), lower directional consistency (d = −2.58 to −2.63), and more positive stabilization rates (d = 0.69–2.03) compared to failed direct responses. These correlations are consistent across mid-layers and strongest in the final layer. We interpret these findings cautiously: they suggest that trajectory geometry may provide a useful descriptive lens for characterizing internal computation, though we make no claims about causation, generalization, or underlying mechanisms.

---

## 1. Introduction

Chain-of-thought prompting elicits intermediate reasoning steps from language models and is associated with improved performance on multi-step reasoning tasks (Wei et al., 2022). Despite widespread practical use, the internal computational changes accompanying this improvement remain poorly characterized.

Most prior analysis has focused on behavioral outcomes—comparing final answer accuracy, evaluating reasoning trace coherence, or testing faithfulness of stated reasoning (Turpin et al., 2024). These approaches provide valuable information but offer limited insight into what changes inside the model during generation.

This paper takes a different approach: we examine *trajectory-level* properties of hidden state evolution during reasoning. Rather than probing for specific features at individual token positions, we measure how hidden states change across the generation sequence. This allows us to characterize properties like speed of representational movement, directional structure, and convergence patterns.

### Research Question

We ask: **Are successful reasoning attempts associated with different trajectory geometry than failed attempts? If so, are these differences more pronounced under CoT prompting?**

### Why Arithmetic?

We study multi-step arithmetic because it offers important experimental controls:

- Ground truth is unambiguous
- Success and failure can be determined automatically
- Performance variation exists within a single model (neither ceiling nor floor)
- Confounds from world knowledge or subjective interpretation are minimal

### What This Paper Does Not Claim

We emphasize what this study does *not* establish:

- **No causal claims**: We report correlations between trajectory geometry and outcomes. We do not claim that geometric properties cause success or failure.
- **No mechanistic interpretation**: We do not claim to know *why* these correlations exist or what computational processes they reflect.
- **No generalization**: Results are from a single model on a narrow task. Generalization to other architectures, scales, or domains is untested.
- **No discrete operators**: We do not claim that the model implements discrete "cognitive operators" or that trajectory analysis reveals discrete computational states.
- **No claims about awareness or experience**: This analysis concerns statistical properties of vector sequences. We make no claims about subjective states, awareness, or any phenomenal properties.

With these limitations clearly stated, we proceed to describe our methodology and findings.

---

## 2. Related Work

### Chain-of-Thought Prompting

Wei et al. (2022) introduced chain-of-thought prompting, showing that eliciting intermediate steps improves performance on reasoning benchmarks. Subsequent work has explored zero-shot variants (Kojima et al., 2022), self-consistency through sampling (Wang et al., 2023), and factors affecting reasoning quality (Jin et al., 2024). These studies characterize behavioral effects rather than internal dynamics.

### Mechanistic Interpretability

Mechanistic interpretability seeks to understand neural network computation by analyzing internal structure (Elhage et al., 2022; Nanda et al., 2023). Techniques include activation patching, circuit analysis, and feature probing. Most work examines static representations at individual positions rather than dynamics across sequences.

### Probing Limitations

Classical probing trains classifiers on hidden states to detect specific features (Belinkov, 2022). Probing is inherently position-wise and may miss properties that emerge from temporal structure. Trajectory analysis complements probing by characterizing sequential dynamics.

### Relevant Caveats

Prior work has shown that:

- Stated reasoning may not faithfully reflect actual computation (Turpin et al., 2024)
- Simple probes can detect confounds rather than intended features (Belinkov, 2022)
- Interpretability findings may not generalize across scales (Anthropic, 2024)

We attempt to address these concerns through our experimental design but acknowledge that undetected confounds may remain.

---

## 3. Methodology

### 3.1 Model

We use **Qwen2.5-0.5B**, a decoder-only transformer with 24 layers. This scale enables efficient experimentation while exhibiting meaningful performance variation on our task.

### 3.2 Task

**Multi-step arithmetic** with the following structure:

- Form: `(A × B) + C` or `A × (B + C)`
- Operand ranges: A ∈ [10, 50], B ∈ [2, 20], C ∈ [10, 100]
- Second operation: randomly addition or subtraction
- Answers range from approximately −4000 to +4500

This produces problems requiring intermediate computation, with error rates high enough to generate ample failure cases.

### 3.3 Prompt Conditions

**Direct Answer Condition:**

```
Answer the following question directly.
Q: Calculate (38 * 9) + 46
A:
```

Maximum 32 generated tokens.

**Chain-of-Thought Condition:**

```
Q: Calculate (38 * 9) + 46
Let's think step by step before answering.
```

Maximum 128 generated tokens.

**Decoding:** Greedy (temperature = 0) for reproducibility.

### 3.4 Outcome Labeling

We extract the final integer from the generated text using regex. An outcome is labeled **Success** if this integer exactly matches the ground-truth answer, otherwise **Failure**.

### 3.5 Hidden State Extraction

For each response, we perform a forward pass with hidden state extraction. We analyze:

- Layer 0 (embedding)
- Layers 10–16 (mid-network)
- Layer 24 (final)

We extract states for the **first 32 generated tokens** to control for response length variation.

### 3.6 Trajectory Metrics

Given hidden states **h**₁, **h**₂, ..., **h**_T:

**Speed**: Mean L2 norm of consecutive differences
$$\text{Speed} = \frac{1}{T-1} \sum_{t=1}^{T-1} \|\mathbf{h}_{t+1} - \mathbf{h}_t\|_2$$

**Directional Consistency**: Norm of the mean normalized direction vector
$$\text{DC} = \left\|\text{mean}\left(\frac{\Delta\mathbf{h}_t}{\|\Delta\mathbf{h}_t\|}\right)\right\|$$
Values near 1 indicate consistent direction; values near 0 indicate varied directions.

**Early Curvature**: Mean angular deviation between consecutive direction vectors (tokens 1–16)
$$\text{Curvature} = 1 - \text{mean}(\cos(\Delta\mathbf{h}_t, \Delta\mathbf{h}_{t+1}))$$

**Stabilization Rate**: Slope of linear fit to displacement norms over time
Negative values indicate decreasing displacement (convergence); positive values indicate increasing displacement.

### 3.7 Statistical Analysis

**Sample Size**: 300 problems × 2 conditions = 600 total responses.

**Group Definitions**:

| Group | Prompt | Outcome |
|-------|--------|---------|
| G1 | Direct | Failure |
| G2 | Direct | Success |
| G3 | CoT | Failure |
| G4 | CoT | Success |

**Primary Comparison**: G4 (CoT Success) vs G1 (Direct Failure)

This comparison maximizes contrast between best-case and worst-case outcomes.

**Test**: Two-tailed permutation test (k = 1,000 shuffles).

**Effect Size**: Cohen's d with pooled standard deviation.

We report all comparisons without multiple comparison correction but interpret only effects with p < 0.01 and |d| > 0.8 as robust.

---

## 4. Results

### 4.1 Group Sizes

| Group | n | Proportion |
|-------|---|------------|
| G1 (Direct Failure) | ~276 | 92% |
| G2 (Direct Success) | ~24 | 8% |
| G3 (CoT Failure) | ~54 | 18% |
| G4 (CoT Success) | ~246 | 82% |

CoT prompting is associated with substantially higher accuracy (82% vs 8%).

### 4.2 Primary Comparison: G4 vs G1

| Layer | Metric | G4 Mean | G1 Mean | p | Cohen's d |
|-------|--------|---------|---------|---|-----------|
| 0 | Speed | 0.56 | 0.42 | <0.001 | 2.68 |
| 0 | Dir. Consistency | 0.05 | 0.51 | <0.001 | −2.60 |
| 0 | Stabilization | 0.00 | −0.12 | <0.001 | 1.60 |
| 10 | Speed | 13.69 | 12.14 | <0.001 | 3.12 |
| 10 | Dir. Consistency | 0.05 | 0.51 | <0.001 | −2.61 |
| 10 | Stabilization | −0.02 | −0.60 | <0.001 | 1.04 |
| 13 | Speed | 14.28 | 12.49 | <0.001 | 3.26 |
| 13 | Dir. Consistency | 0.06 | 0.51 | <0.001 | −2.59 |
| 13 | Stabilization | −0.01 | −0.63 | <0.001 | 1.10 |
| 16 | Speed | 17.27 | 15.20 | <0.001 | 2.99 |
| 16 | Dir. Consistency | 0.06 | 0.52 | <0.001 | −2.58 |
| 16 | Stabilization | 0.01 | −0.93 | <0.001 | 1.15 |
| 24 | Speed | 190.89 | 130.87 | <0.001 | 4.25 |
| 24 | Dir. Consistency | 0.05 | 0.53 | <0.001 | −2.63 |
| 24 | Stabilization | 0.66 | −24.01 | <0.001 | 2.03 |

All comparisons significant at p < 0.001 with large effect sizes.

### 4.3 Summary of Findings

**Speed**: G4 samples show higher displacement norms across all layers. Effect sizes range from d = 2.66 to d = 4.25, increasing through deeper layers.

**Directional Consistency**: G1 samples show consistency values near 0.51 (movement in a consistent direction). G4 samples show values near 0.05 (movement in varied directions). This pattern is consistent across all layers (d ≈ −2.6).

**Stabilization**: G1 samples show negative stabilization (displacement increasing or erratic). G4 samples show near-zero or positive stabilization (displacement stable or decreasing). The effect is largest in Layer 24 (d = 2.03).

**Curvature**: Smaller effects (d ≈ 0.4–0.8), present but less discriminative than other metrics.

### 4.4 Layer Patterns

Effects are present at Layer 0 but strengthen through mid-layers (10–16) and are most pronounced at Layer 24. Speed increases by approximately 45% from Layer 0 to Layer 24 in G4 relative to G1.

---

## 5. Discussion

### 5.1 Interpretation of Findings

We observe robust correlations between trajectory geometry and reasoning outcomes:

1. **Higher speed** in successful CoT samples indicates larger step-to-step changes in hidden states.

2. **Lower directional consistency** in successful samples indicates that the direction of change varies more across steps.

3. **More positive stabilization** in successful samples indicates that displacement magnitude tends to decrease or remain stable rather than increase.

These correlations are consistent, large in effect size, and present across multiple layers.

### 5.2 What These Correlations Might Reflect

We offer speculative interpretations while emphasizing their tentative status:

- High directional consistency in failed samples *might* reflect repetitive or template-based generation patterns
- High speed in successful samples *might* reflect more extensive transformation of representations during reasoning
- Positive stabilization *might* reflect convergence toward a stable solution state

These interpretations are hypotheses, not conclusions.

### 5.3 What These Findings Do Not Establish

**Causation**: We observe that success correlates with certain geometric properties. We do not know whether:

- These properties causally contribute to success
- Success causes these properties
- Both are effects of a common underlying factor

**Mechanism**: We do not know what computational processes produce these geometric signatures. The metrics are purely descriptive.

**Generalization**: Our results come from one model on one task. Different architectures, scales, training procedures, or task domains may yield different patterns.

**Discrete States**: We find no evidence for discrete "cognitive operators" or distinct computational modes. Trajectory metrics vary continuously.

### 5.4 Relationship to Prior Work

These findings are consistent with prior observations that:

- CoT prompting changes model behavior substantially (Wei et al., 2022)
- Internal representations evolve in complex ways during generation (Anthropic, 2024)
- Trajectory-level analysis can reveal properties invisible to point-wise probing

Our contribution is to demonstrate specific, measurable geometric correlates of reasoning success.

### 5.5 Potential Applications

If trajectory-geometric correlations prove robust across settings, they could:

- Provide intrinsic signals of reasoning quality without output parsing
- Enable early detection of likely failures during generation
- Offer a descriptive vocabulary for characterizing internal computation

These applications require validation beyond our current scope.

### 5.6 Limitations

**Single model**: Qwen2.5-0.5B only. No testing at other scales or architectures.

**Narrow task**: Multi-step arithmetic. No testing on language-heavy reasoning, planning, or creative tasks.

**Correlational design**: No causal interventions.

**Fixed window**: First 32 tokens only. Dynamics in longer traces not examined.

**Potential confounds**: Response length, lexical patterns, or other factors may covary with both geometry and outcomes.

**No replication**: Single experimental run.

---

## 6. Conclusion

We have shown that successful chain-of-thought reasoning in a 0.5B parameter transformer is correlated with distinct trajectory geometry: higher representational speed, lower directional consistency, and more stable displacement patterns. These correlations are large (|d| > 2 for most metrics), consistent across mid-layers, and strongest in the final layer.

We interpret these findings as evidence that trajectory-level analysis provides useful descriptive information about model computation during reasoning. We explicitly decline to make causal, mechanistic, or generalizable claims beyond our experimental scope.

Future work should test whether these correlations:

- Replicate across models, scales, and training procedures
- Generalize to diverse reasoning tasks
- Survive causal intervention (e.g., trajectory perturbation during generation)
- Provide practical value for monitoring or steering model behavior

---

## References

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.

Kojima, T., Gu, S.S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. *NeurIPS*.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR*.

Turpin, M., Michael, J., Perez, E., & Bowman, S.R. (2024). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS*.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2022). A mathematical framework for transformer circuits. *Anthropic*.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR*.

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*.

Jin, Z., Ren, J., Cai, D., Chen, Z., Xie, P., & Neubig, G. (2024). Impact of reasoning step length on large language models. *arXiv*.

Anthropic. (2024). Mapping the mind of a large language model. *Anthropic Research*.

Qwen Team. (2024). Qwen2.5 technical report. *Alibaba*.
