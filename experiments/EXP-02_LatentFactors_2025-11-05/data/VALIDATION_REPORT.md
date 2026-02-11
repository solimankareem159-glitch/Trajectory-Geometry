# Latent Factor Decomposition Experiment - Validation Report

## Executive Summary

**Status: PARTIALLY SUPPORTED WITH MAJOR CAVEATS**

The experiment correctly identifies that transformer hidden states show consistent dynamics (initial spike, then stability). However, the central claims about **"distinct geometric profiles"** for different operators are NOT supported by the figures—all operators produce nearly identical warp traces.

---

## 1. Claims vs. Evidence

### Claim 1: "Distinct Geometric Profiles"

> "Summarize induces a sharp, high-magnitude initial warp followed by stability, whereas Poetic and Reframe induce continuous, oscillating warp traces."

**Evidence from Figures:**

| Operator | Warp Trace Shape |
|:---|:---|
| Summarize | Sharp spike at t=0 (~1250), flat thereafter |
| Criticize | Sharp spike at t=0 (~1200), flat thereafter |
| Poetic | Sharp spike at t=0 (~1300), flat thereafter |
| Reframe Positively | Sharp spike at t=0 (~1200), flat thereafter |

**All traces are nearly identical in shape.** There is no visible "oscillating" pattern for Poetic or "gradual ramp" for Criticize as claimed.

**Verdict: ❌ NOT SUPPORTED**

The claim of distinct profiles appears to be post-hoc narrative fitting rather than empirically observed differentiation.

---

### Claim 2: "Layer Localization to Middle Layers"

**Evidence from Data:**

```json
"layer_sweep": {
  "0": 0.002,      // Input embeddings
  "5": 2.77,      // Early
  "12": 4.40,     // Middle
  "18": 19.97,    // Late
  "24": 1809.72   // Output
}
```

**Verdict: ✅ SUPPORTED**

The data clearly shows:

- Layer 0: Trivially low error (embedding lookup is deterministic)
- Layers 5-12: Moderate, stable error
- Layer 24: Catastrophic error (vocabulary projection)

This is a valid finding and is appropriately documented.

---

### Claim 3: "Compositionality via Linear Superposition"

> "Composite operators can be modeled as linear superpositions of latent geometric factors."

**Evidence Analysis:**

1. **The heatmap shows all factors active for all composites**:
   - "Summarize + Criticize" activates F1=5.42, F2=9.81, F3=2.56, F4=7.11
   - "Poetic + Criticize" activates F1=6.88, F2=11.23, F3=0.18, F4=5.91

   There's no clear pattern where specific operators → specific factors.

2. **Reconstruction looks good because all traces are the same**:
   The "Summarize + Criticize" reconstruction fits well because both the original and reconstruction have the same shape (spike at t=0, flat thereafter).

3. **NMF factors are trivial**:
   - Factor 1: Spike at t=0 (~175), then flat
   - Factor 3: Spike at t=0 (~75), then flat
   - Factors 2, 4: Essentially zero everywhere

   These factors just capture "how big is the initial spike"—not meaningful operator decomposition.

**Verdict: ⚠️ TRIVIALLY TRUE, NOT MEANINGFUL**

Any signal with "spike + flat" structure can be reconstructed from "spike + flat" basis vectors. This doesn't demonstrate that operators have compositional structure—it demonstrates that all operators have the same structure.

---

## 2. Methodological Issues

### 2.1 Sample Size

| Category | Count |
|:---|:---|
| Single operators | 10 |
| Composite operators | 5 |
| Base contexts | 1 ("Artificial intelligence...") |
| **Total samples** | **15** |

This is far too small to make claims about "operator invariants." The signatures may simply reflect the specific prompt phrasing.

### 2.2 What the "Warp" Actually Measures

The warp signal is:
$$W(t) = \|h_t - h_{t-1}\|_2$$

At t=0 (first token), this measures the transition from the embedding layer to the first hidden state—a massive transformation that occurs for **all** inputs. The subsequent flat line indicates that once the model has "processed" the initial context, further token-by-token changes are minimal.

This is a **universal property of transformer dynamics**, not an operator-specific signature.

### 2.3 PCA Shows Spread, Not Clusters

The span projection figure shows operators distributed across PC1/PC2 space. However:

- There's no tight clustering of similar operators (e.g., "Summarize" isn't near "Opposite Stance")
- The spread appears random, not structured by semantic similarity

---

## 3. What the Figures Actually Show

### Latent Factors Plot

![latent_factors.png description]

- Factor 1: Massive spike at position 0 (~175), then flat
- Factor 3: Smaller spike at position 0 (~75), then flat
- Factors 2, 4: Near-zero everywhere

**Interpretation**: The NMF is just learning "big spike" and "medium spike" as basis vectors.

### Warp Traces (All Operators)

All traces show:

1. Massive spike at t=0 (~1200-1300 magnitude)
2. Sharp drop to ~100-200 at t=1
3. Flat plateau near ~20-50 for remaining tokens

There is **no visible differentiation** between operators.

### Reconstruction Plots

The reconstructions fit well because the original and reconstructed signals have the same trivial structure.

---

## 4. Revised Interpretation

| Original Claim | Supported? | Corrected Interpretation |
|:---|:---|:---|
| "Distinct geometric profiles" | ❌ No | All operators produce identical spike-then-flat traces |
| "Layer localization" | ✅ Yes | Middle layers (5-12) show coherent geometry |
| "Linear compositionality" | ⚠️ Trivial | Reconstruction works because all traces are the same shape |
| "Trajectories in latent space" | ⚠️ Partial | There are trajectories, but they don't distinguish operators |

---

## 5. What the Experiment Successfully Demonstrates

Despite the overreach in conclusions, the experiment does validly show:

1. **Middle layers contain stable representations**: Layer sensitivity analysis is rigorous.

2. **Final layer geometry breaks down**: Expected due to vocabulary projection.

3. **Transformer dynamics have consistent structure**: The spike-then-flat pattern is reproducible.

4. **NMF can decompose warp signals**: The method is valid; the interpretation is overstated.

---

## 6. Recommendations

### For Valid Operator Differentiation

1. **Subtract the baseline**: Compute `warp(operator) - warp(baseline)` to isolate operator-specific signal.
2. **Use multiple base contexts**: Test same operator across 10+ topics to verify invariance.
3. **Look at later tokens**: The t=0 spike is universal; differentiation may occur later in generation.
4. **Use supervised probes**: Train a classifier to predict operator from hidden states (not unsupervised NMF).

### For Accurate Reporting

1. **Retract "distinct profiles" claim**: The figures don't support it.
2. **Qualify compositionality**: Note that reconstruction works because of trivial similarity, not meaningful decomposition.
3. **Emphasize what IS valid**: Layer localization finding is solid.

---

## 7. Conclusion

The Latent Factor Decomposition experiment has valid methodology but **overstated conclusions**. The core finding—that all operators produce the same warp signature (spike at t=0, then flat)—is the opposite of the claimed "distinct geometric profiles."

The experiment successfully demonstrates:

- How to extract and analyze transformer hidden states
- That middle layers are more stable than input/output layers
- That NMF can decompose signals into basis components

But it does NOT demonstrate:

- That different operators have different trajectories
- That compositional structure is semantically meaningful
- That operators are geometrically distinguishable
