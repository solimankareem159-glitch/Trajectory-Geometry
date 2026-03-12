# Research History & Evolution of Intuition

**Date:** February 2026
**Scope:** Experiment 01 - Experiment 13

This document outlines the evolution of intuition, theory, and empirical findings that led to the "Trajectory Geometry" framework. It traces the path from the initial search for static operator embeddings to the discovery of dynamic capability signatures.

---

## Phase 1: The Intuition of "Shapes" (Exp 01 - 02)

**Core Intuition:** "Cognitive moves (operators) like *Summarize* or *Critique* should have distinct geometric signatures in latent space."

### Experiment 01: Geometric Signatures (API)
- **Question:** Can we see operators in the embedding space of a closed API model?
- **Method:** Embed turn-level outputs of different operators.
- **Finding:** ✅ **Success**. Operators Cluster. Distinct traces found for "Summarize", "Critique", "Reframe".
- **Legacy:** Established that "thought has a shape."

### Experiment 02: Latent Factors (Local)
- **Question:** Are these shapes monolithic, or compositional?
- **Method:** Non-negative Matrix Factorization (NMF) on open-weights model hidden states.
- **Finding:** ✅ **Success**. Composite prompts (e.g., "Summarize + Critique") decompose into linear combinations of base factors.
- **Legacy:** Confirmed that the geometry is structured and potentially predictable.

---

## Phase 2: The "Invariant" Trap (Exp 03 - 07)

**Core Intuition:** "If these shapes are real, they must be *invariant* across layers and regimes. We can find a universal 'Summarize' vector."

### Experiment 03: Regime Invariants
- **Question:** Does the "Think" regime (internal reasoning) share the same geometry as the "Speak" regime (output)?
- **Method:** 3-Regime probing (Listen -> Think -> Speak) with cross-mapping.
- **Finding:** ❌ **Failure (Negative Result)**.
    - **Confounding:** Topic information leaked perfectly into operator probes ($Acc=1.0$).
    - **No Invariance:** Coupling maps between regimes failed ($R^2 < 0.07$). CCA found no shared space.
- **Lesson:** **"Thought" is not a static vector.** It transforms non-linearly. You cannot simply map "Listening" to "Thinking".

### Experiments 04 - 07: The Wilderness
- **Focus:** Searching for better models (MPT, early Qwen) and valid controls to escape the Exp 3 confounding.
- **Outcome:** Slow realization that *static* probing was a dead end. The signal is not in *where* the state is, but *how it moves*.

---

## Phase 3: The Pivot to Dynamics (Exp 08 - 09)

**Core Intuition:** "It's not the coordinates; it's the *trajectory*. Competence looks like a specific kind of motion."

### Experiment 08: Trajectory Geometry & Transitions
- **Question:** Can we define metrics for the *motion* of thought?
- **Method:** Defined differential geometry metrics: **Speed** (step size), **Curvature** (angle change), **Tortuosity**.
- **Finding:** Mixed. "Transition tests" (Exp 8') failed to find simple causal triggers, but the metrics proved stable and measurable.

### Experiment 09: Geometry-Capability Correlations (The Breakthrough)
- **Question:** Does the *geometry* predict *correctness*?
- **Method:** Monitor trajectories of **CoT Success (G4)** vs **Direct Failure (G1)**.
- **Findings:** ✅ **Major Success**.
    - **Speed**: Successful reasoning moves *faster* (larger steps) in middle layers ($d \approx 3.2$).
    - **Stabilization**: Successful reasoning *stabilizes* (slows down) before answering ($d \approx 1.1$).
    - **Directional Consistency (DC)**: Successful trajectories are "straighter" ($d \approx 2.6$).
- **Significance:** This was the first proof that internal geometry correlates with output correctness. It shifted the paradigm from classifying "intent" to monitoring "capability."

---

## Phase 4: The Dimensionality Nuance (Exp 11 - 13)

**Core Intuition:** "If CoT 'straightens' the path, does it also change the dimensionality of the space being explored?"

### Experiment 11: Extended Metrics (The "Richness" Discovery)
- **Question:** Is CoT just a straight line, or is it exploring a larger space?
- **Method:** Introduced **Effective Dimension** (PCA-based) and **Tortuosity**.
- **Finding:**
    - **CoT Success (G4)** has *much higher* dimensionality ($D_{eff} \approx 13$) than **Direct Failure (G1)** ($D_{eff} \approx 3$).
    - **Interpretation:** Failure often manifests as a "collapse" into low-dimensional repetitive loops. Success maintains a rich, high-dimensional representation.
    - **Nuance:** However, *within* CoT, Failure (G3) had even *higher* dimensionality than Success (G4), suggesting "wandering."

### Experiment 12: Advanced Diagnostics (The "Focus" Hypothesis)
- **Question:** Can we distinguish "rich exploration" (Success) from "aimless wandering" (CoT Failure)?
- **Method:** Applied **Fractal Dimension** and **Intrinsic Dimension (MLE)**.
- **Findigs:** ✅ **Confirmed "Focused Exploration"**.
    - **Intrinsic Dim:** G4 (Success) is lower than G3 (Failure), confirming that success confines itself to a relevant manifold.
    - **Convergence:** Successful trajectories exhibit linear convergence to a final state; failures diverge or loop.
    - **Fractal Dim:** Success is "smoother" (lower fractal dim) than failure.

### Experiment 13: Regime Mining & Subtyping
- **Question:** Can we blindly discover these types without labels?
- **Method:** Unsupervised clustering of trajectory metrics.
- **Finding:** ✅ **Subtypes Identified**.
    - **Type A (Stable-but-wrong)**: High speed, low tortuosity, but wrong answer. (Confident Hallucination).
    - **Type B (Failed Exploration)**: High effective dimension, wandering. (Confusion).
    - **Prediction:** Geometry alone predicts success with $AUC \approx 0.77$, outperforming prompt-type baselines.

---

## Summary of Evolution

| Phase | Key Question | Method | Verdict | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Signatures** | Does thought have a shape? | Clustering / NMF | ✅ Yes | Foundational |
| **2. Invariants** | Is that shape universal? | Linear Probing / CCA | ❌ No | **Invalidated** |
| **3. Dynamics** | Is the *motion* predictive? | Speed / Curvature | ✅ Yes | **Current Paradigm** |
| **4. Topology** | What is the *texture* of thought? | Fractal / Intrinsic Dim | ✅ Yes | State of the Art |

## Current Frontier

We are now investigating **Phase Transitions**: looking for the precise moment a trajectory "collapses" from high-dimensional exploration (Exp 11) to low-dimensional commitment (Exp 12).
