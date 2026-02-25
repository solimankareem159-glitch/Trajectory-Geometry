# Trajectory Geometry: Defense of Core Claims

This document prepares the Principal Investigator (PI) to defend the core findings of this research program in academic contexts (interviews, viva, peer review).

## Core Claim 1: Dimensional Collapse in Failure

**The Finding**: When a model fails at reasoning, its hidden state trajectories collapse into a low-dimensional subspace ($\approx 3$ dimensions) compared to $\approx 13$ dimensions in success.

### Common Criticisms & Defenses

* **Critique**: "Isn't this just a result of shorter response lengths?"
  * **Defense**: No. Experiment 15 controlled for length. We measure the first 32 tokens (or use windowed normalization), and the effect persists. Even on identical token lengths, failed trajectories are ballistic (straight lines) while successful ones are expansive.
* **Critique**: "High dimensionality could just be noise."
  * **Defense**: If it were noise, it wouldn't correlate with correctness. Experiment 9 shows that the high-dimensional expansion is structured and predicts success better than any baseline (AUC 0.898).

---

## Core Claim 2: Success is Regime-Relative

**The Finding**: There is no universal "good" geometry. What looks like success for Chain-of-Thought (CoT) looks like failure for Direct answering (10/14 metrics flip sign).

### Common Criticisms & Defenses

* **Critique**: "Why would success look different?"
  * **Defense**: Because the computational tasks are different. Direct answering is a *retrieval* task (low dimensionality, high speed). CoT is a *computational* task (high dimensionality, controlled speed). A "good" retriever is a "bad" calculator.
* **Critique**: "How do you know which regime is active?"
  * **Defense**: Layers 0-7 function as a regime detector. The model internalizes the instruction (CoT vs Direct) and shifts its geometry before the first token is even output.

---

## Core Claim 3: Difficulty-Driven Expansion

**The Finding**: Models allocate representational volume proportional to problem complexity.

### Common Criticisms & Defenses

* **Critique**: "Maybe it's just the model getting 'confused' by more tokens."
  * **Defense**: Actually, 'overthinking' (EXP-15) shows that artificial expansion on EASY problems *hurts* performance. The model is most efficient when its geometric expansion matches the intrinsic difficulty of the task.

---

## Methodology Highlights

* **Permutation Testing**: All results use 10,000 shuffles; p < 0.001 is the standard.
* **Cross-Scale Stability**: Identical signatures found in models from 70M (Pythia) to 1.5B (Qwen) parameters.
* **Dynamic Metrics**: We move beyond "positions" to "velocity," "curvature," and "gyration."
